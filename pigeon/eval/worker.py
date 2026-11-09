import os
import queue
import sys
import multiprocessing as mp
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple
from functools import partial
import traceback
import signal

import numpy as np
from habitat.config.default import get_config
from habitat.core.env import Env
from habitat.tasks.nav.nav import NavigationEpisode
from pathlib import Path
import datasets
from datasets import Dataset

from ..utils.csv_logger import CSVLogger
from ..utils.video_writer import VideoWriterFactory
from ..utils.viz_manager import VizManager
from .eqa_utils import append_eqa_answer_json, episode_field, patch_eqa_habitat_config
from pigeon.utils.episode_stats_logger import (
    add_success_feasibility_suffix,
    determine_failure_cause,
    episodic_xy_to_habitat_xy,
)
from pigeon.utils.episode_trace_logger import EpisodeTraceLogger
from pigeon.utils.eval_randomness import seed_episode
from pigeon.models.vlm import VLMServiceUnavailable


def _redirect_stdio(log_path: Path) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_path, "a", buffering=1)
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        sys.stdout = f  # type: ignore[assignment]
        sys.stderr = f  # type: ignore[assignment]
    except Exception:
        pass


@dataclass
class WorkerContext:
    worker_id: int
    logger: CSVLogger
    video_writer_factory: VideoWriterFactory
    planner_name: str
    habitat_cfg: str
    realtime_viz: bool
    only_floor_transition: bool
    stop_event: Any  # multiprocessing.Event
    total_episodes: int
    model_request_q: Any  # mp.Queue
    model_reply_q: Any  # mp.Queue
    dense_view_sampling: bool = False
    final_check: bool = False
    example_dataset: Optional[Dataset] = None
    goal_proximity_thresh: Optional[float] = None
    trace_dir: Optional[str] = None


def make_planner(planner_name: str, data_queue: mp.Queue = None, env: Env = None) -> Any:
    # rather than import all planners at top-level, do it here to avoid
    # 'unable to create windowless context' issue
    from ..planner.random_planner import RandomPlanner
    from ..planner.random_frontier_planner import RandomFrontierPlanner
    from ..planner.random_object_planner import RandomObjectPlanner
    from ..planner.closest_object_planner import ClosestObjectPlanner
    from ..planner.valuemap_object_planner import ValueMapObjectPlanner
    from ..planner.random_fusion_planner import RandomFusionPlanner
    from ..planner.vlm_object_planner import VLMObjectPlanner
    from ..planner.vlm_fusion_planner import VLMFusionPlanner
    from ..planner.vlm_fusion_bev_planner import VLMFusionBEVPlanner
    from ..planner.vlm_fusion_bev_eqa import VLMFusionBEVEQAPlanner
    from ..planner.valuemap_fusion_planner import ValueMapFusionPlanner
    from ..planner.vlm_fusion_value_planner import VLMFusionValuePlanner

    name = (planner_name or "").strip().lower()
    if name in ("vlm object", "vlm_object", "vlm"):
        return VLMObjectPlanner()
    elif name in ("vfv"):
        return VLMFusionValuePlanner()
    elif name in ("valuemap object", "valuemap_object", "value_map object", "value_map_object"):
        return ValueMapObjectPlanner()
    elif name in ("closest object", "closest_object"):
        return ClosestObjectPlanner()
    elif name in ("random frontier", "random_frontier"):
        return RandomFrontierPlanner()
    elif name in ("random object", "random_object"):
        return RandomObjectPlanner()
    elif name in ("random fusion", "random_fusion"):
        return RandomFusionPlanner()
    elif name in ("vlm fusion", "vlm_fusion"):
        return VLMFusionPlanner()
    elif name in ("vlm fusion bev", "vlm_fusion_bev", "vfbev"):
        return VLMFusionBEVPlanner()
    elif name in ("vlm fusion bev eqa", "vlm_fusion_bev_eqa", "vfbev_eqa", "eqa"):
        return VLMFusionBEVEQAPlanner()
    elif name in ("valuemap fusion", "valuemap_fusion"):
        return ValueMapFusionPlanner()
    elif name in ("random",):
        return RandomPlanner()
    elif name in ("dummy"):
        from ..planner.dummy_planner import DummyPlanner

        return DummyPlanner(data_queue=data_queue, env=env)
    else:
        # default fallback
        return RandomPlanner()


class EvalWorker(mp.Process):
    def __init__(self, ctx: WorkerContext, episode_queue: "queue.Queue[object]") -> None:
        # Non-daemon to allow graceful teardown on Ctrl+C
        super().__init__(daemon=False)
        self.ctx = ctx
        self.episode_queue = episode_queue
        self._env = None
        self._win_name = f"pigeon-worker-{self.ctx.worker_id}"
        self._viz_manager: Optional[VizManager] = None
        self._active_trace: Optional[EpisodeTraceLogger] = None

    def _attach_object_cloud(self, info: Dict[str, Any]) -> None:
        """Attach target object cloud from planner's objmap for topdown viz when available."""
        objmap = getattr(self.planner, "_objmap", None)
        target_class = getattr(self.planner, "_target_class", None)
        if objmap is not None and target_class and hasattr(objmap, "has_object") and objmap.has_object(target_class):
            cloud = objmap.get_target_cloud(target_class)
            if isinstance(cloud, np.ndarray) and cloud.size > 0:
                info["object_point_cloud"] = cloud
                return
        info.pop("object_point_cloud", None)

    def _attach_stair_cloud(self, info: Dict[str, Any]) -> None:
        """Attach staircase cloud for topdown viz when available."""
        objmap = getattr(self.planner, "_objmap", None)
        if objmap is None or not hasattr(objmap, "has_object") or not hasattr(objmap, "get_target_cloud"):
            info.pop("stair_point_cloud", None)
            info.pop("stair_confidence", None)
            return
        try:
            has_stairs = objmap.has_object("staircases")
        except TypeError:
            has_stairs = objmap.has_object("staircases", True)
        except Exception:
            has_stairs = False
        if not has_stairs:
            info.pop("stair_point_cloud", None)
            info.pop("stair_confidence", None)
            return
        try:
            cloud = objmap.get_target_cloud("staircases")
        except TypeError:
            cloud = objmap.get_target_cloud("staircases", True)
        except Exception:
            info.pop("stair_point_cloud", None)
            info.pop("stair_confidence", None)
            return
        if isinstance(cloud, np.ndarray) and cloud.size > 0:
            info["stair_point_cloud"] = cloud
            confidence = self._get_stair_confidence(objmap)
            if confidence is not None:
                info["stair_confidence"] = confidence
            else:
                info.pop("stair_confidence", None)
            return
        info.pop("stair_point_cloud", None)
        info.pop("stair_confidence", None)

    @staticmethod
    def _get_stair_confidence(objmap: Any) -> Optional[float]:
        """Return the strongest staircase confidence when the map tracks it."""
        stair_objects = getattr(objmap, "_staircases", None)
        if not isinstance(stair_objects, list):
            return None

        best_confidence: Optional[float] = None
        for obj in stair_objects:
            if getattr(obj, "blacklisted", False):
                continue
            labels = getattr(obj, "labels", None)
            if not isinstance(labels, dict):
                continue
            track = labels.get("staircases")
            if track is None or getattr(track, "is_empty", False):
                continue
            try:
                confidence = float(getattr(track, "confidence", 0.0))
            except (TypeError, ValueError):
                continue
            confidence = max(0.0, min(1.0, confidence))
            if best_confidence is None or confidence > best_confidence:
                best_confidence = confidence
        return best_confidence

    @staticmethod
    def _trace_point(value: Any) -> Optional[list]:
        if value is None:
            return None
        try:
            array = value.get() if hasattr(value, "get") else np.asarray(value)
            array = np.asarray(array).reshape(-1)
            if array.size == 0:
                return None
            return [float(x) for x in array.tolist()]
        except Exception:
            return None

    @staticmethod
    def _trace_scalar(value: Any) -> Optional[float]:
        point = EvalWorker._trace_point(value)
        return None if not point else float(point[0])

    def _trace_sim_agent_state(self) -> Dict[str, Any]:
        """Capture simulator-frame pose; GPS intentionally omits floor height."""
        sim = getattr(self._env, "_sim", None)
        if sim is None:
            sim = getattr(self._env, "sim", None)
        if sim is None or not callable(getattr(sim, "get_agent_state", None)):
            return {}
        try:
            state = sim.get_agent_state()
        except Exception:
            return {}
        result = {
            "position": self._trace_point(getattr(state, "position", None)),
            "rotation": self._trace_point(getattr(state, "rotation", None)),
        }
        sensor_states = getattr(state, "sensor_states", None)
        if isinstance(sensor_states, dict):
            result["sensor_states"] = {
                str(name): {
                    "position": self._trace_point(getattr(sensor_state, "position", None)),
                    "rotation": self._trace_point(getattr(sensor_state, "rotation", None)),
                }
                for name, sensor_state in sensor_states.items()
            }
        return result

    @staticmethod
    def _trace_points(value: Any, limit: int = 64) -> list:
        if value is None:
            return []
        try:
            array = value.get() if hasattr(value, "get") else np.asarray(value)
            array = np.asarray(array)
            if array.ndim != 2 or array.shape[1] < 2:
                return []
            return array[:limit, :2].astype(float).tolist()
        except Exception:
            return []

    @staticmethod
    def _trace_mask_summary(value: Any) -> Dict[str, Any]:
        if value is None:
            return {"pixels": 0, "bbox_xyxy": None}
        try:
            mask = value.get() if hasattr(value, "get") else np.asarray(value)
            mask = np.asarray(mask).astype(bool)
            if mask.ndim != 2:
                return {"pixels": 0, "bbox_xyxy": None}
            rows, cols = np.where(mask)
            if rows.size == 0:
                return {"pixels": 0, "bbox_xyxy": None}
            return {
                "pixels": int(rows.size),
                "bbox_xyxy": [
                    int(cols.min()),
                    int(rows.min()),
                    int(cols.max()),
                    int(rows.max()),
                ],
            }
        except Exception:
            return {"pixels": None, "bbox_xyxy": None}

    @staticmethod
    def _trace_array_stats(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        try:
            array = np.asarray(value)
            result: Dict[str, Any] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            if array.size == 0:
                return result
            if np.issubdtype(array.dtype, np.number):
                finite = np.isfinite(array)
                result["finite_fraction"] = float(np.mean(finite))
                finite_values = array[finite]
                if finite_values.size:
                    result.update(
                        {
                            "min": float(np.min(finite_values)),
                            "max": float(np.max(finite_values)),
                            "mean": float(np.mean(finite_values)),
                            "std": float(np.std(finite_values)),
                        }
                    )
            return result
        except Exception:
            return {}

    def _trace_observation(self, rgb: Any, depth: Any) -> Dict[str, Any]:
        result = {
            "rgb": self._trace_array_stats(rgb),
            "depth": self._trace_array_stats(depth),
        }
        try:
            depth_array = np.asarray(depth)
            if depth_array.ndim == 3:
                depth_array = depth_array[..., 0]
            if depth_array.size:
                result["depth"]["zero_fraction"] = float(
                    np.mean(depth_array <= 0)
                )
                result["depth"]["max_normalized_fraction"] = float(
                    np.mean(depth_array >= 0.95)
                )
        except Exception:
            pass
        return result

    def _trace_detection_segmentation(self, det_seg: Any) -> Dict[str, Any]:
        if not isinstance(det_seg, dict):
            return {}
        classes = list(det_seg.get("classes") or [])
        scores = list(det_seg.get("scores") or [])
        boxes = list(det_seg.get("bboxes_px") or [])
        masks = list(det_seg.get("masks") or [])
        detections = []
        for idx in range(max(len(classes), len(scores), len(boxes), len(masks))):
            detections.append(
                {
                    "class": classes[idx] if idx < len(classes) else None,
                    "score": scores[idx] if idx < len(scores) else None,
                    "bbox_xyxy": boxes[idx] if idx < len(boxes) else None,
                    "mask": self._trace_mask_summary(masks[idx])
                    if idx < len(masks)
                    else None,
                }
            )
        return {"count": len(detections), "detections": detections}

    def _trace_map_state(self) -> Dict[str, Any]:
        omap = getattr(self.planner, "_omap", None)
        if omap is None:
            return {}

        def count(name: str) -> int:
            try:
                return int(np.count_nonzero(getattr(omap, name)))
            except Exception:
                return 0

        def points(name: str, limit: int = 64) -> list:
            return self._trace_points(getattr(omap, name, None), limit=limit)

        frustum = getattr(omap, "last_update_frustum_mask", None)
        camera_positions = getattr(omap, "_camera_positions", [])
        frontier_array = np.asarray(getattr(omap, "frontiers", []))
        frontier_count = (
            int(frontier_array.reshape(-1, 2).shape[0])
            if frontier_array.size >= 2 and frontier_array.size % 2 == 0
            else 0
        )
        result: Dict[str, Any] = {
            "robot_xy": self._trace_point(getattr(omap, "_latest_agent_xy", None)),
            "camera_yaw": getattr(omap, "_last_camera_yaw", None),
            "map_size": getattr(omap, "size", None),
            "pixels_per_meter": getattr(omap, "pixels_per_meter", None),
            "counts": {
                "explored": getattr(omap, "explored_area_pixels", 0),
                "steppable_explored": count("steppable_explored_area"),
                "obstacle": count("_map"),
                "real_obstacle": count("_real_map"),
                "navigable": count("_navigable_map"),
                "observe_obstacle": count("_observe_map"),
                "nonsteppable_obstacle": count("_nonsteppable_map"),
                "frustum": self._trace_mask_summary(frustum).get("pixels"),
                "exposed_boundary": getattr(omap, "exposed_boundary_pixels", 0),
                "frontier": frontier_count,
                "pending_downstair": count("_pending_downstair_map"),
                "downstair": count("_downstair_map"),
                "upstair": count("_upstair_map"),
            },
            "exploration_degree": getattr(omap, "exploration_degree", None),
            "view_mask_revision": getattr(omap, "view_mask_revision", -1),
            "frustum": self._trace_mask_summary(frustum),
            "frontiers": points("frontiers"),
            "pending_downstair_frontier": points("pending_downstair_frontier"),
            "downstair_frontier": points("downstair_frontier"),
            "upstair_frontier": points("upstair_frontier"),
            "downstair_view_facing": points("downstair_view_facing"),
            "upstair_view_facing": points("upstair_view_facing"),
            "trajectory_tail": self._trace_points(
                np.asarray(camera_positions[-64:])
                if isinstance(camera_positions, (list, tuple))
                and len(camera_positions) > 0
                else camera_positions,
                limit=64,
            ),
        }
        return result

    def _trace_planner_state(self) -> Dict[str, Any]:
        planner = self.planner
        vlm = getattr(planner, "_vlm", None)
        goal_selection = {}
        selection_fn = getattr(planner, "_trace_goal_selection_context", None)
        if callable(selection_fn):
            try:
                goal_selection = selection_fn()
            except Exception:
                goal_selection = {}
        objmap = getattr(planner, "_objmap", None)
        return {
            "phase": getattr(planner, "_phase", None),
            "init_steps": getattr(planner, "_init_steps", None),
            "tstep": getattr(planner, "_tstep", None),
            "layer_step_count": getattr(planner, "_layer_step_count", None),
            "current_xy": self._trace_point(getattr(planner, "_curr_xy", None)),
            "current_yaw": getattr(planner, "_curr_yaw", None),
            "camera_pitch_rad": getattr(planner, "_camera_pitch_rad", None),
            "camera_tf": getattr(planner, "_tf_camera_to_episodic", None),
            "goal_xy": self._trace_point(getattr(planner, "_goal_xy", None)),
            "goal_view_facing": self._trace_point(
                getattr(planner, "_goal_view_facing", None)
            ),
            "should_reset_pointnav": bool(
                getattr(planner, "_should_reset_pointnav", False)
            ),
            "banned_points": self._trace_points(
                getattr(planner, "_banned_points", None)
            ),
            "temporary_banned_points": self._trace_points(
                getattr(planner, "_tmp_ban_list", None)
            ),
            "stair_transition_active": bool(
                getattr(planner, "_stair_transition_active", False)
            ),
            "stair_transition_goal_sign": getattr(
                planner, "_stair_transition_goal_sign", None
            ),
            "downstair_detected": bool(getattr(planner, "downstair_detected", False)),
            "upstair_detected": bool(getattr(planner, "upstair_detected", False)),
            "stair_confidence": self._get_stair_confidence(objmap),
            "stair_frontier_goal_idx": getattr(planner, "stair_frontier_goal_idx", None),
            "stair_frontiers": self._trace_points(
                getattr(planner, "stair_frontiers", None)
            ),
            "stair_frontiers_view_facing": self._trace_points(
                getattr(planner, "stair_frontiers_view_facing", None)
            ),
            "stair_goal_vec": self._trace_points(
                getattr(planner, "stair_goal_vec", None)
            ),
            "stairnav_start": self._trace_point(
                getattr(planner, "_stairnav_start", None)
            ),
            "stairnav_failed": bool(getattr(planner, "_stairnav_failed", False)),
            "should_reset_stairnav": bool(
                getattr(planner, "should_reset_stairnav", False)
            ),
            "post_stair_goal_xy": self._trace_point(
                getattr(planner, "_post_stair_goal_xy", None)
            ),
            "stairnav_steps": getattr(planner, "_stairnav_steps", None),
            "stairnav_max_dist": getattr(planner, "_stairnav_max_dist", None),
            "ascended": getattr(planner, "ascended", None),
            "descended": getattr(planner, "descended", None),
            "stair_idx": getattr(planner, "stair_idx", None),
            "detector_output": getattr(planner, "_last_detector_output", None),
            "detection_filter": getattr(planner, "_last_detection_filter", None),
            "detector_routing": getattr(planner, "_last_detector_routing", None),
            "category_setup": getattr(planner, "_last_category_setup", None),
            "vlm_decision": getattr(planner, "_last_vlm_decision", None),
            "goal_selection": goal_selection,
            "decision_events": (
                planner._consume_trace_events()
                if callable(getattr(planner, "_consume_trace_events", None))
                else []
            ),
            "vlm_call_count": getattr(vlm, "call_count", None),
            "vlm_input_tokens": getattr(vlm, "input_token_count", None),
            "vlm_output_tokens": getattr(vlm, "output_token_count", None),
        }

    def _trace_habitat_state(self, info: Dict[str, Any]) -> Dict[str, Any]:
        top_down = info.get("top_down_map")
        top_down_summary: Dict[str, Any] = {}
        if isinstance(top_down, dict):
            top_down_summary = {
                "is_feasible": top_down.get("is_feasible"),
                "map_shape": list(np.asarray(top_down["map"]).shape)
                if top_down.get("map") is not None
                else None,
                "fog_of_war_pixels": self._trace_mask_summary(
                    top_down.get("fog_of_war_mask")
                ).get("pixels"),
                "target_bbox_pixels": self._trace_mask_summary(
                    top_down.get("target_bboxes_mask")
                ).get("pixels"),
                "tf_episodic_to_global": top_down.get("tf_episodic_to_global"),
                "upper_bound": top_down.get("upper_bound"),
                "lower_bound": top_down.get("lower_bound"),
                "grid_resolution": top_down.get("grid_resolution"),
            }
        collision = {}
        for key in ("collisions", "collision", "is_collision"):
            if key in info:
                collision[key] = info.get(key)
        return {
            "success": info.get("success"),
            "distance_to_goal": info.get("distance_to_goal"),
            "traveled_distance": info.get("traveled_distance"),
            "traveled_stairs": info.get("traveled_stairs"),
            "spl": info.get("spl"),
            "collision": collision,
            "episode_active": getattr(
                getattr(self._env, "_task", None), "is_episode_active", None
            ),
            "top_down_map": top_down_summary,
        }

    @staticmethod
    def _trace_episode_metadata(ep: NavigationEpisode) -> Dict[str, Any]:
        goals = []
        for goal in getattr(ep, "goals", []) or []:
            goal_record: Dict[str, Any] = {
                "object_id": getattr(goal, "object_id", None),
                "position": EvalWorker._trace_point(getattr(goal, "position", None)),
                "view_points": [],
            }
            for view_point in getattr(goal, "view_points", []) or []:
                agent_state = getattr(view_point, "agent_state", view_point)
                position = getattr(agent_state, "position", None)
                if position is None and isinstance(view_point, dict):
                    position = view_point.get("position")
                traced_position = EvalWorker._trace_point(position)
                if traced_position is not None:
                    goal_record["view_points"].append(traced_position)
            goals.append(goal_record)
        return {
            "start_position": EvalWorker._trace_point(
                getattr(ep, "start_position", None)
            ),
            "start_rotation": EvalWorker._trace_point(
                getattr(ep, "start_rotation", None)
            ),
            "goals": goals,
        }

    @staticmethod
    def _trace_nearest_goal_viewpoint(
        ep: NavigationEpisode, stop_position: Any
    ) -> Dict[str, Any]:
        """Compare the simulator-frame stop pose with every labeled viewpoint."""
        if stop_position is None:
            return {}
        try:
            stop = np.asarray(stop_position, dtype=np.float32).reshape(-1)
            if stop.size < 3:
                return {}
        except Exception:
            return {}

        nearest = None
        for goal_index, goal in enumerate(getattr(ep, "goals", []) or []):
            for viewpoint_index, view_point in enumerate(
                getattr(goal, "view_points", []) or []
            ):
                agent_state = getattr(view_point, "agent_state", view_point)
                position = getattr(agent_state, "position", None)
                if position is None and isinstance(view_point, dict):
                    position = view_point.get("position")
                try:
                    viewpoint = np.asarray(position, dtype=np.float32).reshape(-1)
                    if viewpoint.size < 3:
                        continue
                except Exception:
                    continue
                delta = stop[:3] - viewpoint[:3]
                record = {
                    "goal_index": int(goal_index),
                    "object_id": getattr(goal, "object_id", None),
                    "viewpoint_index": int(viewpoint_index),
                    "position": viewpoint[:3].astype(float).tolist(),
                    "distance_3d": float(np.linalg.norm(delta[:3])),
                    "distance_planar": float(np.linalg.norm(delta[[0, 2]])),
                    "height_delta": float(delta[1]),
                }
                if nearest is None or record["distance_3d"] < nearest["distance_3d"]:
                    nearest = record
        return nearest or {}

    def _trace_action(self, action_dict: Dict[str, Any], action: Any) -> Dict[str, Any]:
        pointnav = {
            key: action_dict.get(key)
            for key in (
                "pointnav_goal_rho_theta",
                "pointnav_reset",
                "pointnav_raw_action_id",
                "pointnav_raw_action",
                "pointnav_executed_action",
                "pointnav_stop_overridden",
                "pointnav_override_reason",
            )
            if key in action_dict
        }
        if "pointnav_executed_action" in pointnav:
            pointnav["pointnav_executed_action"] = action
        stairnav = {
            key: action_dict.get(key)
            for key in (
                "stairnav_goal_rho_theta",
                "stairnav_reset",
                "stairnav_raw_action_id",
                "stairnav_raw_action",
                "stairnav_executed_action",
                "stairnav_stop_overridden",
            )
            if key in action_dict
        }
        if "stairnav_executed_action" in stairnav:
            stairnav["stairnav_executed_action"] = action
        return {
            "action": action,
            "stop_called": bool(action_dict.get("stop_called", action == "stop")),
            "target_detected": bool(action_dict.get("target_detected", False)),
            "nav_goal": action_dict.get("nav_goal"),
            "forced_stop": bool(action_dict.get("forced_stop", False)),
            "forced_stop_reason": action_dict.get("forced_stop_reason"),
            "pre_forced_stop": action_dict.get("pre_forced_stop"),
            "eqa_answer": action_dict.get("eqa_answer"),
            "pointnav": pointnav,
            "stairnav": stairnav,
            "raw_action": action_dict.get("action"),
            "detection_segmentation": self._trace_detection_segmentation(
                action_dict.get("det_seg")
            ),
            "right_panel_text": action_dict.get("right_panel_text"),
        }

    def _trace_object_map(self) -> Dict[str, Any]:
        objmap = getattr(self.planner, "_objmap", None)
        if objmap is None:
            return {}

        target = getattr(self.planner, "_target_class", None)
        result: Dict[str, Any] = {
            "target_class": target,
            "candidate_names": list(getattr(objmap, "_candidate_object_names", [])),
            "suspected_ids": [
                int(getattr(obj, "id", -1))
                for obj in getattr(objmap, "_suspected_objects", [])
            ],
            "reliable_ids": [
                int(getattr(obj, "id", -1))
                for obj in getattr(objmap, "_reliable_objects", [])
            ],
            "candidate_view_positions": self._trace_points(
                getattr(objmap, "_candidate_view_positions", None)
            ),
            "last_nav_object_id": getattr(objmap, "last_nav_object_id", None),
            "current_image_id": getattr(objmap, "current_image_id", None),
            "map_step": getattr(objmap, "_step", None),
            "detections": list(getattr(objmap, "last_detection_records", [])),
            "update_diagnostics": getattr(objmap, "last_update_diagnostics", {}),
            "navigation_filter_records": list(
                getattr(objmap, "_last_navigation_filter_records", [])
            ),
            "confirmation_records": list(
                getattr(objmap, "last_confirmation_records", [])
            ),
            "image_detection_observations": [],
            "objects": [],
        }
        current_image_id = getattr(objmap, "current_image_id", None)
        image_history = getattr(objmap, "image_history", {})
        current_image = image_history.get(current_image_id) if current_image_id is not None else None
        if isinstance(current_image, dict):
            for detection in current_image.get("detections", []):
                result["image_detection_observations"].append(
                    {
                        key: detection.get(key)
                        for key in (
                            "detection_index",
                            "object_id",
                            "label",
                            "confidence",
                            "mask_shape",
                            "mask_bbox_xyxy",
                            "mask_pixels",
                            "accepted",
                            "vlm_response",
                            "vlm_context",
                            "view_ray_angle",
                        )
                    }
                )
        for obj in getattr(objmap, "_objects", []):
            labels = {}
            for name, track in getattr(obj, "labels", {}).items():
                pcd = getattr(track, "pcd", None)
                try:
                    points = pcd.get() if hasattr(pcd, "get") else np.asarray(pcd)
                    points = np.asarray(points)
                    if points.ndim != 2 or points.shape[0] == 0:
                        centroid = bounds = None
                    else:
                        centroid = self._trace_point(np.mean(points, axis=0))
                        bounds = [
                            self._trace_point(np.min(points, axis=0)),
                            self._trace_point(np.max(points, axis=0)),
                        ]
                    point_count = int(points.shape[0]) if points.ndim == 2 else 0
                except Exception:
                    centroid = bounds = None
                    point_count = 0
                if point_count or float(getattr(track, "confidence", 0.0)) > 0:
                    labels[str(name)] = {
                        "confidence": float(getattr(track, "confidence", 0.0)),
                        "volume": int(getattr(track, "volume", 0)),
                        "point_count": point_count,
                        "centroid": centroid,
                        "bounds": bounds,
                    }
            view_positions = np.asarray(
                getattr(obj, "view_positions", np.zeros((0, 2))), dtype=np.float32
            )
            valid = np.asarray(
                getattr(obj, "view_pos_valid", np.zeros((0,), dtype=bool)), dtype=bool
            )
            result["objects"].append(
                {
                    "id": int(getattr(obj, "id", -1)),
                    "best_label": getattr(obj, "best_label", ""),
                    "confirmed": bool(getattr(obj, "confirmed", False)),
                    "blacklisted": bool(getattr(obj, "blacklisted", False)),
                    "in_sight": bool(getattr(obj, "in_sight", False)),
                    "need_more_view_pos": bool(
                        getattr(obj, "need_more_view_pos", False)
                    ),
                    "confirmation_count": int(getattr(obj, "confirmation_count", 0)),
                    "true_confirmation_count": int(
                        getattr(obj, "true_confirmation_count", 0)
                    ),
                    "directional_confirmation_count": int(
                        getattr(obj, "directional_confirmation_count", 0)
                    ),
                    "navigation_fail_count": int(
                        getattr(obj, "navigation_fail_count", 0)
                    ),
                    "bbox_min": self._trace_point(getattr(obj, "bbox_min", None)),
                    "bbox_max": self._trace_point(getattr(obj, "bbox_max", None)),
                    "center": self._trace_point(getattr(obj, "center", None)),
                    "view_positions": view_positions.tolist()
                    if view_positions.ndim == 2
                    else [],
                    "view_pos_ray_angles": np.asarray(
                        getattr(obj, "view_pos_ray_angles", []), dtype=np.float32
                    ).tolist(),
                    "view_pos_ray_end_distances": np.asarray(
                        getattr(obj, "view_pos_ray_end_distances", []), dtype=np.float32
                    ).tolist(),
                    "ray_penetration_limit_m": float(
                        getattr(obj, "ray_penetration_limit_m", 0.0)
                    ),
                    "view_ray_traces": getattr(obj, "view_ray_traces", []),
                    "view_pos_is_unknown": np.asarray(
                        getattr(obj, "view_pos_is_unknown", []), dtype=bool
                    ).tolist(),
                    "view_pos_was_unknown": np.asarray(
                        getattr(obj, "view_pos_was_unknown", []), dtype=bool
                    ).tolist(),
                    "failed_unknown_view_positions": [
                        self._trace_point(point)
                        for point in getattr(obj, "failed_unknown_view_positions", [])
                    ],
                    "failed_unknown_view_ray_angles": [
                        float(angle)
                        for angle in getattr(obj, "failed_unknown_view_ray_angles", [])
                    ],
                    "exhausted_unknown_view_ray_angles": [
                        float(angle)
                        for angle in getattr(obj, "exhausted_unknown_view_ray_angles", [])
                    ],
                    "visited_view_ray_angles": [
                        float(angle)
                        for angle in getattr(obj, "visited_view_ray_angles", [])
                    ],
                    "attempted_view_ray_angles": [
                        float(angle)
                        for angle in getattr(obj, "attempted_view_ray_angles", [])
                    ],
                    "view_attempt_records": getattr(obj, "view_attempt_records", []),
                    "retry_unknown_view_ray_angle": getattr(
                        obj, "retry_unknown_view_ray_angle", None
                    ),
                    "valid_view_positions": int(valid.sum())
                    if valid.ndim == 1
                    else 0,
                    "unknown_view_positions": int(
                        np.asarray(getattr(obj, "view_pos_is_unknown", []), dtype=bool)[valid].sum()
                    )
                    if valid.ndim == 1
                    and len(np.asarray(getattr(obj, "view_pos_is_unknown", []))) == len(valid)
                    else 0,
                    "unknown_view_fail_count": int(
                        getattr(obj, "unknown_view_fail_count", 0)
                    ),
                    "visited_view_pos_count": int(
                        getattr(obj, "visited_view_pos_count", 0)
                    ),
                    "last_submitted_view_pos": getattr(
                        obj, "last_submitted_view_pos", None
                    ),
                    "last_submitted_view_ray_angle": getattr(
                        obj, "last_submitted_view_ray_angle", None
                    ),
                    "dense_view_sampling_attempted": bool(
                        getattr(obj, "dense_view_sampling_attempted", False)
                    ),
                    "image_ids": getattr(obj, "image_ids", None),
                    "goal_confirmed": bool(getattr(obj, "goal_confirmed", False)),
                    "goal_confirmation_rejected": bool(
                        getattr(obj, "goal_confirmation_rejected", False)
                    ),
                    "goal_confirmation_count": int(
                        getattr(obj, "goal_confirmation_count", 0)
                    ),
                    "labels": labels,
                }
            )
        return result

    def _write_trace_step(
        self,
        trace: EpisodeTraceLogger,
        ep_idx: int,
        ep_name: str,
        target: str,
        step: int,
        action_dict: Dict[str, Any],
        action: Any,
        pose_before: Dict[str, Any],
        pose_after: Dict[str, Any],
        observation: Dict[str, Any],
        action_input: Any,
        timing_delta: Dict[str, Any],
        info: Dict[str, Any],
    ) -> None:
        if not trace.enabled:
            return
        phase = getattr(self.planner, "_phase", None)
        goal = action_dict.get("nav_goal")
        vlm = getattr(getattr(self.planner, "_vlm", None), "last_response", None)
        goal_key = self._trace_point(goal)
        previous_vlm = getattr(self, "_trace_last_vlm", None)
        delta_3d = None
        vertical_delta = None
        translation_norm = None
        rotation_changed = None
        if (
            pose_after.get("sim", {}).get("position") is not None
            and pose_before.get("sim", {}).get("position") is not None
        ):
            before_position = np.asarray(
                pose_before["sim"]["position"], dtype=np.float32
            )
            after_position = np.asarray(
                pose_after["sim"]["position"], dtype=np.float32
            )
            if before_position.size >= 3 and after_position.size >= 3:
                delta_3d = (after_position[:3] - before_position[:3]).astype(float).tolist()
                vertical_delta = float(delta_3d[1])
                translation_norm = float(np.linalg.norm(delta_3d))
        if (
            pose_after.get("sim", {}).get("rotation") is not None
            and pose_before.get("sim", {}).get("rotation") is not None
        ):
            before_rotation = np.asarray(
                pose_before["sim"]["rotation"], dtype=np.float32
            )
            after_rotation = np.asarray(
                pose_after["sim"]["rotation"], dtype=np.float32
            )
            if before_rotation.shape == after_rotation.shape:
                rotation_changed = bool(
                    not np.allclose(before_rotation, after_rotation, atol=1e-5)
                )
        translation_blocked = bool(
            action == "move_forward"
            and translation_norm is not None
            and translation_norm < 0.01
        )
        self._trace_last_phase = phase
        self._trace_last_goal = goal_key
        self._trace_last_target = bool(action_dict.get("target_detected", False))
        self._trace_last_vlm = vlm
        trace.write(
            "step",
            episode_index=ep_idx,
            scene=ep_name,
            target=target,
            step=step,
            observation=observation,
            action=self._trace_action(action_dict, action),
            action_input=action_input,
            phase=phase,
            pose_before=pose_before,
            pose_after=pose_after,
            motion={
                "delta_xy": (
                    self._trace_point(
                        np.asarray(pose_after["gps"], dtype=np.float32)
                        - np.asarray(pose_before["gps"], dtype=np.float32)
                    )
                    if pose_after.get("gps") is not None
                    and pose_before.get("gps") is not None
                    else None
                ),
                "yaw_delta": (
                    float(pose_after["yaw"]) - float(pose_before["yaw"])
                    if pose_after.get("yaw") is not None
                    and pose_before.get("yaw") is not None
                    else None
                ),
                "delta_3d": delta_3d,
                "vertical_delta": vertical_delta,
                "translation_norm": translation_norm,
                "translation_blocked": translation_blocked,
                "rotation_changed": rotation_changed,
            },
            nav_goal=goal,
            target_detected=bool(action_dict.get("target_detected", False)),
            stop_called=bool(action == "stop"),
            forced_stop=bool(action_dict.get("forced_stop", False)),
            forced_stop_reason=action_dict.get("forced_stop_reason"),
            habitat=self._trace_habitat_state(info),
            planner={
                "banned_points": self._trace_point(
                    getattr(self.planner, "_banned_points", None)
                ),
                "goal_view_facing": self._trace_point(
                    getattr(self.planner, "_goal_view_facing", None)
                ),
                "frontiers": self._trace_points(
                    getattr(self.planner, "_frontiers", None)
                ),
                "frontier_memory": [
                    self._trace_point(entry.get("xy"))
                    for entry in getattr(self.planner, "_frontier_mem", [])
                    if isinstance(entry, dict) and entry.get("valid", False)
                ][:64],
            },
            vlm_response=vlm if vlm != previous_vlm else None,
            planner_state=self._trace_planner_state(),
            map_state=self._trace_map_state(),
            object_map=self._trace_object_map(),
            timing_delta=timing_delta,
        )

    def _attach_fusion_overlays(self, info: Dict[str, Any], target_class: Optional[str]) -> None:
        """Collect fusion suspects/reliables from planner's objmap and candidate view positions for viz."""
        objmap = getattr(self.planner, "_objmap", None)
        if objmap is None or getattr(objmap, "__class__", None).__name__ != "FusionPointCloudMap":
            info.pop("fusion_objects_viz", None)
            info.pop("suspect_view_positions", None)
            return

        fusion_objects = {"suspected": [], "reliable": []}
        view_positions = []
        view_visited = []

        buckets = [
            ("suspected", getattr(objmap, "_suspected_objects", [])),
            ("reliable", getattr(objmap, "_reliable_objects", [])),
        ]
        for bucket_name, objs in buckets:
            for obj in objs or []:
                labels = getattr(obj, "labels", {})
                track = labels.get(target_class) if isinstance(labels, dict) else None
                if track is None or getattr(track, "is_empty", False):
                    continue
                pts = getattr(track, "pcd", None)
                pts_np = pts.get() if hasattr(pts, "get") else np.asarray(pts)
                if not isinstance(pts_np, np.ndarray) or pts_np.size == 0:
                    continue
                if pts_np.shape[0] > 1500:
                    idx = np.random.choice(pts_np.shape[0], 1500, replace=False)
                    pts_np = pts_np[idx]
                label = getattr(obj, "best_label", target_class) or target_class
                fusion_objects[bucket_name].append(
                    {
                        "points": pts_np.astype(np.float32, copy=False),
                        "label": label,
                        "conf": obj.labels.get(label).confidence,
                    }
                )
                if bucket_name == "suspected":
                    vp = getattr(obj, "view_positions", None)
                    vv = getattr(obj, "view_pos_valid", None)
                    if vp is None or vv is None:
                        continue
                    vp_np = np.asarray(vp, dtype=np.float32)
                    vv_np = np.asarray(vv, dtype=bool)
                    if vp_np.ndim == 2 and vv_np.shape[0] == vp_np.shape[0]:
                        view_positions.append(vp_np)
                        view_visited.append(~vv_np)

        if any(fusion_objects[k] for k in fusion_objects):
            info["fusion_objects_viz"] = fusion_objects
        else:
            info.pop("fusion_objects_viz", None)
        if view_positions:
            info["suspect_view_positions"] = {
                "positions": np.concatenate(view_positions, axis=0),
                "visited": np.concatenate(view_visited, axis=0),
            }
        else:
            info.pop("suspect_view_positions", None)

    def _attach_vlm_text(self, info: Dict[str, Any]) -> None:
        """Expose last VLM response from planner for bottom text when no right panel is provided."""
        vlm_client = getattr(self.planner, "_vlm", None)
        resp = getattr(vlm_client, "last_response", None) if vlm_client is not None else None
        if isinstance(resp, str) and resp.strip():
            info["vlm_text"] = resp.strip()
        else:
            info.pop("vlm_text", None)

    def _build_status_text(self, ep_text: str, gps_status: Optional[str]) -> str:
        parts = [ep_text]
        planner_status_fn = getattr(self.planner, "get_status_text", None)
        if callable(planner_status_fn):
            try:
                planner_status = planner_status_fn()
            except Exception:
                planner_status = None
            if isinstance(planner_status, str) and planner_status.strip():
                parts.append(planner_status.strip())
        if gps_status:
            parts.append(gps_status)
        return " | ".join(parts)

    def _ensure_planner(self) -> None:
        if not hasattr(self, "planner") or self.planner is None:
            self.planner = make_planner(self.ctx.planner_name, getattr(self, "data_queue", None), self._env)
            if hasattr(self.planner, "set_final_check_enabled"):
                self.planner.set_final_check_enabled(self.ctx.final_check)

    @staticmethod
    def _requires_floor_transition(info: Dict[str, Any]) -> bool:
        top_down = info.get("top_down_map", {})
        if not isinstance(top_down, dict):
            return False
        return not bool(top_down.get("is_feasible", True))

    @staticmethod
    def _configure_planner_for_habitat_cfg(
        planner: Any,
        habitat_cfg: str,
        dense_view_sampling: bool = False,
        goal_proximity_thresh: Optional[float] = None,
    ) -> Optional[float]:
        habitat_cfg_lower = habitat_cfg.lower()
        is_hm3d_v2 = "hm3d.v2" in habitat_cfg_lower
        is_hm3d_ovon = "hm3d.ovon" in habitat_cfg_lower
        disable_multifloor = is_hm3d_v2 or is_hm3d_ovon

        # Optional CLI override for the object-goal stop distance. Only touches
        # _goal_proximity_thresh (STOP on a detected object); the separate
        # _proximity_threshold used for frontier arrival is left alone.
        if goal_proximity_thresh is not None and hasattr(planner, "_goal_proximity_thresh"):
            planner._goal_proximity_thresh = float(goal_proximity_thresh)

        planner.multifloor = not disable_multifloor
        planner._objmap.multifloor = not disable_multifloor
        planner.mp3d = "mp3d" in habitat_cfg_lower
        objmap = getattr(planner, "_objmap", None)
        if objmap is not None and hasattr(objmap, "dense_view_sampling_enabled"):
            objmap.dense_view_sampling_enabled = bool(dense_view_sampling)

        reliable_threshold_override = (
            0.6 if planner.mp3d or is_hm3d_ovon else None
        )

        if hasattr(planner, "_stair_query_exploration_threshold"):
            planner._stair_query_exploration_threshold = (
                float("inf") if disable_multifloor else 100.0
            )

        if hasattr(planner, "_stair_query_stuck_threshold"):
            if disable_multifloor:
                planner._stair_query_stuck_threshold = float("inf")
            elif "hm3d" in habitat_cfg_lower:
                planner._stair_query_stuck_threshold = 0.020
            else:
                planner._stair_query_stuck_threshold = 0.010
        return reliable_threshold_override

    def run(self) -> None:
        _redirect_stdio(Path("error_msg") / f"worker_{self.ctx.worker_id}_pid{os.getpid()}.log")
        hcfg = get_config(config_path=self.ctx.habitat_cfg)
        patch_eqa_habitat_config(hcfg)
        # disable dataset auto-loading
        self._env = Env(config=hcfg, dataset=self.ctx.example_dataset)
        self._ensure_planner()
        self._env._episode_from_iter_on_reset = False
        self._env._episode_force_changed = True
        # Initialize planner runtime once per worker (not per-episode)
        if hasattr(self.planner, "init_runtime"):
            sim_sensors = self._env._config.simulator.agents.main_agent.sim_sensors
            self._viz_manager = VizManager(
                window_name=self._win_name,
                enable_display=bool(self.ctx.realtime_viz and self.ctx.worker_id == 0),
            )
            self.planner.init_runtime(
                worker_id=self.ctx.worker_id,
                request_q=self.ctx.model_request_q,
                reply_q=self.ctx.model_reply_q,
                sim_sensors=sim_sensors,
                viz_manager=self._viz_manager,
            )
        # Simple prefetch loop: while idle, try to get next episodes quickly.
        try:
            while not self.ctx.stop_event.is_set():
                try:
                    ep = self.episode_queue.get(timeout=0.2)
                except queue.Empty:
                    if self.ctx.stop_event.is_set():
                        break
                    time.sleep(0.2)
                    continue
                except Exception as e:
                    print(f"Episode queue get() failed: {e}")
                    time.sleep(10)
                    continue
                try:
                    self._run_one(ep)
                except VLMServiceUnavailable as error:
                    if self._active_trace is not None:
                        self._active_trace.write("infrastructure_failure", error=repr(error), traceback=traceback.format_exc())
                        self._active_trace.close()
                        self._active_trace = None
                    traceback.print_exc()
                    self.ctx.stop_event.set()
                    os.kill(os.getppid(), signal.SIGTERM)
                    break
                except Exception:
                    if self._active_trace is not None:
                        self._active_trace.write(
                            "episode_exception",
                            error=repr(sys.exc_info()[1]),
                            traceback=traceback.format_exc(),
                        )
                        self._active_trace.close()
                        self._active_trace = None
                    traceback.print_exc()
                    if isinstance(sys.exc_info()[1], MemoryError):
                        self.ctx.stop_event.set()
                        os.kill(os.getppid(), signal.SIGTERM)
                        break
                finally:
                    self.episode_queue.task_done()
        finally:
            with Path(self.ctx.video_writer_factory.out_dir).parent.joinpath("err.log").open("a") as f:
                traceback.print_exc(file=f)
            if self._viz_manager is not None:
                self._viz_manager.close()
            # Explicitly close Habitat env to avoid C++ teardown at interpreter exit
            if self._env is not None:
                self._env.close()
                self._env = None
            # Release GPU client's shared-memory slots to reduce resource_tracker warnings
            try:
                gpu = getattr(self.planner, "_gpu", None)
                if gpu is not None and hasattr(gpu, "close"):
                    gpu.close()
            except Exception:
                pass
            # make sure to exit process to avoid hanging
            os._exit(0)

    def _run_one(self, episode: Tuple[int, NavigationEpisode]) -> None:

        t0 = time.time()
        ep_idx, ep = episode
        self._env.current_episode = ep
        evaluation_seed = seed_episode(ep_idx)
        if evaluation_seed is not None:
            self._env.seed(evaluation_seed % (2**32))
            gpu_client = getattr(self.planner, "_gpu", None)
            if gpu_client is not None:
                gpu_client.set_episode_seed(evaluation_seed)

        # Use global episode index for ID; show original episode name if available
        ep_id = f"{ep_idx}"
        ep_name = ep.scene_id.split("/")[-2]
        is_eqa = bool(getattr(self.planner, "is_eqa_planner", False))
        ep_goal = episode_field(ep, ("object_category",), fallback="")
        ep_question = episode_field(
            ep,
            ("question", "episode_question", "instruction", "object_category"),
            fallback=ep_goal,
        )
        ep_question_id = episode_field(ep, ("question_id", "episode_id"), fallback=ep_id)
        ep_ground_truth_answer = episode_field(
            ep,
            ("answer", "ground_truth_answer", "gt_answer"),
            fallback="",
        )
        if is_eqa:
            ep_text = f"episode: {ep_name} ({ep_idx}/{self.ctx.total_episodes}) question: {ep_question}"
        else:
            ep_text = f"episode: {ep_name} ({ep_idx}/{self.ctx.total_episodes}) goal: {ep_goal}"

        # Reset env first so ExplorationMap can report whether the target is
        # reachable on the start floor, matching count_required_stairs.py.
        obs = self._env.reset()
        info = self._env.get_metrics()  # type: ignore[attr-defined]
        if self.ctx.only_floor_transition and not self._requires_floor_transition(info):
            print(f"Skipping episode {ep_idx}: target reachable without floor transition.")
            return

        video = self.ctx.video_writer_factory.create(ep_id)
        trace = EpisodeTraceLogger(self.ctx.trace_dir, ep_id)
        self._active_trace = trace
        self._trace_last_phase = None
        self._trace_last_goal = None
        self._trace_last_target = False
        self._trace_last_vlm = None
        trace.write(
            "episode_start",
            episode_index=ep_idx,
            episode_id=ep_id,
            scene=ep_name,
            target=ep_question if is_eqa else ep_goal,
            worker_id=self.ctx.worker_id,
            planner=self.ctx.planner_name,
            habitat_cfg=self.ctx.habitat_cfg,
            evaluation_seed=evaluation_seed,
            vlm_temperature=os.environ.get("VLM_TEMPERATURE"),
            vlm_seed=os.environ.get("VLM_SEED"),
            max_steps=getattr(self._env._config.environment, "max_episode_steps", 500),
            episode=self._trace_episode_metadata(ep),
        )
        # Reset planner hidden state for the new episode
        reliable_threshold_override = self._configure_planner_for_habitat_cfg(
            self.planner,
            self.ctx.habitat_cfg,
            dense_view_sampling=self.ctx.dense_view_sampling,
            goal_proximity_thresh=self.ctx.goal_proximity_thresh,
        )
        self.planner.reset()
        if reliable_threshold_override is not None:
            self.planner._objmap.reliable_threshold = reliable_threshold_override
        # Provide target object to planners that support object goals
        if is_eqa and hasattr(self.planner, "set_episode_metadata"):
            self.planner.set_episode_metadata(question=ep_question, question_id=ep_question_id)
        if hasattr(self.planner, "set_target_object"):
            self.planner.set_target_object(ep_question if is_eqa else ep_goal)

        steps = 0
        success = False
        max_steps = getattr(self._env._config.environment, "max_episode_steps", 500)
        last_policy_info: Dict[str, Any] = {}
        last_agent_xy: Optional[Tuple[float, float]] = None
        last_frame_bgr = None
        gps_history = deque(maxlen=20)
        while steps < max_steps and not self.ctx.stop_event.is_set():
            rgb = obs.get("rgb")
            depth = obs.get("depth")
            gps = obs.get("gps")
            yaw = obs.get("compass") if obs.get("compass") is not None else obs.get("heading")
            pose_before = {
                "gps": self._trace_point(gps),
                "yaw": self._trace_scalar(yaw),
                "sim": self._trace_sim_agent_state(),
            }
            observation = self._trace_observation(rgb, depth)

            gps_status = None
            gps_delta_norm = None
            try:
                gps_arr = np.asarray(gps, dtype=np.float32).reshape(-1)
                if gps_arr.size >= 2:
                    gps_xy = gps_arr[:2].copy()
                    gps_history.append(gps_xy)
                    gps_avg = np.mean(np.stack(gps_history, axis=0), axis=0)
                    gps_delta = gps_xy - gps_avg
                    gps_delta_norm = float(np.linalg.norm(gps_delta))
            except Exception:
                gps_status = None

            need_viz = (self.ctx.realtime_viz and self.ctx.worker_id == 0) or (video is not None)
            if need_viz and self._viz_manager is not None:
                target_class = getattr(self.planner, "_target_class", None)
                self._attach_object_cloud(info)
                self._attach_stair_cloud(info)
                self._attach_fusion_overlays(info, target_class)
                self._attach_vlm_text(info)
                status_text = self._build_status_text(ep_text, gps_status)
                self._viz_manager.update_observation(
                    rgb=rgb,
                    info=info,
                    depth=depth,
                    steps=steps,
                    max_steps=max_steps,
                    ep_text=status_text,
                )

            if self._env._task.is_episode_active is False:  # multi-floor signal set by exploration_map measurement
                action_dict = {"map_viz": None, "det_seg": None, "value_map_viz": None}
                action = "stop"
                timing_delta = {}
            else:
                try:
                    action_dict = self.planner.act(
                        obs={
                            "rgb": rgb,
                            "depth": depth,
                            "gps": gps,
                            "yaw": yaw,
                            "gps_delta_norm": gps_delta_norm,
                            "eval_step": steps,
                            "max_steps": max_steps,
                        }
                    )
                except Exception as exc:
                    trace.write(
                        "step_exception",
                        episode_index=ep_idx,
                        scene=ep_name,
                        target=ep_question if is_eqa else ep_goal,
                        step=steps,
                        stage="planner.act",
                        error=repr(exc),
                        traceback=traceback.format_exc(),
                        pose=pose_before,
                        observation=observation,
                        planner_state=self._trace_planner_state(),
                        map_state=self._trace_map_state(),
                    )
                    raise
                timing_delta = {}
            action = action_dict.get("action", "stop")

            # Record the decision before stepping Habitat, then attach the
            # resulting metrics below so each line describes one transition.
            if need_viz:
                if video is not None and self._viz_manager is not None:
                    for buffered_frame in self._viz_manager.drain_rendered_frames():
                        video.write(buffered_frame)
                        last_frame_bgr = buffered_frame
                target_class = getattr(self.planner, "_target_class", None)
                self._attach_object_cloud(info)
                self._attach_stair_cloud(info)
                self._attach_fusion_overlays(info, target_class)
                self._attach_vlm_text(info)
                status_text = self._build_status_text(ep_text, gps_status)
                if self._viz_manager is not None:
                    self._viz_manager.update_observation(
                        rgb=rgb,
                        info=info,
                        depth=depth,
                        steps=steps,
                        max_steps=max_steps,
                        ep_text=status_text,
                    )

                color_bev_viz = None
                if hasattr(self.planner, "get_color_bev_viz"):
                    try:
                        color_bev_viz = self.planner.get_color_bev_viz()
                    except Exception:
                        color_bev_viz = None
                height_bev_viz = None
                if hasattr(self.planner, "get_height_bev_viz"):
                    try:
                        height_bev_viz = self.planner.get_height_bev_viz()
                    except Exception:
                        height_bev_viz = None
                val_map_panel = action_dict.get("value_map_viz") if not "right_panel_viz" in action_dict else None
                bottom_text_value = None
                if action_dict.get("right_panel_viz") is not None:
                    bottom_text_value = action_dict.get("right_panel_text", "")
                elif isinstance(info.get("vlm_text"), str):
                    bottom_text_value = info.get("vlm_text")
                annotated = None
                if video is not None and self._viz_manager is not None:
                    for buffered_frame in self._viz_manager.drain_rendered_frames():
                        video.write(buffered_frame)
                        last_frame_bgr = buffered_frame
                if self._viz_manager is not None:
                    annotated = self._viz_manager.update_panels(
                        planner_viz=action_dict.get("map_viz"),
                        det_seg=action_dict.get("det_seg"),
                        value_map_viz=val_map_panel,
                        aux_map_viz=color_bev_viz,
                        height_map_viz=height_bev_viz,
                        right_panel_viz=action_dict.get("right_panel_viz"),
                        bottom_text=bottom_text_value,
                        render=True,
                    )
                if video is not None and annotated is not None:
                    # Write the annotated frame so videos show step count too
                    video.write(annotated)
                    # Remember last frame for trailing hold
                    last_frame_bgr = annotated

            # Remember where the agent stood on this step, in the same episodic
            # frame the planner uses (see _set_agent_pose: y is negated), so it
            # can be converted to habitat world coords for offline analysis.
            try:
                gps_arr_pos = np.asarray(gps, dtype=np.float64).reshape(-1)
                if gps_arr_pos.size >= 2:
                    last_agent_xy = (float(gps_arr_pos[0]), -float(gps_arr_pos[1]))
            except Exception:
                pass

            # Track policy info needed for failure cause analysis
            last_policy_info = {
                "nav_goal": action_dict.get("nav_goal", None),
                "target_detected": bool(action_dict.get("target_detected", False)),
                "stop_called": bool(action == "stop"),
                "forced_stop": bool(action_dict.get("forced_stop", False)),
                "forced_stop_reason": action_dict.get("forced_stop_reason", None),
                "pre_forced_stop": action_dict.get("pre_forced_stop", None),
            }

            if isinstance(action, str):
                step_in = {"action": action}
            else:
                step_in = action
            try:
                obs = self._env.step(step_in)
            except Exception as exc:
                trace.write(
                    "step_exception",
                    episode_index=ep_idx,
                    scene=ep_name,
                    target=ep_question if is_eqa else ep_goal,
                    step=steps,
                    stage="env.step",
                    error=repr(exc),
                    traceback=traceback.format_exc(),
                    pose=pose_before,
                    observation=observation,
                    action=self._trace_action(action_dict, action),
                    action_input=step_in,
                    planner_state=self._trace_planner_state(),
                    map_state=self._trace_map_state(),
                )
                raise

            # Update metrics and check success
            info = self._env.get_metrics()  # type: ignore[attr-defined]
            info["agent_position"] = self._trace_point(obs.get("gps"))
            pose_after = {
                "gps": self._trace_point(obs.get("gps")),
                "yaw": self._trace_scalar(
                    obs.get("compass")
                    if obs.get("compass") is not None
                    else obs.get("heading")
                ),
                "sim": self._trace_sim_agent_state(),
            }
            self._write_trace_step(
                trace,
                ep_idx,
                ep_name,
                ep_question if is_eqa else ep_goal,
                steps,
                action_dict,
                action,
                pose_before,
                pose_after,
                observation,
                step_in,
                timing_delta,
                info,
            )
            # Habitat reports success as 1.0/0.0 in metrics
            s = info.get("success", 0)
            success = bool(s if isinstance(s, bool) else (float(s) > 0.0))
            if is_eqa and str(action_dict.get("eqa_answer") or "").strip():
                break
            if success and not is_eqa:
                break
            steps += 1

            if self._env._task.is_episode_active is False:  # multi-floor signal set by exploration_map measurement
                break

        # Determine if this episode was aborted due to external stop
        aborted = self.ctx.stop_event.is_set()

        # Ensure video resources are finalized promptly to avoid memory leaks
        if video is not None:
            # Extend the last frame for ~1 second at the end
            if last_frame_bgr is not None:
                for _ in range(int(getattr(video, "fps", 10))):
                    video.write(last_frame_bgr)
            video.close()

        if aborted:
            trace.write("episode_aborted", steps=steps)
            trace.close()
            self._active_trace = None
            # Only persist results for completed episodes (success or reached max_steps)
            if hasattr(self.planner, "clear_mapping_obs_history"):
                self.planner.clear_mapping_obs_history()
            return

        # Refresh metrics at episode end for accurate logging/filenames
        final_info = self._env.get_metrics()  # type: ignore[attr-defined]
        s_val = final_info.get("success", 0)
        sr_int = int(s_val)
        spl_val = final_info.get("spl", 0)
        spl_float = float(spl_val)
        traveled = float(final_info.get("traveled_distance", 0.0))
        distance_to_goal_raw = final_info.get("distance_to_goal")
        distance_to_goal = (
            float(distance_to_goal_raw) if distance_to_goal_raw is not None else None
        )

        elapsed = time.time() - t0
        # Determine failure cause using vlfm's logic + pigeon planner signals
        failure_cause = "did_not_fail" if sr_int == 1 else "Unknown"
        if sr_int == 0:
            infos_for_cause = dict(final_info)
            # Ensure top_down_map exists for analysis; if absent, leave Unknown
            if isinstance(infos_for_cause.get("top_down_map"), dict):
                infos_for_cause.update(last_policy_info)
                # Use GPU transform for robustness when available
                gpu = getattr(self.planner, "_gpu", None)
                failure_cause = determine_failure_cause(infos_for_cause, gpu=gpu)

        # Convert the final agent position and the goal it was pursuing into
        # habitat world coordinates, so offline analysis can compare them
        # against the dataset's goal view_points (same frame).
        gpu_for_tf = getattr(self.planner, "_gpu", None)
        stop_habitat_xy = episodic_xy_to_habitat_xy(last_agent_xy, final_info, gpu=gpu_for_tf)
        nav_goal_habitat_xy = episodic_xy_to_habitat_xy(
            last_policy_info.get("nav_goal"), final_info, gpu=gpu_for_tf
        )
        final_sim_state = self._trace_sim_agent_state()
        nearest_goal_viewpoint = self._trace_nearest_goal_viewpoint(
            ep, final_sim_state.get("position")
        )

        # Episode-level stats: VLM call count (per-episode) and number of banned points.
        vlm_calls = 0
        vlm_input_tokens = 0
        vlm_output_tokens = 0
        try:
            vlm = getattr(self.planner, "_vlm", None)
            if vlm is not None and hasattr(vlm, "call_count"):
                vlm_calls = int(getattr(vlm, "call_count", 0))
                vlm_input_tokens = int(getattr(vlm, "input_token_count", 0))
                vlm_output_tokens = int(getattr(vlm, "output_token_count", 0))
        except Exception:
            vlm_calls = 0
            vlm_input_tokens = 0
            vlm_output_tokens = 0

        banned_points = 0
        try:
            banned_arr = getattr(self.planner, "_banned_points", None)
            if banned_arr is not None and hasattr(banned_arr, "shape") and banned_arr.ndim >= 1:
                banned_points = int(banned_arr.shape[0])
        except Exception:
            banned_points = 0

        self.ctx.logger.append(
            episode_id=ep_id,
            worker_id=self.ctx.worker_id,
            steps=steps,
            success=sr_int,
            spl=spl_float,
            elapsed_sec=elapsed,
            failure_cause=failure_cause,
            traveled_distance=traveled,
            distance_to_goal=distance_to_goal,
            downstair_detected=bool(getattr(self.planner, "downstair_detected", False)),
            upstair_detected=bool(getattr(self.planner, "upstair_detected", False)),
            vlm_calls=vlm_calls,
            vlm_input_tokens=vlm_input_tokens,
            vlm_output_tokens=vlm_output_tokens,
            banned_points=banned_points,
            scene=ep_name,
            target=ep_question if is_eqa else ep_goal,
            stop_x=None if stop_habitat_xy is None else stop_habitat_xy[0],
            stop_y=None if stop_habitat_xy is None else stop_habitat_xy[1],
            nav_goal_x=None if nav_goal_habitat_xy is None else nav_goal_habitat_xy[0],
            nav_goal_y=None if nav_goal_habitat_xy is None else nav_goal_habitat_xy[1],
        )
        trace.write(
            "episode_end",
            episode_index=ep_idx,
            scene=ep_name,
            target=ep_question if is_eqa else ep_goal,
            steps=steps,
            success=sr_int,
            spl=spl_float,
            distance_to_goal=distance_to_goal,
            traveled_distance=traveled,
            failure_cause=failure_cause,
            stop_position=stop_habitat_xy,
            stop_position_3d=final_sim_state.get("position"),
            stop_rotation=final_sim_state.get("rotation"),
            nearest_goal_viewpoint=nearest_goal_viewpoint,
            nav_goal=nav_goal_habitat_xy,
        )
        trace.close()
        self._active_trace = None

        if is_eqa and hasattr(self.planner, "get_eqa_result"):
            try:
                eqa_result = self.planner.get_eqa_result()
            except Exception:
                eqa_result = {}
            question_id = str(eqa_result.get("question_id") or ep_question_id or ep_id)
            question = str(eqa_result.get("question") or ep_question)
            answer = str(eqa_result.get("answer") or "")
            proposed_target = str(eqa_result.get("proposed_target") or "")
            ground_truth_answer = str(eqa_result.get("ground_truth_answer") or ep_ground_truth_answer)
            append_eqa_answer_json(
                Path(self.ctx.video_writer_factory.out_dir).parent,
                question_id,
                question,
                answer,
                proposed_target=proposed_target,
                ground_truth_answer=ground_truth_answer,
                path_length=traveled,
            )

        # Rename video to include SR and SPL if we wrote one
        if video is not None:
            old_path = Path(video.path)
            scene = ep_name.split('-')
            scene = scene[1] if len(scene) > 1 else scene[0]
            video_failure_cause = add_success_feasibility_suffix(failure_cause, final_info)
            new_name = f"{scene}_{ep_id}_sr{sr_int}_spl{spl_float:.3f}_{video_failure_cause}.mp4"
            new_path = old_path.with_name(new_name)
            if old_path.exists():
                old_path.rename(new_path)
        if hasattr(self.planner, "clear_mapping_obs_history"):
            self.planner.clear_mapping_obs_history()


class DataGenWorker(EvalWorker):
    def __init__(
        self, ctx: WorkerContext, episode_queue: "queue.Queue[object]", data_queue: "queue.Queue[object]"
    ) -> None:
        super().__init__(ctx, episode_queue)
        self.data_queue = data_queue


class DatasetWorker(mp.Process):
    def __init__(self, data_queue: "queue.Queue[object]", path: str) -> None:
        # Non-daemon to allow graceful teardown on Ctrl+C
        super().__init__(daemon=False)
        self.data_queue = data_queue
        self._queue = queue.Queue()
        self.save_path = path
        self.filler_thread = threading.Thread(target=self.filler)

    # When Dataset.from_generator serializes the generator function, it cannot
    # serialize the mp.Queue object directly, so we use this static method to
    # fill the internal queue, and let data_generator read from it.
    def filler(self):
        while True:
            item = self.data_queue.get()
            self._queue.put(item)

    @staticmethod
    def data_generator(queue):
        while True:
            data_item = queue.get()
            if data_item is None:
                break
            yield data_item

    def run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        datasets.config.DEFAULT_MAX_BATCH_SIZE = 100
        self.filler_thread.start()
        func = partial(self.data_generator, self._queue)
        dataset = Dataset.from_generator(func, cache_dir=Path(self.save_path).parent.as_posix())
        dataset.save_to_disk(self.save_path)
        os._exit(0)
