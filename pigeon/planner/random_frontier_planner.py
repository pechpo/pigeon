import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

import cv2
import numpy as np
import cupy as cp
from ..mapping.obstacle_map import ObstacleMap
from ..mapping.color_topdown_map import ColorTopDownMap
from ..mapping.heightmap import HeightMap
from ..utils.shared_memory import ShmArray, attach_ndarray
from .gpu_client import GPUWorkerClient
from pigeon.utils.geometry_utils import rho_theta, xyz_yaw_pitch_to_tf_matrix

_UNSET = object()


class RandomFrontierPlanner:
    actions = ("move_forward", "turn_left", "turn_right", "stop", "look_up", "look_down")
    models = ["pointnav_rnn"]

    def __init__(self) -> None:
        # Two-phase navigation: initial spin then explore
        self._phase = "init"
        self._init_steps = 0
        self._max_init_steps = 12
        self._frontiers: np.ndarray = np.array([])
        self._goal_xy: np.ndarray = None
        self._curr_xy: np.ndarray = None
        self._curr_yaw: float = 0.0
        self._camera_pitch_rad: float = 0.0
        self._pitch_step_rad: float = float(np.deg2rad(30.0))
        self._pending_after_level: Optional[Dict[str, Any]] = None
        self._tf_camera_to_episodic: Optional[np.ndarray] = None
        self._req_id = 0
        self._should_reset_pointnav = False
        self._mapping_obs_history: List[Dict[str, Any]] = []
        self._stair_transition_active = False
        self._stair_transition_goal_sign = 0
        self._stair_transition_update_mask: Optional[np.ndarray] = None
        self._post_stair_goal_xy: Optional[np.ndarray] = None
        self._post_stair_goal_sign = 0
        self._post_stair_flat_height_tol_m = 0.08
        self._post_stair_directional_score_min_m = 0.02
        self._post_stair_settle_steps = 0
        self._post_stair_settle_max_steps: Optional[int] = None

        # Obstacle map
        self._omap = ObstacleMap()
        self._default_min_height = float(self._omap._min_height)
        self._enable_color_bev = os.environ.get("PIGEON_COLOR_BEV", "0") == "1"
        self._color_bev: Optional[ColorTopDownMap] = None
        self._height_bev: Optional[HeightMap] = None

        # Runtime wiring (set by init_runtime)
        self._worker_id: int = None
        self._request_q = None
        self._reply_q = None
        self._viz_manager = None
        self._proximity_threshold = 1.0
        self._ban_threshold = 0.5
        self._banned_points: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._layer_step_count = 0
        # Short-lived decision history consumed by the evaluator after every step.
        # The full state is still logged separately; these events preserve the
        # reason and before/after values for decisions that would otherwise be
        # overwritten on the next step.
        self._trace_events: List[Dict[str, Any]] = []
        self._last_goal_selection_source: Optional[str] = None

        # Shared-memory slots
        self._depth_slot = ShmArray()
        self._goal_slot = ShmArray()
        self._mask_slot = ShmArray()
        self._tf_slot = ShmArray()
        self._points_slot = ShmArray()

        # GPU worker client (set in init_runtime)
        self._gpu: GPUWorkerClient

    @property
    def downstair_detected(self) -> bool:
        return bool(getattr(self._omap, "downstair_detected", False))

    @property
    def upstair_detected(self) -> bool:
        return bool(getattr(self._omap, "upstair_detected", False))

    def init_runtime(
        self,
        worker_id: int,
        request_q: Any,
        reply_q: Any,
        sim_sensors: Any,
        viz_manager: Any = None,
    ) -> None:
        self._worker_id = worker_id
        self._request_q = request_q
        self._reply_q = reply_q
        self._viz_manager = viz_manager
        camera_height = sim_sensors.rgb_sensor.position[1]
        min_depth = sim_sensors.depth_sensor.min_depth
        max_depth = sim_sensors.depth_sensor.max_depth
        camera_hfov_deg = sim_sensors.depth_sensor.hfov
        self._camera_height = float(camera_height)
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._camera_fov_rad = float(np.deg2rad(camera_hfov_deg))
        if hasattr(self._omap, "set_observe_min_height"):
            self._omap.set_observe_min_height(self._camera_height - 0.10)
        # Create GPU worker client
        self._gpu = GPUWorkerClient(worker_id=self._worker_id, request_q=self._request_q, reply_q=self._reply_q)
        if self._enable_color_bev:
            self._color_bev = ColorTopDownMap(
                size=self._omap.size,
                pixels_per_meter=self._omap.pixels_per_meter,
                episode_pixel_origin=self._omap._episode_pixel_origin,
            )
            self._height_bev = HeightMap(
                size=self._omap.size,
                pixels_per_meter=self._omap.pixels_per_meter,
                episode_pixel_origin=self._omap._episode_pixel_origin,
            )


    def _visualize_map(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._omap.visualize(*args, **kwargs)

    def _choose_goal_with_trace(self, *args: Any, **kwargs: Any) -> Any:
        before_goal = self._trace_xy(self._goal_xy)
        before_phase = self._phase
        result = self._choose_goal(*args, **kwargs)
        self._record_trace_event(
            "goal_selection",
            chooser="_choose_goal",
            source=self._last_goal_selection_source or "unknown",
            before_goal=before_goal,
            after_goal=self._trace_xy(self._goal_xy),
            phase_before=before_phase,
            phase_after=self._phase,
            frontier_count=self._count_frontiers(),
            reset_pointnav=bool(self._should_reset_pointnav),
            **self._trace_goal_selection_context(),
        )
        self._last_goal_selection_source = None
        return result

    def _choose_goal_vlm_with_trace(self, *args: Any, **kwargs: Any) -> Any:
        before_goal = self._trace_xy(self._goal_xy)
        before_phase = self._phase
        result = self._choose_goal_vlm(*args, **kwargs)
        self._record_trace_event(
            "goal_selection",
            chooser="_choose_goal_vlm",
            source=self._last_goal_selection_source or "unknown",
            before_goal=before_goal,
            after_goal=self._trace_xy(self._goal_xy),
            phase_before=before_phase,
            phase_after=self._phase,
            frontier_count=self._count_frontiers(),
            suspected_viewpoint_count=self._count_suspected_viewpoints(),
            reset_pointnav=bool(self._should_reset_pointnav),
            **self._trace_goal_selection_context(),
        )
        self._last_goal_selection_source = None
        return result

    @staticmethod
    def _trace_xy(value: Any) -> Optional[List[float]]:
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            return arr[:2].tolist() if arr.size >= 2 else None
        except Exception:
            return None

    def _count_frontiers(self) -> int:
        count = 0
        if isinstance(self._frontiers, np.ndarray) and self._frontiers.ndim == 2:
            count += int(len(self._frontiers))
        count += sum(1 for entry in getattr(self, "_frontier_mem", []) if entry.get("valid", False))
        return count

    def _count_suspected_viewpoints(self) -> int:
        objmap = getattr(self, "_objmap", None)
        if objmap is None:
            return 0
        count = 0
        for obj in getattr(objmap, "_suspected_objects", []):
            if self._target_class and self._target_class not in getattr(obj, "labels", {}):
                continue
            valid = np.asarray(getattr(obj, "view_pos_valid", []), dtype=bool)
            count += int(valid.sum()) if valid.ndim == 1 else 0
        return count

    def _trace_goal_selection_context(self) -> Dict[str, Any]:
        """Snapshot object candidates so a selected viewpoint is auditable."""
        objmap = getattr(self, "_objmap", None)
        if objmap is None:
            return {}

        target_class = getattr(self, "_target_class", None)
        goal = self._goal_xy if isinstance(self._goal_xy, np.ndarray) else None
        candidates: List[Dict[str, Any]] = []
        seen_ids = set()
        objects = list(getattr(objmap, "_suspected_objects", []))
        objects.extend(getattr(objmap, "_reliable_objects", []))
        for obj in objects:
            object_id = getattr(obj, "id", None)
            if object_id in seen_ids:
                continue
            seen_ids.add(object_id)
            labels = getattr(obj, "labels", {})
            track = labels.get(target_class) if isinstance(labels, dict) else None
            if track is None or getattr(track, "is_empty", False):
                continue
            view_positions = np.asarray(
                getattr(obj, "view_positions", np.zeros((0, 2))),
                dtype=np.float32,
            )
            valid = np.asarray(
                getattr(obj, "view_pos_valid", np.zeros((0,), dtype=bool)),
                dtype=bool,
            )
            submitted = getattr(obj, "last_submitted_view_pos", None)
            view_index = (
                int(submitted)
                if submitted is not None
                and view_positions.ndim == 2
                and 0 <= int(submitted) < len(view_positions)
                else None
            )
            view_xy = (
                view_positions[view_index].astype(float).tolist()
                if view_index is not None
                else None
            )
            distance_to_goal = None
            if view_xy is not None and goal is not None and goal.size >= 2:
                distance_to_goal = float(
                    np.linalg.norm(np.asarray(view_xy) - goal.reshape(-1)[:2])
                )
            candidates.append(
                {
                    "object_id": int(object_id) if object_id is not None else None,
                    "best_label": getattr(obj, "best_label", None),
                    "confirmed": bool(getattr(obj, "confirmed", False)),
                    "blacklisted": bool(getattr(obj, "blacklisted", False)),
                    "confidence": float(getattr(track, "confidence", 0.0)),
                    "true_confirmation_count": int(
                        getattr(obj, "true_confirmation_count", 0)
                    ),
                    "bbox_min": (
                        np.asarray(getattr(obj, "bbox_min"), dtype=float).reshape(-1).tolist()
                        if getattr(obj, "bbox_min", None) is not None
                        else None
                    ),
                    "bbox_max": (
                        np.asarray(getattr(obj, "bbox_max"), dtype=float).reshape(-1).tolist()
                        if getattr(obj, "bbox_max", None) is not None
                        else None
                    ),
                    "center": (
                        np.asarray(getattr(obj, "center"), dtype=float).reshape(-1).tolist()
                        if getattr(obj, "center", None) is not None
                        else None
                    ),
                    "view_index": view_index,
                    "view_xy": view_xy,
                    "view_is_valid": (
                        bool(valid[view_index])
                        if view_index is not None
                        and valid.ndim == 1
                        and view_index < len(valid)
                        else None
                    ),
                    "distance_to_goal": distance_to_goal,
                    "valid_view_position_count": int(valid.sum())
                    if valid.ndim == 1
                    else 0,
                }
            )

        selected_id = getattr(objmap, "last_nav_object_id", None)
        if selected_id is None and candidates and goal is not None:
            matching = [
                candidate
                for candidate in candidates
                if candidate.get("distance_to_goal") is not None
            ]
            if matching:
                selected_id = min(
                    matching, key=lambda candidate: candidate["distance_to_goal"]
                ).get("object_id")
        selected = next(
            (candidate for candidate in candidates if candidate["object_id"] == selected_id),
            None,
        )
        return {
            "selected_object_id": selected_id,
            "selected_object": selected,
            "candidate_objects": candidates[:32],
        }

    def _record_trace_event(self, event_type: str, **payload: Any) -> None:
        """Buffer compact decision events until the worker writes this step."""
        record: Dict[str, Any] = {
            "type": str(event_type),
            "step": int(getattr(self, "_tstep", getattr(self, "_layer_step_count", 0))),
        }
        record.update(payload)
        self._trace_events.append(record)
        if len(self._trace_events) > 128:
            del self._trace_events[:-128]

    def _consume_trace_events(self) -> List[Dict[str, Any]]:
        events = self._trace_events
        self._trace_events = []
        return events

    def _update_viz(
        self,
        *,
        planner_viz: Any = _UNSET,
        det_seg: Any = _UNSET,
        value_map_viz: Any = _UNSET,
        aux_map_viz: Any = _UNSET,
        height_map_viz: Any = _UNSET,
        right_panel_viz: Any = _UNSET,
        bottom_text: Any = _UNSET,
        refresh: bool = False,
    ) -> Any:
        if self._viz_manager is None:
            return None
        kwargs: Dict[str, Any] = {"render": refresh}
        if planner_viz is not _UNSET:
            kwargs["planner_viz"] = planner_viz
        if det_seg is not _UNSET:
            kwargs["det_seg"] = det_seg
        if value_map_viz is not _UNSET:
            kwargs["value_map_viz"] = value_map_viz
        if aux_map_viz is not _UNSET:
            kwargs["aux_map_viz"] = aux_map_viz
        if height_map_viz is not _UNSET:
            kwargs["height_map_viz"] = height_map_viz
        elif self._height_bev is not None:
            kwargs["height_map_viz"] = self.get_height_bev_viz()
        if right_panel_viz is not _UNSET:
            kwargs["right_panel_viz"] = right_panel_viz
        if bottom_text is not _UNSET:
            kwargs["bottom_text"] = bottom_text
        return self._viz_manager.update_panels(**kwargs)

    def _refresh_detection_viz(
        self,
        *,
        det_seg: Any = _UNSET,
        planner_viz: Any = _UNSET,
        value_map_viz: Any = _UNSET,
        aux_map_viz: Any = _UNSET,
        height_map_viz: Any = _UNSET,
        right_panel_viz: Any = _UNSET,
        bottom_text: Any = _UNSET,
    ) -> Any:
        if self._viz_manager is None:
            return None
        kwargs: Dict[str, Any] = {"refresh": True}
        if planner_viz is _UNSET:
            kwargs["planner_viz"] = self._visualize_map()
        else:
            kwargs["planner_viz"] = planner_viz
        if det_seg is not _UNSET:
            kwargs["det_seg"] = det_seg
        if value_map_viz is not _UNSET:
            kwargs["value_map_viz"] = value_map_viz
        if aux_map_viz is not _UNSET:
            kwargs["aux_map_viz"] = aux_map_viz
        if height_map_viz is not _UNSET:
            kwargs["height_map_viz"] = height_map_viz
        if right_panel_viz is not _UNSET:
            kwargs["right_panel_viz"] = right_panel_viz
        if bottom_text is not _UNSET:
            kwargs["bottom_text"] = bottom_text
        return self._update_viz(**kwargs)

    def reset(self, reset_stairs=True) -> None:
        self._phase = "init"
        self._init_steps = 0
        self._frontiers = np.array([])
        self._goal_xy = None
        self._curr_xy = None
        self._curr_yaw = 0.0
        self._camera_pitch_rad = 0.0
        self._pending_after_level = None
        self._tf_camera_to_episodic = None
        self._req_id = 0
        self.clear_mapping_obs_history()
        target_min_height = 0.78 if getattr(self, "mp3d", False) else self._default_min_height
        if hasattr(self._omap, "set_min_height"):
            self._omap.set_min_height(target_min_height)
        self._omap.reset()
        # Ensure pointnav RNN state resets at episode start
        self._should_reset_pointnav = True
        # Clear banned points each episode
        self._banned_points = np.zeros((0, 2), dtype=np.float32)
        self._layer_step_count = 0
        self._trace_events = []
        self._last_goal_selection_source = None
        self._stairnav_steps = 0
        self._stairnav_start = None
        self._stairnav_max_dist = 0
        self._stairnav_failed = False
        self._descent_goal_xy = None
        self._descent_entry_xy = None
        self.stair_frontier_goal_idx = None
        self._stair_transition_active = False
        self._stair_transition_goal_sign = 0
        self._stair_transition_update_mask = None
        self._post_stair_goal_xy = None
        self._post_stair_goal_sign = 0
        self._post_stair_settle_steps = 0
        if self._color_bev is not None:
            self._color_bev.reset()
        if self._height_bev is not None:
            self._height_bev.reset()
        if reset_stairs:
            self.ascended = False
            self.descended = False
            stair_rec = [np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 1))]
            self.stair_idx = 0
            self.stair_recs = [deepcopy(stair_rec) for _ in range(10)]
            self.should_reset_stairnav = True

    @property
    def mapping_obs_history(self) -> List[Dict[str, Any]]:
        return self._mapping_obs_history

    def clear_mapping_obs_history(self) -> None:
        self._mapping_obs_history.clear()

    def _start_stair_transition_tracking(self, goal_sign: int) -> None:
        goal_sign = int(np.sign(goal_sign))
        if goal_sign == 0:
            return
        if self._height_bev is None:
            self._stair_transition_active = True
            self._stair_transition_goal_sign = goal_sign
            self._stair_transition_update_mask = None
            return
        if (
            self._stair_transition_update_mask is None
            or self._stair_transition_update_mask.shape != (self._height_bev.size, self._height_bev.size)
            or self._stair_transition_goal_sign != goal_sign
        ):
            self._stair_transition_update_mask = np.zeros((self._height_bev.size, self._height_bev.size), dtype=bool)
        self._stair_transition_active = True
        self._stair_transition_goal_sign = goal_sign

    def _clear_stair_transition_tracking(self) -> None:
        self._stair_transition_active = False
        self._stair_transition_goal_sign = 0
        self._stair_transition_update_mask = None

    def _clear_post_stair_goal(self) -> None:
        self._post_stair_goal_xy = None
        self._post_stair_goal_sign = 0
        self._post_stair_settle_steps = 0

    def _accumulate_stair_transition_updates(self) -> None:
        if not self._stair_transition_active or self._height_bev is None:
            return
        update_mask = getattr(self._height_bev, "_last_update_mask", None)
        if not isinstance(update_mask, np.ndarray) or update_mask.shape != (self._height_bev.size, self._height_bev.size):
            return
        if self._stair_transition_update_mask is None or self._stair_transition_update_mask.shape != update_mask.shape:
            self._stair_transition_update_mask = np.zeros_like(update_mask, dtype=bool)
        self._stair_transition_update_mask |= update_mask.astype(bool, copy=False)

    def get_status_text(self) -> str:
        exploration_degree = getattr(self._omap, "exploration_degree", 0.0)
        explored_area_pixels = int(getattr(self._omap, "explored_area_pixels", 0))
        exposed_boundary_pixels = int(getattr(self._omap, "exposed_boundary_pixels", 0))
        stuck_degree = self._get_stuck_degree()
        exploration_degree_str = "inf" if np.isinf(exploration_degree) else f"{exploration_degree:.2f}"
        phase = self._get_phase_status_text()
        return (
            f"state: {phase}"
            f" | exploration_degree: {exploration_degree_str} ({explored_area_pixels}/{exposed_boundary_pixels})"
            f" | stuck_degree: {stuck_degree:.4f}"
        )

    def _get_phase_status_text(self) -> str:
        phase = str(getattr(self, "_phase", "unknown") or "unknown").strip().lower()
        phase_labels = {
            "init": "init scan",
            "spin": "spin",
            "explore": "explore",
            "escape": "escape",
            "target": "nav to target",
            "nav_to_stair": "nav to stair",
            "pointnav_descent": "pointnav descent",
            "stair_nav": "stair nav",
            "post_stair_settle": "post stair settle",
        }
        return phase_labels.get(phase, phase.replace("_", " "))

    def _get_stuck_degree(self) -> float:
        explored_area_pixels = int(getattr(self._omap, "explored_area_pixels", 0))
        if explored_area_pixels <= 0:
            return 0.0
        return float(self._layer_step_count) / float(explored_area_pixels)

    def _start_layer_step(self) -> None:
        self._layer_step_count += 1

    def _camera_is_level(self) -> bool:
        return abs(float(getattr(self, "_camera_pitch_rad", 0.0))) < 1e-4

    def _build_action_result(
        self,
        action: str,
        extra: Optional[Dict[str, Any]] = None,
        stop_called: Optional[bool] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "action": action,
            "map_viz": self._visualize_map(),
            "nav_goal": self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None,
            "stop_called": bool(action == "stop") if stop_called is None else bool(stop_called),
            "target_detected": False,
        }
        if extra:
            out.update(extra)
        return out

    def _queue_camera_action(
        self,
        action: str,
        pitch_rad: float,
        *,
        extra: Optional[Dict[str, Any]] = None,
        pending_after_level: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._camera_pitch_rad = float(pitch_rad)
        self._pending_after_level = pending_after_level
        return self._build_action_result(action, extra=extra, stop_called=False)

    def _apply_pending_after_level(self) -> None:
        pending = getattr(self, "_pending_after_level", None)
        if pending is None or not self._camera_is_level():
            return
        self._pending_after_level = None
        kind = str(pending.get("kind", ""))
        if kind == "finish_stair_transition":
            self._finish_stair_transition(int(pending.get("goal_sign", 0)))
            return
        if kind == "resume_stair_nav":
            self._phase = "stair_nav"
            stairnav_start = pending.get("stairnav_start")
            if stairnav_start is None:
                self._stairnav_start = None
            else:
                self._stairnav_start = np.asarray(stairnav_start, dtype=np.float32).reshape(2)
            return
        if kind == "start_post_stair_settle":
            goal_sign = int(pending.get("goal_sign", 0))
            if isinstance(self._post_stair_goal_xy, np.ndarray) and self._post_stair_goal_xy.size == 2:
                self._begin_post_stair_settle(goal_sign)
            else:
                self._finish_stair_transition(goal_sign)
            return
        if kind == "resume_explore":
            self._phase = "explore"
            self._goal_xy = None
            self._descent_goal_xy = None
            self._descent_entry_xy = None
            self.stair_frontier_goal_idx = None
            self._clear_stair_transition_tracking()
            self._clear_post_stair_goal()
            if hasattr(self, "_goal_view_facing"):
                self._goal_view_facing = None
            self._should_reset_pointnav = True
            self._omap.set_selected_frontier(None)

    def _set_agent_pose(self, gps: np.ndarray, yaw: float) -> None:
        self._apply_pending_after_level()
        x, y = float(gps[0]), float(gps[1])
        camera_position = np.array([x, -y, self._camera_height], dtype=np.float32)
        self._curr_xy = camera_position[:2].copy()
        self._curr_yaw = float(yaw)
        self._tf_camera_to_episodic = xyz_yaw_pitch_to_tf_matrix(
            camera_position,
            self._curr_yaw,
            float(self._camera_pitch_rad),
        )

    def _remember_mapping_observation(self, rgb: Optional[np.ndarray]) -> None:
        if rgb is None:
            return
        frustum_mask = getattr(self._omap, "last_update_frustum_mask", None)
        if not isinstance(frustum_mask, np.ndarray) or frustum_mask.size == 0:
            return
        record = {
            "frustum_mask": frustum_mask.astype(bool, copy=True),
            "frustum_revision": int(getattr(self._omap, "view_mask_revision", -1)),
            "robot_xy": self._curr_xy.copy() if isinstance(self._curr_xy, np.ndarray) else None,
            "robot_yaw": float(self._curr_yaw),
            "layer_idx": int(getattr(self, "stair_idx", 0)),
        }
        objmap = getattr(self, "_objmap", None)
        image_id = getattr(objmap, "current_image_id", None)
        image_history = getattr(objmap, "image_history", {})
        if image_id is not None and image_id in image_history:
            record["image_id"] = int(image_id)
        else:
            # Keep standalone frontier planners working when no object image cache exists.
            record["rgb"] = rgb.copy()
            record["agent_tf"] = (
                self._tf_camera_to_episodic.copy()
                if isinstance(self._tf_camera_to_episodic, np.ndarray)
                else None
            )
        self._mapping_obs_history.append(record)

    @property
    def stair_frontiers(self):
        return self.stair_recs[self.stair_idx + 5][0]

    @stair_frontiers.setter
    def stair_frontiers(self, value):
        self.stair_recs[self.stair_idx + 5][0] = value

    @property
    def stair_frontiers_view_facing(self):
        return self.stair_recs[self.stair_idx + 5][1]

    @stair_frontiers_view_facing.setter
    def stair_frontiers_view_facing(self, value):
        self.stair_recs[self.stair_idx + 5][1] = value

    @property
    def stair_goal_vec(self):
        return self.stair_recs[self.stair_idx + 5][2]

    @stair_goal_vec.setter
    def stair_goal_vec(self, value):
        self.stair_recs[self.stair_idx + 5][2] = value

    def _maybe_spin(self, extra=None, post_spin_phase="explore", orientation="turn_left"):
        if self._phase == "spin":
            self._steps_to_spin -= 1
            if self._steps_to_spin <= 0:
                self._phase = post_spin_phase
                self._choose_goal_vlm_with_trace(allow_spin=False)
                self._xy_history_start = self._tstep
                self._xy_history[self._tstep % len(self._xy_history)] = self._curr_xy
            out = {
                "action": orientation,
                "map_viz": self._visualize_map(),
                "right_panel_viz": self._vlm_grid_bgr,  # dedicated VLM panel
                "right_panel_text": self._get_bottom_text(),
                "nav_goal": self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None,
                "stop_called": False,
            }
            if extra:
                out.update(extra)
            return out
        return None

    def _valid_post_stair_goal_px(self, goal_px: np.ndarray) -> bool:
        if goal_px is None:
            return False
        goal_px = np.asarray(goal_px, dtype=np.int32).reshape(-1)
        if goal_px.size != 2:
            return False
        if self._height_bev is None:
            return False
        x, y = int(goal_px[0]), int(goal_px[1])
        if x < 0 or x >= self._height_bev.size or y < 0 or y >= self._height_bev.size:
            return False
        if not bool(self._height_bev._valid[y, x]) or not np.isfinite(self._height_bev._height[y, x]):
            return False
        goal_xy = self._omap._px_to_xy(goal_px.reshape(1, 2))[0]
        if self._omap.is_occupied_xy(goal_xy):
            return False
        if hasattr(self._omap, "is_near_stairs") and self._omap.is_near_stairs(goal_xy, radius_m=0.15):
            return False
        return True


    def _compute_post_stair_ground_goal(self, goal_sign: int = 0) -> np.ndarray:
        if self._height_bev is None:
            return np.array([])
        goal_sign = int(np.sign(goal_sign if goal_sign != 0 else self._stair_transition_goal_sign))
        transition_mask = self._stair_transition_update_mask
        if not isinstance(transition_mask, np.ndarray) or transition_mask.shape != (self._height_bev.size, self._height_bev.size):
            return np.array([])
        candidate = transition_mask & self._height_bev._valid & np.isfinite(self._height_bev._height)
        if not np.any(candidate):
            return np.array([])

        height_map = self._height_bev._height
        visited = np.zeros_like(candidate, dtype=bool)
        flat_tol_m = float(getattr(self, "_post_stair_flat_height_tol_m", 0.08))
        direction_threshold_m = float(getattr(self, "_post_stair_directional_score_min_m", 0.02))
        best_rows = np.zeros((0,), dtype=np.int32)
        best_cols = np.zeros((0,), dtype=np.int32)
        best_size = 0
        best_directional_score = -np.inf
        best_component_score = -np.inf

        rows, cols = np.where(candidate)
        for start_r, start_c in zip(rows.tolist(), cols.tolist()):
            if visited[start_r, start_c]:
                continue
            stack = [(int(start_r), int(start_c))]
            visited[start_r, start_c] = True
            component_rows: List[int] = []
            component_cols: List[int] = []
            while stack:
                r, c = stack.pop()
                component_rows.append(r)
                component_cols.append(c)
                curr_h = float(height_map[r, c])
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if nr < 0 or nr >= candidate.shape[0] or nc < 0 or nc >= candidate.shape[1]:
                        continue
                    if visited[nr, nc] or not candidate[nr, nc]:
                        continue
                    if abs(float(height_map[nr, nc]) - curr_h) > flat_tol_m:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            component_size = len(component_rows)
            component_height = float(
                np.median(height_map[np.asarray(component_rows, dtype=np.int32), np.asarray(component_cols, dtype=np.int32)])
            )
            if goal_sign == 0:
                better_component = component_size > best_size
            else:
                directional_score = float(goal_sign) * component_height
                if directional_score <= direction_threshold_m:
                    continue
                component_score = directional_score * float(component_size)
                better_component = component_score > best_component_score
                if not better_component and np.isclose(component_score, best_component_score, rtol=1e-6, atol=1e-6):
                    better_component = component_size > best_size
                    if not better_component and component_size == best_size:
                        better_component = directional_score > best_directional_score
            if better_component:
                best_size = component_size
                if goal_sign != 0:
                    best_directional_score = directional_score
                    best_component_score = component_score
                best_rows = np.asarray(component_rows, dtype=np.int32)
                best_cols = np.asarray(component_cols, dtype=np.int32)

        if best_size <= 0:
            return np.array([])

        median_px = np.array(
            [
                int(np.rint(np.median(best_cols.astype(np.float32)))),
                int(np.rint(np.median(best_rows.astype(np.float32)))),
            ],
            dtype=np.int32,
        )
        dist_sq = (best_cols.astype(np.float32) - float(median_px[0])) ** 2 + (
            best_rows.astype(np.float32) - float(median_px[1])
        ) ** 2
        for idx in np.argsort(dist_sq):
            goal_px = np.array([best_cols[int(idx)], best_rows[int(idx)]], dtype=np.int32)
            if self._valid_post_stair_goal_px(goal_px):
                return self._omap._px_to_xy(goal_px.reshape(1, 2))[0]
        return np.array([])

    def _begin_post_stair_settle(self, goal_sign: int) -> None:
        goal_sign = int(np.sign(goal_sign if goal_sign != 0 else self._post_stair_goal_sign))
        if not isinstance(self._post_stair_goal_xy, np.ndarray) or self._post_stair_goal_xy.size != 2:
            return
        self._post_stair_goal_sign = goal_sign
        self._phase = "post_stair_settle"
        self._goal_xy = self._post_stair_goal_xy.astype(np.float32, copy=True)
        self._omap.set_selected_frontier(self._goal_xy)
        self._should_reset_pointnav = True
        self._post_stair_settle_steps = 0

    def _resume_explore_after_post_stair_settle_timeout(self, goal_sign: int) -> None:
        goal_sign = int(np.sign(goal_sign))
        if goal_sign != 0:
            self._finish_stair_transition(goal_sign)
        self._clear_post_stair_goal()
        self._clear_stair_transition_tracking()
        self._phase = "explore"
        self._init_steps = getattr(self, "_max_init_steps", getattr(self, "_init_steps", 0))
        self._goal_xy = None
        self._descent_goal_xy = None
        self._descent_entry_xy = None
        self._pending_after_level = None
        self.stair_frontier_goal_idx = None
        if hasattr(self, "_goal_view_facing"):
            self._goal_view_facing = None
        self._should_reset_pointnav = True
        self._omap.set_selected_frontier(None)

    def _get_post_stair_fallback_goal(self) -> np.ndarray:
        if not isinstance(self._curr_xy, np.ndarray) or self._curr_xy.size != 2:
            return np.array([])

        direction = None
        stairnav_start = getattr(self, "_stairnav_start", None)
        if isinstance(stairnav_start, np.ndarray) and stairnav_start.size == 2:
            delta = self._curr_xy.astype(np.float32, copy=False) - stairnav_start.astype(np.float32, copy=False)
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > 1e-3:
                direction = delta / delta_norm

        if direction is None:
            yaw = float(getattr(self, "_curr_yaw", 0.0))
            direction = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)

        settle_dist_m = max(float(getattr(self, "_proximity_threshold", 1.0)) + 0.25, 1.25)
        return (self._curr_xy.astype(np.float32, copy=False) + direction.astype(np.float32) * settle_dist_m).astype(
            np.float32
        )

    def _handle_successful_stair_transition(
        self,
        depth: np.ndarray,
        extra: Optional[Dict[str, Any]],
        goal_sign: int,
    ) -> Optional[Dict[str, Any]]:
        goal_sign = int(np.sign(goal_sign))
        post_goal_xy = self._compute_post_stair_ground_goal(goal_sign)
        if not (isinstance(post_goal_xy, np.ndarray) and post_goal_xy.size == 2):
            post_goal_xy = self._get_post_stair_fallback_goal()
        if isinstance(post_goal_xy, np.ndarray) and post_goal_xy.size == 2:
            self._post_stair_goal_xy = post_goal_xy.astype(np.float32)
            self._post_stair_goal_sign = goal_sign
            if not self._camera_is_level():
                return self._queue_camera_action(
                    "look_up",
                    0.0,
                    extra=extra,
                    pending_after_level={"kind": "start_post_stair_settle", "goal_sign": goal_sign},
                )
            self._begin_post_stair_settle(goal_sign)
            return self.post_stair_settle(depth, extra)
        self._clear_post_stair_goal()
        self._clear_stair_transition_tracking()
        if not self._camera_is_level():
            return self._queue_camera_action(
                "look_up",
                0.0,
                extra=extra,
                pending_after_level={"kind": "finish_stair_transition", "goal_sign": goal_sign},
            )
        self._finish_stair_transition(goal_sign)
        return None

    def post_stair_settle(self, depth: np.ndarray, extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        goal_sign = int(np.sign(self._post_stair_goal_sign))
        if not isinstance(self._post_stair_goal_xy, np.ndarray) or self._post_stair_goal_xy.size != 2 or goal_sign == 0:
            self._clear_post_stair_goal()
            self._clear_stair_transition_tracking()
            if goal_sign != 0:
                self._finish_stair_transition(goal_sign)
            return None
        self._phase = "post_stair_settle"
        self._goal_xy = self._post_stair_goal_xy.astype(np.float32, copy=False)
        self._omap.set_selected_frontier(self._goal_xy)
        max_steps = getattr(self, "_post_stair_settle_max_steps", None)
        if max_steps is not None and getattr(self, "_post_stair_settle_steps", 0) >= int(max_steps):
            self._resume_explore_after_post_stair_settle_timeout(goal_sign)
            return None
        if self._near_goal(self._proximity_threshold):
            self._clear_post_stair_goal()
            self._clear_stair_transition_tracking()
            self._finish_stair_transition(goal_sign)
            return None
        self._post_stair_settle_steps = getattr(self, "_post_stair_settle_steps", 0) + 1
        act = self._pointnav_step(depth, self._goal_xy, extra=extra, ignore_ban=True)
        if act["action"] == "stop":
            act["action"] = "move_forward"
        if extra:
            act.update(extra)
        return act

    def stairnav(self, depth, extra):
        goal_vec = self.stair_goal_vec[self.stair_frontier_goal_idx]
        self._start_stair_transition_tracking(int(np.sign(goal_vec[0])))
        stairnav_reset = bool(self.should_reset_stairnav)
        stairnav_goal_rho_theta = None
        stairnav_raw_action_id = None
        stairnav_raw_action = None
        stairnav_stop_overridden = False
        d = depth.squeeze(-1)
        filled_depth = d.copy()
        filled_depth[d == 0] = 1.0
        low_mask = None
        if self._stairnav_steps < 100:
            dense_points_cam = self._omap.dense_points_cam
            if goal_vec[0] > 0:
                dense_points_cam = dense_points_cam[filled_depth < 0.6]
            else:
                mask = filled_depth < 0.8
                mask = cp.asarray(mask)
                low_mask_raw = (dense_points_cam[..., 2] < -1) & mask
                dense_points_cam = dense_points_cam[low_mask_raw]
            if len(dense_points_cam) == 0:
                goal = np.array([0.0, 0.0])
            else:
                if goal_vec[0] > 0:
                    candidate = cp.max(dense_points_cam[:, -1])
                    mask = dense_points_cam[:, -1] > candidate * 0.9
                else:
                    candidate = cp.mean(dense_points_cam[:, -1])
                    mask = dense_points_cam[:, -1] < candidate
                if mask.sum() == 0:
                    goal = np.array([0.0, 0.0])
                else:
                    goal = dense_points_cam[mask, :2].mean(axis=0).get()
            rho = np.linalg.norm(goal)
            theta = np.arctan2(goal[1], goal[0])
            if goal_vec[0] < 0:
                rho = -rho
            stairnav_goal_rho_theta = [float(rho), float(theta)]
            act_id = self._gpu.stairnav(
                depth,
                np.array([rho, theta]),
                stairnav_reset,
            )
            stairnav_raw_action_id = int(act_id)
            stairnav_raw_action = {
                0: "stop",
                1: "move_forward",
                2: "turn_left",
                3: "turn_right",
            }.get(stairnav_raw_action_id, "unknown")
            if act_id == 0 and low_mask is not None and low_mask.sum() > 0:
                act_id = 1
                stairnav_stop_overridden = True
                self.should_reset_stairnav = True
            else:
                self.should_reset_stairnav = False
            mapping = {0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right"}

            out: Dict[str, Any] = {
                "action": mapping.get(int(act_id), "move_forward"),
                "map_viz": self._visualize_map(),
                "stairnav_goal_rho_theta": stairnav_goal_rho_theta,
                "stairnav_reset": stairnav_reset,
                "stairnav_raw_action_id": stairnav_raw_action_id,
                "stairnav_raw_action": stairnav_raw_action,
                "stairnav_executed_action": mapping.get(int(act_id), "move_forward"),
                "stairnav_stop_overridden": stairnav_stop_overridden,
            }
            self._stairnav_steps += 1
            self._stairnav_max_dist = max(self._stairnav_max_dist, np.linalg.norm(self._curr_xy - self._stairnav_start))
        else:
            out: Dict[str, Any] = {
                "action": "stop",
                "map_viz": self._visualize_map(),
                "stairnav_goal_rho_theta": None,
                "stairnav_reset": stairnav_reset,
                "stairnav_raw_action_id": None,
                "stairnav_raw_action": "step_limit",
                "stairnav_executed_action": "stop",
                "stairnav_stop_overridden": False,
            }
        if extra:
            out.update(extra)
        if out["action"] == "stop" or self._stairnav_failed:
            self.should_reset_stairnav = True
            # if we failed to cross stairs within max steps, we should avoid
            # going the same direction again
            self._stairnav_failed = self._stairnav_max_dist < 1.0 and self._stairnav_steps == 100
            if (goal_vec[0] > 0) ^ self._stairnav_failed:
                self.ascended = True
                self.descended = False
            else:
                self.ascended = False
                self.descended = True
            return self._handle_successful_stair_transition(depth, extra, int(np.sign(goal_vec[0])))
        return out

    def _finish_stair_transition(self, goal_sign: int) -> None:
        goal_sign = int(np.sign(goal_sign))
        self._clear_post_stair_goal()
        self._clear_stair_transition_tracking()
        if goal_sign > 0:
            self.ascended = True
            self.descended = False
        else:
            self.ascended = False
            self.descended = True
        self.reset(reset_stairs=False)
        target_name = self._vlm_target_name() if hasattr(self, "_vlm_target_name") else None
        try:
            self._objmap.set_candidate_objects(self._classes, target_name=target_name)
        except TypeError:
            self._objmap.set_candidate_objects(self._classes)
        self.stair_idx += int(goal_sign)

    def pointnav_descent(self, depth, extra):
        self._start_stair_transition_tracking(-1)
        if hasattr(self._omap, "set_downstair_obstacle_blocking"):
            self._omap.set_downstair_obstacle_blocking(False)
        if hasattr(self._omap, "approve_pending_downstair") and self._omap.approve_pending_downstair():
            self._refresh_filtered_frontiers()
        goal_xy = self._get_downstair_descent_goal(self._descent_entry_xy)
        if not isinstance(goal_xy, np.ndarray) or goal_xy.size != 2:
            if not self._camera_is_level():
                stairnav_start = self._curr_xy.copy() if isinstance(self._curr_xy, np.ndarray) else None
                return self._queue_camera_action(
                    "look_up",
                    0.0,
                    extra=extra,
                    pending_after_level={
                        "kind": "resume_stair_nav",
                        "stairnav_start": stairnav_start,
                    },
                )
            self._phase = "stair_nav"
            self._stairnav_start = self._curr_xy.copy()
            return self.stairnav(depth, extra)
        self._descent_goal_xy = goal_xy.astype(np.float32)
        self._goal_xy = self._descent_goal_xy
        self._omap.set_selected_frontier(self._goal_xy)
        if self._near_goal(self._proximity_threshold):
            return self._handle_successful_stair_transition(depth, extra, -1)
        if self._phase != "pointnav_descent":
            self._should_reset_pointnav = True
            self._phase = "pointnav_descent"
            return self._queue_camera_action("look_down", float(self._pitch_step_rad), extra=extra)
        self._phase = "pointnav_descent"
        act = self._pointnav_step(depth, self._goal_xy, extra=extra, ignore_ban=True)
        if act["action"] == "stop":
            act["action"] = "move_forward"
        return act

    def _has_regular_frontier_option(self) -> bool:
        frontiers = getattr(self, "_frontiers", None)
        return bool(isinstance(frontiers, np.ndarray) and len(frontiers) > 0)

    def _try_relaxed_obstacle_recovery(self) -> bool:
        if getattr(self._omap, "_ignore_real_obstacles_for_exploration", False):
            return False
        setter = getattr(self._omap, "set_ignore_real_obstacles_for_exploration", None)
        if not callable(setter):
            return False
        setter(True)
        self._refresh_filtered_frontiers()
        if not self._has_regular_frontier_option():
            return False
        self._phase = "explore"
        self._goal_xy = None
        self._should_reset_pointnav = True
        self._omap.set_selected_frontier(None)
        return True

    def _maybe_handle_stairs_or_escape(
        self,
        depth: np.ndarray,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not getattr(self, "multifloor", False):
            return None
        if self._phase in {"nav_to_stair", "pointnav_descent", "stair_nav", "post_stair_settle"}:
            return self.handle_stairs(update_only=False, depth=depth, extra=extra)
        if self._phase not in {"explore", "spin", "escape"} or self._has_regular_frontier_option():
            return None
        return self.handle_stairs(update_only=False, depth=depth, extra=extra)

    def _get_escape_goal(self) -> np.ndarray:
        if self._height_bev is None:
            return np.array([])
        px, heights = self._height_bev.get_valid_height_points()
        if heights.size == 0:
            return np.array([])
        dist_map = self._get_escape_distance_to_exploration_m()
        if dist_map is None:
            scores = heights
        else:
            scores = heights + dist_map[px[:, 1], px[:, 0]]
        goal_px = px[int(np.argmin(scores))].astype(np.int32, copy=False)
        return self._omap._px_to_xy(goal_px.reshape(1, 2))[0]

    def _get_escape_distance_to_exploration_m(self) -> Optional[np.ndarray]:
        if self._height_bev is None:
            return None
        shape = (self._height_bev.size, self._height_bev.size)
        for name in ("steppable_explored_area", "explored_area"):
            area = getattr(self._omap, name, None)
            if isinstance(area, np.ndarray) and area.shape == shape and np.any(area):
                dist_px = cv2.distanceTransform((~area.astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)
                return dist_px.astype(np.float32) / float(self._height_bev.pixels_per_meter)
        return None

    def escape(self, depth: np.ndarray, extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        goal_xy = self._get_escape_goal()
        if not isinstance(goal_xy, np.ndarray) or goal_xy.size != 2:
            self._phase = "explore"
            self._goal_xy = None
            self._omap.set_selected_frontier(None)
            return None
        entering_escape = self._phase != "escape"
        self._phase = "escape"
        self._goal_xy = goal_xy.astype(np.float32)
        self._omap.set_selected_frontier(self._goal_xy)
        self._should_reset_pointnav = entering_escape
        act = self._pointnav_step(depth, self._goal_xy, extra=extra, ignore_ban=True)
        if act["action"] == "stop":
            act["action"] = "move_forward"
        return act

    def handle_stairs(self, update_only, depth=None, extra=None) -> None:
        return self._handle_stairs_impl(update_only, depth=depth, extra=extra)

    def _handle_stairs_impl(self, update_only, depth=None, extra=None) -> None:
        if self._phase == "pointnav_descent":
            return self.pointnav_descent(depth, extra)
        if self._phase == "stair_nav":
            return self.stairnav(depth, extra)
        if self._phase == "post_stair_settle":
            return self.post_stair_settle(depth, extra)
        stair_frontiers = []
        view_facing = []
        goal_vec = []
        if len(self._omap.downstair_frontier) and (len(self._omap.upstair_frontier) == 0 or not self.ascended):
            down_frontiers, down_view = self._omap.get_stair_frontier_candidates("down")
            if len(down_frontiers):
                stair_frontiers.append(down_frontiers)
                view_facing.append(down_view)
                goal_vec.append(-np.ones((len(down_frontiers), 1), dtype=np.float32))
        if len(self._omap.upstair_frontier) and (len(self._omap.downstair_frontier) == 0 or not self.descended):
            up_frontiers, up_view = self._omap.get_stair_frontier_candidates("up")
            if len(up_frontiers):
                stair_frontiers.append(up_frontiers)
                view_facing.append(up_view)
                goal_vec.append(np.ones((len(up_frontiers), 1), dtype=np.float32))
        if len(stair_frontiers):
            self.stair_frontiers = np.vstack(stair_frontiers)
            self.stair_frontiers_view_facing = np.vstack(view_facing)
            self.stair_goal_vec = np.vstack(goal_vec)
        elif not self._has_regular_frontier_option() and len(self._omap.downstair_frontier):
            # If stairs are the only remaining option but all valid stair frontiers
            # were filtered out, keep a raw downstairs escape hatch.
            raw_down = np.asarray(self._omap.downstair_frontier, dtype=np.float32).reshape(-1, 2)
            raw_view = getattr(self._omap, "downstair_view_facing", None)
            if isinstance(raw_view, np.ndarray) and raw_view.size >= 2:
                raw_view = np.asarray(raw_view, dtype=np.float32).reshape(-1, 2)
            else:
                raw_view = raw_down.copy()
            self.stair_frontiers = raw_down
            self.stair_frontiers_view_facing = raw_view
            self.stair_goal_vec = -np.ones((len(raw_down), 1), dtype=np.float32)
        else:
            self.stair_frontiers = np.zeros((0, 2), dtype=np.float32)
            self.stair_frontiers_view_facing = np.zeros((0, 2), dtype=np.float32)
            self.stair_goal_vec = np.zeros((0, 1), dtype=np.float32)

        has_regular_frontier = self._has_regular_frontier_option()
        if self._phase == "escape" and (has_regular_frontier or len(self.stair_frontiers) > 0):
            self._phase = "explore"
            self._goal_xy = None
            self._should_reset_pointnav = True
            self._omap.set_selected_frontier(None)

        if not has_regular_frontier and len(self.stair_frontiers) == 0:
            if self._try_relaxed_obstacle_recovery():
                return None
            return self.escape(depth, extra)

        if update_only or len(self.stair_frontiers) == 0:
            if len(self.stair_frontiers) == 0 and self._phase in ["nav_to_stair", "pointnav_descent", "stair_nav"]:
                if self._phase == "pointnav_descent" and not self._camera_is_level():
                    return self._queue_camera_action(
                        "look_up",
                        0.0,
                        extra=extra,
                        pending_after_level={"kind": "resume_explore"},
                    )
                self._phase = "explore"
                self._goal_xy = None
                self._descent_goal_xy = None
                self._descent_entry_xy = None
                self.stair_frontier_goal_idx = None
                self._clear_stair_transition_tracking()
                self._clear_post_stair_goal()
                if hasattr(self, "_goal_view_facing"):
                    self._goal_view_facing = None
                self._should_reset_pointnav = True
                self._omap.set_selected_frontier(None)
            return None

        dist = np.linalg.norm(self._curr_xy - self.stair_frontiers, axis=-1)
        selected_idx = None
        if isinstance(self._goal_xy, np.ndarray) and self._goal_xy.size == 2:
            goal_dist = np.linalg.norm(self.stair_frontiers - self._goal_xy.reshape(1, 2), axis=-1)
            best_idx = int(np.argmin(goal_dist))
            if float(goal_dist[best_idx]) < 1.0:
                selected_idx = best_idx
        if selected_idx is None:
            chooser = getattr(self, "_select_stair_frontier_goal_idx", None)
            if callable(chooser):
                selected_idx = chooser(self.stair_frontiers, self.stair_frontiers_view_facing, self.stair_goal_vec)
                if selected_idx is None:
                    self._phase = "explore"
                    self._goal_xy = None
                    self._descent_goal_xy = None
                    self._descent_entry_xy = None
                    self.stair_frontier_goal_idx = None
                    if hasattr(self, "_goal_view_facing"):
                        self._goal_view_facing = None
                    self._should_reset_pointnav = True
                    self._omap.set_selected_frontier(None)
                    return None
            else:
                selected_idx = int(np.argmin(dist))
        selected_idx = int(selected_idx)
        self.stair_frontier_goal_idx = selected_idx
        self._goal_xy = self.stair_frontiers[selected_idx]
        self._omap.set_selected_frontier(self._goal_xy)
        if dist[selected_idx] < 0.5:
            self._goal_view_facing = self.stair_frontiers_view_facing[selected_idx]
            align_action = self._maybe_align_view(extra.get("det_seg", None), extra.get("has_obj", False), 0.6)
            if align_action is not None:
                if extra:
                    align_action.update(extra)
                return align_action
            should_stop = getattr(self, "_target_is_stair", lambda: False)
            if callable(should_stop) and should_stop():
                out = {
                    "action": "stop",
                    "map_viz": self._visualize_map(),
                    "nav_goal": self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None,
                    "stop_called": True,
                    "target_detected": True,
                }
                if extra:
                    out.update(extra)
                    out["target_detected"] = True
                return out
            if self.stair_goal_vec[selected_idx][0] < 0:
                self._descent_goal_xy = None
                self._descent_entry_xy = self.stair_frontiers[selected_idx].astype(np.float32)
                return self.pointnav_descent(depth, extra)
            self._start_stair_transition_tracking(1)
            self._phase = "stair_nav"
            self._stairnav_start = self._curr_xy.copy()
            return self.stairnav(depth, extra)
        else:
            if self._phase == "nav_to_stair":
                self._should_reset_pointnav = False
            else:
                self._should_reset_pointnav = True
            self._phase = "nav_to_stair"
            self._start_stair_transition_tracking(int(np.sign(self.stair_goal_vec[selected_idx][0])))
            act = self._pointnav_step(depth, self.stair_frontiers[selected_idx], extra=extra, ignore_ban=True)
            if act["action"] == "stop":
                act["action"] = "move_forward"
            return act

    # ------------------------------ Ban Logic ------------------------------
    def is_banned(self, point_xy: Optional[np.ndarray]) -> bool:
        if point_xy is None:
            return False
        if not isinstance(point_xy, np.ndarray):
            point_xy = np.asarray(point_xy, dtype=np.float32)
        if point_xy.size != 2 or self._banned_points.size == 0:
            return False
        dists = np.linalg.norm(self._banned_points - point_xy.reshape(1, 2), axis=1)
        return bool(np.any(dists < float(self._ban_threshold)))

    def _ban_point(self, point_xy: Optional[np.ndarray], reason: str = "unspecified") -> None:
        if point_xy is None:
            return False
        if not isinstance(point_xy, np.ndarray):
            point_xy = np.asarray(point_xy, dtype=np.float32)
        if point_xy.size != 2:
            return False
        # Avoid adding duplicates very close to existing bans
        if self.is_banned(point_xy):
            return False
        if self._banned_points.size == 0:
            self._banned_points = point_xy.reshape(1, 2).astype(np.float32)
        else:
            self._banned_points = np.vstack([self._banned_points, point_xy.reshape(1, 2).astype(np.float32)])

        # Keep the cached frontier state consistent immediately.  Goal selection can
        # run again in the same act() call, before the next mapping update has a
        # chance to filter the newly banned frontier.
        self._refresh_filtered_frontiers()
        invalidate_memory = getattr(self, "_invalidate_banned_frontier_memory", None)
        if callable(invalidate_memory):
            invalidate_memory()
        self._record_trace_event(
            "goal_banned",
            point_xy=self._trace_xy(point_xy),
            reason=str(reason),
            ban_threshold=float(self._ban_threshold),
            banned_count=int(len(self._banned_points)),
        )
        return True

    def _should_enforce_goal_ban(self) -> bool:
        """Ignore generic goal-ban halos while executing stair-transition goals."""
        return getattr(self, "_phase", None) not in {"nav_to_stair", "pointnav_descent", "stair_nav", "post_stair_settle"}

    def _goal_is_occupied(self) -> bool:
        """Check whether the current goal lies on an obstacle in the obstacle map."""
        if self._phase in {"pointnav_descent", "post_stair_settle"}:
            return False
        if self._goal_xy is None:
            return False
        if self._omap.is_occupied_xy(self._goal_xy):
            return True
        if hasattr(self._omap, "is_near_stairs") and self._omap.is_near_stairs(self._goal_xy):
            return True
        return False

    def _sync_frontier_visualization(self) -> None:
        if isinstance(self._frontiers, np.ndarray) and self._frontiers.size >= 2:
            raw_frontiers = getattr(self._omap, "frontiers", None)
            raw_frontiers_px = getattr(self._omap, "_frontiers_px", None)
            self._omap.frontiers = self._frontiers.copy()
            if (
                isinstance(raw_frontiers, np.ndarray)
                and isinstance(raw_frontiers_px, np.ndarray)
                and raw_frontiers.ndim == 2
                and raw_frontiers_px.ndim == 2
                and raw_frontiers.shape[0] == raw_frontiers_px.shape[0]
            ):
                dists = np.linalg.norm(self._frontiers[:, None, :] - raw_frontiers[None, :, :], axis=-1)
                matched_idx = np.argmin(dists, axis=1)
                matched_dists = dists[np.arange(self._frontiers.shape[0]), matched_idx]
                if np.all(matched_dists < 1e-6) and len(np.unique(matched_idx)) == len(matched_idx):
                    self._omap._frontiers_px = raw_frontiers_px[matched_idx].copy()
                    return
            self._omap._frontiers_px = self._omap._xy_to_px(self._frontiers)
            return
        self._omap.frontiers = np.array([])
        self._omap._frontiers_px = np.array([])

    def _refresh_filtered_frontiers(self) -> None:
        self._omap.set_banned_frontiers(self._banned_points)
        fr = self._omap.frontiers
        if isinstance(fr, np.ndarray) and fr.size >= 2 and self._banned_points.size > 0:
            keep_mask = np.ones(fr.shape[0], dtype=bool)
            for b in self._banned_points:
                d = np.linalg.norm(fr - b.reshape(1, 2), axis=1)
                keep_mask &= d >= float(self._ban_threshold)
            self._frontiers = fr[keep_mask]
        else:
            self._frontiers = fr
        self._sync_frontier_visualization()

    def _update_mapping(self, depth: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
        # Ensure depth is HxW float32
        d = depth
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32, copy=False)
        h, w = d.shape[:2]
        fx = fy = float(w) / (2.0 * np.tan(float(self._camera_fov_rad) / 2.0))
        # Use transform precomputed at the start of act
        tf_camera_to_episodic = self._tf_camera_to_episodic

        # Update obstacle and explored maps; compute frontiers
        if hasattr(self._omap, "set_downstair_obstacle_blocking"):
            self._omap.set_downstair_obstacle_blocking(self._phase != "pointnav_descent")
        allow_exploration_update = self._camera_is_level() or self._phase == "pointnav_descent"
        self._omap.update_map(
            d,
            tf_camera_to_episodic,
            float(self._min_depth),
            float(self._max_depth),
            fx,
            fy,
            float(self._camera_fov_rad),
            explore=allow_exploration_update,
            gpu_worker=self._gpu,
        )
        self._omap.update_agent_traj(self._curr_xy, self._curr_yaw)
        if self._color_bev is not None and rgb is not None:
            self._color_bev.update_map(
                rgb,
                getattr(self._omap, "dense_points_epi", None),
                getattr(self._omap, "out_of_range_mask", None),
                max_height_m=self._camera_height,
            )
            self._color_bev.update_agent_traj(self._curr_xy, self._curr_yaw)
        if self._height_bev is not None:
            height_offset_m = self._get_height_bev_update_offset()
            self._height_bev.update_map(
                getattr(self._omap, "dense_points_epi", None),
                getattr(self._omap, "out_of_range_mask", None),
                max_height_m=self._camera_height,
                height_offset_m=height_offset_m,
            )
            self._height_bev.update_agent_traj(self._curr_xy, self._curr_yaw)
            self._accumulate_stair_transition_updates()
        self._refresh_filtered_frontiers()
        self._remember_mapping_observation(rgb)

    def _update_stair_nav_height_map(self, depth: np.ndarray) -> None:
        """Refresh dense geometry for stair traversal without reopening frontier updates."""
        self._omap.update_agent_traj(self._curr_xy, self._curr_yaw)
        if self._height_bev is None:
            return

        d = depth
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32, copy=False)
        _, w = d.shape[:2]
        fx = fy = float(w) / (2.0 * np.tan(float(self._camera_fov_rad) / 2.0))

        self._omap.update_map(
            d,
            self._tf_camera_to_episodic,
            float(self._min_depth),
            float(self._max_depth),
            fx,
            fy,
            float(self._camera_fov_rad),
            explore=False,
            gpu_worker=self._gpu,
        )
        self._omap.update_agent_traj(self._curr_xy, self._curr_yaw)

        height_offset_m = self._get_height_bev_update_offset()
        self._height_bev.update_map(
            getattr(self._omap, "dense_points_epi", None),
            getattr(self._omap, "out_of_range_mask", None),
            max_height_m=self._camera_height,
            height_offset_m=height_offset_m,
        )
        self._height_bev.update_agent_traj(self._curr_xy, self._curr_yaw)
        self._accumulate_stair_transition_updates()

    def _get_height_bev_update_offset(self) -> float:
        if self._height_bev is None or not isinstance(self._curr_xy, np.ndarray):
            return 0.0
        median_height = self._height_bev.get_local_median_height(self._curr_xy)
        if median_height is None or not np.isfinite(median_height):
            return 0.0
        return min(float(median_height), 0.0)

    def get_color_bev_viz(self) -> Optional[np.ndarray]:
        if self._color_bev is None:
            return None
        goal_xy = self._goal_xy if isinstance(self._goal_xy, np.ndarray) and self._goal_xy.size == 2 else None
        banned = self._banned_points if isinstance(self._banned_points, np.ndarray) else None
        frontiers = self._frontiers if isinstance(self._frontiers, np.ndarray) else None
        return self._color_bev.visualize(
            frontiers_xy=frontiers,
            selected_frontier_xy=goal_xy,
            banned_xy=banned,
        )

    def get_height_bev_viz(self) -> Optional[np.ndarray]:
        if self._height_bev is None:
            return None
        goal_xy = self._goal_xy if isinstance(self._goal_xy, np.ndarray) and self._goal_xy.size == 2 else None
        banned = self._banned_points if isinstance(self._banned_points, np.ndarray) else None
        frontiers = self._frontiers if isinstance(self._frontiers, np.ndarray) else None
        downstair_overlay_mask, downstair_overlay_height = self._get_downstair_height_overlay()
        return self._height_bev.visualize(
            frontiers_xy=frontiers,
            selected_frontier_xy=goal_xy,
            banned_xy=banned,
            height_overlay_mask=downstair_overlay_mask,
            height_overlay_values=downstair_overlay_height,
        )

    def _get_downstair_height_overlay(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._height_bev is None or not hasattr(self._omap, "get_downstair_overlay_mask"):
            return None, None
        return self._height_bev.get_height_overlay(self._omap.get_downstair_overlay_mask())

    def _get_downstair_descent_reference_xy(self, entry_xy: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if entry_xy is not None:
            entry_xy_np = np.asarray(entry_xy, dtype=np.float32).reshape(-1)
            if entry_xy_np.size == 2:
                return entry_xy_np
        latest_agent_xy = getattr(self._omap, "_latest_agent_xy", None)
        if latest_agent_xy is not None:
            return np.asarray(latest_agent_xy, dtype=np.float32).reshape(2)
        camera_positions = getattr(self._omap, "_camera_positions", [])
        if len(camera_positions) > 0:
            return np.asarray(camera_positions[-1], dtype=np.float32).reshape(2)
        return None

    def _get_downstair_descent_goal(self, entry_xy: Optional[np.ndarray] = None) -> np.ndarray:
        if self._height_bev is None or not hasattr(self._omap, "get_downstair_descent_component_masks"):
            return np.array([])
        component_mask, seed_mask = self._omap.get_downstair_descent_component_masks(entry_xy)
        if not np.any(component_mask):
            return np.array([])

        px, heights = self._height_bev.get_valid_height_points(component_mask)
        if heights.size > 0:
            below_floor = heights < -0.05
            if np.any(below_floor):
                px = px[below_floor]
                heights = heights[below_floor]

        if heights.size == 0:
            fallback_mask = seed_mask if np.any(seed_mask) else component_mask
            rows, cols = np.where(fallback_mask)
            if len(rows) == 0:
                return np.array([])
            goal_px = np.array([int(np.rint(cols.mean())), int(np.rint(rows.mean()))], dtype=np.int32)
        else:
            reference_xy = self._get_downstair_descent_reference_xy(entry_xy)
            if reference_xy is not None:
                reference_px = self._omap._xy_to_px(reference_xy.reshape(1, 2))[0].astype(np.float32)
                distance_to_reference_m = (
                    np.linalg.norm(px.astype(np.float32) - reference_px.reshape(1, 2), axis=1)
                    / float(self._omap.pixels_per_meter)
                )
                target_distance_m = float(getattr(self._omap, "_downstair_descent_target_distance_m", 1.0))
                weight = float(getattr(self._omap, "_downstair_descent_distance_weight", 0.35))
                height_score = np.clip(-heights, 0.0, None)
                distance_score = -np.abs(distance_to_reference_m - target_distance_m)
                scores = height_score + weight * distance_score
                goal_idx = int(np.argmax(scores))
            else:
                goal_idx = int(np.argmin(heights))
            goal_px = px[goal_idx].astype(np.int32, copy=False)

        return self._omap._px_to_xy(goal_px.reshape(1, 2))[0]

    def _choose_goal(self) -> None:
        if isinstance(self._frontiers, np.ndarray) and self._frontiers.size >= 2:
            idx = random.randrange(self._frontiers.shape[0])
            self._goal_xy = self._frontiers[idx].astype(np.float32)
        else:
            self._goal_xy = None
        # Update visualization highlight for the selected frontier
        if hasattr(self._omap, "set_selected_frontier"):
            self._omap.set_selected_frontier(self._goal_xy)

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _near_goal(self, threshold: float) -> bool:
        if self._goal_xy is None or self._curr_xy is None:
            raise ValueError("Cannot determine proximity to goal without active goal and current position.")
        d = self._distance(self._goal_xy, self._curr_xy)
        return d <= float(threshold)

    def _pointnav_step(self, depth: np.ndarray, goal_xy: np.ndarray, extra: Optional[Dict[str, Any]] = None, ignore_ban: bool = False) -> Dict[str, Any]:
        return self._pointnav_step_impl(depth, goal_xy, extra=extra, ignore_ban=ignore_ban)

    def _pointnav_step_impl(
        self, depth: np.ndarray, goal_xy: np.ndarray, extra: Optional[Dict[str, Any]] = None, ignore_ban: bool = False
    ) -> Dict[str, Any]:
        """Compute pointgoal (rho, theta), run pointnav, and return action dict.

        Args:
            depth: Depth observation array (HxW or HxWx1).
            goal_xy: Goal position in world XY coordinates.
            extra: Optional extra key/values to merge into the returned dict
                   (e.g., det_seg for object planner visualizations).

        Returns:
            Dict with keys: action (str), map_viz (np.ndarray), and any extras.
        """
        rho, theta = rho_theta(self._curr_xy, self._curr_yaw, goal_xy)
        goal_vec = np.array([rho, theta], dtype=np.float32)
        # Capture and consume the current reset flag so that callers can
        # request a one-shot RNN reset. Any resets requested *inside* this
        # method (e.g., on STOP/ban) should apply to the *next* call.
        reset_flag = bool(self._should_reset_pointnav)
        act_id = self._gpu.pointnav(depth, goal_vec, reset=reset_flag)
        raw_act_id = int(act_id)
        raw_action_name = {
            0: "stop",
            1: "move_forward",
            2: "turn_left",
            3: "turn_right",
        }.get(raw_act_id, "unknown")
        stop_overridden = False
        override_reason = None
        if act_id == 0 and not ignore_ban:
            # If the RNN outputs STOP, ban this goal to avoid revisiting
            new_ban = self._ban_point(goal_xy, reason="pointnav_stop")
            if new_ban:
                stop_overridden = True
                override_reason = "pointnav_stop_banned_goal"
                print("PointNav STOP: banning current goal and overriding to MOVE_FORWARD.")
                # Update obstacle-map viz so the banned point is crossed out immediately
                if hasattr(self._omap, "set_banned_frontiers"):
                    self._omap.set_banned_frontiers(self._banned_points)
                if isinstance(self._frontiers, np.ndarray) and self._frontiers.size >= 2:
                    keep_mask = np.ones(self._frontiers.shape[0], dtype=bool)
                    for b in self._banned_points:
                        d = np.linalg.norm(self._frontiers - b.reshape(1, 2), axis=1)
                        keep_mask &= d >= float(self._ban_threshold)
                    self._frontiers = self._frontiers[keep_mask]
                self._sync_frontier_visualization()
                # Clear current goal so the planner switches targets immediately
                try:
                    if self._goal_xy is not None and np.allclose(self._goal_xy.reshape(1, 2), goal_xy.reshape(1, 2)):
                        self._goal_xy = None
                        if hasattr(self._omap, "set_selected_frontier"):
                            self._omap.set_selected_frontier(None)
                        self._should_reset_pointnav = True
                except Exception:
                    # Be robust to any shape/type mismatch
                    self._goal_xy = None
                    if hasattr(self._omap, "set_selected_frontier"):
                        self._omap.set_selected_frontier(None)
                    self._should_reset_pointnav = True
            act_id = 1  # Override stop to move_forward to continue exploration
        # If a caller requested a reset for this step, clear it now so that
        # the reset is one-shot. Any resets set above (e.g., after STOP/ban)
        # will remain in effect for the next call.
        if reset_flag:
            self._should_reset_pointnav = False
        mapping = {0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right"}
        out: Dict[str, Any] = {
            "action": mapping.get(int(act_id), "move_forward"),
            "map_viz": self._visualize_map(),
        }
        # Attach policy info useful for failure-cause analysis
        try:
            nav_goal_list = goal_xy.tolist() if hasattr(goal_xy, "tolist") else [float(goal_xy[0]), float(goal_xy[1])]
        except Exception:
            nav_goal_list = None
        out.update(
            {
                "nav_goal": nav_goal_list,
                "stop_called": out["action"] == "stop",
                "target_detected": False,  # frontier planner has no object detection
                "pointnav_goal_rho_theta": goal_vec.tolist(),
                "pointnav_reset": reset_flag,
                "pointnav_raw_action_id": raw_act_id,
                "pointnav_raw_action": raw_action_name,
                "pointnav_executed_action": out["action"],
                "pointnav_stop_overridden": stop_overridden,
                "pointnav_override_reason": override_reason,
            }
        )
        if raw_act_id == 0 or stop_overridden or reset_flag:
            self._record_trace_event(
                "pointnav_decision",
                goal_xy=nav_goal_list,
                rho_theta=goal_vec.tolist(),
                raw_action=raw_action_name,
                executed_action=out["action"],
                reset=reset_flag,
                stop_overridden=stop_overridden,
                override_reason=override_reason,
                ignore_ban=bool(ignore_ban),
                goal_distance=float(np.linalg.norm(np.asarray(goal_xy) - self._curr_xy)),
            )
        if extra:
            out.update(extra)
        return out

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._start_layer_step()
        rgb = obs.get("rgb")  # unused here, but could be logged
        depth = obs.get("depth")
        gps = obs.get("gps")
        yaw = obs.get("yaw")

        # Precompute agent pose and transform once per step.
        self._set_agent_pose(gps, yaw)

        # Update mapping with the precomputed transform
        self._update_mapping(depth, rgb)

        # If the current goal has been banned, immediately clear it so we switch targets
        if self._goal_xy is not None and self._should_enforce_goal_ban() and self.is_banned(self._goal_xy):
            self._goal_xy = None
            # Clear highlight on the map as well
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)

        # If staircase protection (or other updates) mark the current goal as
        # an obstacle, drop and ban it so we replan instead of walking into it.
        if self._goal_xy is not None and self._goal_is_occupied():
            self._ban_point(self._goal_xy)
            self._goal_xy = None
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)
            self._should_reset_pointnav = True

        # Initial spin for situational awareness
        if self._phase == "init" and self._init_steps < self._max_init_steps:
            self._init_steps += 1
            # Provide map visualization for UI
            out = {
                "action": "turn_left",
                "map_viz": self._visualize_map(),
                "nav_goal": self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None,
                "stop_called": False,
                "target_detected": False,
            }
            if self._init_steps >= self._max_init_steps:
                self._phase = "explore"
            return out

        # If we have an active goal and are close enough, request a reset of the
        # pointnav RNN state before pursuing the next goal to avoid state carryover.
        if self._phase == "explore":
            stair_out = self._maybe_handle_stairs_or_escape(depth)
            if stair_out is not None:
                return stair_out

        if self._goal_xy is not None:
            if self._near_goal(self._proximity_threshold):
                self._should_reset_pointnav = True
                self._choose_goal_with_trace()
        else:
            # No active goal yet; just pick one without resetting.
            self._choose_goal_with_trace()

        if self._goal_xy is None:
            return {
                "action": "stop",
                "map_viz": self._visualize_map(),
                "nav_goal": None,
                "stop_called": True,
                "target_detected": False,
            }

        # Step pointnav towards goal
        return self._pointnav_step(depth, self._goal_xy)
