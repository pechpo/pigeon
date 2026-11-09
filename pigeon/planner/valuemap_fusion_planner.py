import copy
import os
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from .random_fusion_planner import RandomFusionPlanner
from .valuemap_object_planner import ValueMapObjectPlanner
from ..mapping.fusion_pointcloud_map import FusionPointCloudMap
from ..models.vlm import VLMClient
from ..utils.geometry_utils import xyz_yaw_to_tf_matrix

# TODO: This planner is currently not reliable. It's performance needs to be notably better than valuemap object planner.

class ValueMapFusionPlanner(ValueMapObjectPlanner, RandomFusionPlanner):
    """Fusion-aware ValueMap planner: RandomFusion detection/fusion + value-map frontier selection."""

    models = ["pointnav_rnn", "yolov7", "mobile_sam"]

    def __init__(self) -> None:
        super().__init__()
        # Replace default object map with fusion map and hook VLM for candidate expansion/confirmation.
        self._objmap = FusionPointCloudMap(
            erosion_size=5,
            explored_area=self._omap.explored_area,
            steppable_explored_area=self._omap.steppable_explored_area,
            episode_pixel_origin=self._omap._episode_pixel_origin,
            pixels_per_meter=self._omap.pixels_per_meter,
            obstacle_map=self._omap,
        )
        self._vlm: Optional[VLMClient] = VLMClient(os.environ.get("VLM_MODEL", "gpt-4o"))
        self._objmap.set_confirmation_client(self._vlm)
        self._bind_downstair_approval()
        self.candidate_prompt = None
        self._goal_view_facing: Optional[np.ndarray] = None

    def set_target_object(self, name: str) -> None:
        # Keep RandomFusion VLM candidate expansion.
        super().set_target_object(name)
        self._setup_object_map()

    def reset(self) -> None:
        super().reset()
        # Ensure fusion map keeps the VLM confirmation client.
        self._objmap.set_confirmation_client(self._vlm)
        self._bind_downstair_approval()
        self._goal_view_facing = None
        self.candidate_prompt = None

    # ------------------------------ Goal selection (suspected views first) ------------------------------
    def _choose_goal(self) -> None:
        # Prefer suspected viewpoints; otherwise fall back to value-map-ranked frontiers.
        self._goal_view_facing = None
        suspects, centers = self._objmap.get_suspected_view_positions(self._target_class, self._curr_xy)
        if isinstance(suspects, np.ndarray) and suspects.size >= 2:
            candidates = [(p, c) for p, c in zip(suspects, centers) if not self.is_banned(p)]
            if candidates:
                idx = int(np.random.randint(len(candidates)))
                pos, center = candidates[idx]
                self._goal_xy = np.asarray(pos, dtype=np.float32)
                if isinstance(center, np.ndarray) and center.size >= 2 and not np.any(np.isnan(center)):
                    self._goal_view_facing = center.astype(np.float32)
                if hasattr(self._omap, "set_selected_frontier"):
                    self._omap.set_selected_frontier(self._goal_xy)
                return

        self._choose_goal_by_value()

    # ------------------------------ Act loop ------------------------------
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._start_layer_step()
        rgb = obs.get("rgb")
        depth = obs.get("depth")
        gps = obs.get("gps")
        yaw = obs.get("yaw")

        x, y = float(gps[0]), float(gps[1])
        camera_position = np.array([x, -y, self._camera_height], dtype=np.float32)
        self._curr_xy = camera_position[:2].copy()
        self._curr_yaw = float(yaw)
        self._tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, self._curr_yaw)

        # Update obstacle/frontier maps and value map.
        self._update_mapping(depth, rgb)

        if self._goal_xy is not None and self._should_enforce_goal_ban() and self.is_banned(self._goal_xy):
            self._goal_xy = None
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)

        self._update_value_map(rgb, depth)

        det_seg = self._detect_and_update(rgb, depth)
        self._refresh_detection_viz(
            det_seg=det_seg,
            value_map_viz=self._vmap.visualize(self._markers_for_viz(), obstacle_map=self._omap),
        )
        has_obj = bool(self._target_class) and self._objmap.has_object(self._target_class)

        # Initial spin.
        if self._phase == "init" and self._init_steps < self._max_init_steps and self._phase != "target":
            self._init_steps += 1
            out = {
                "action": "turn_left",
                "map_viz": self._visualize_map(),
                "value_map_viz": self._vmap.visualize(self._markers_for_viz(), obstacle_map=self._omap),
                "det_seg": det_seg,
                "nav_goal": (self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None),
                "stop_called": False,
                "target_detected": has_obj,
            }
            if self._init_steps >= self._max_init_steps:
                self._phase = "explore"
            return out

        # Target phase.
        if self._phase == "target":
            if has_obj:
                tgt_xy = self._objmap.get_best_object(self._target_class, self._curr_xy)
                if tgt_xy is not None and len(tgt_xy) == 2:
                    tgt_xy_arr = np.asarray(tgt_xy, dtype=np.float32)
                    if self.is_banned(tgt_xy_arr):
                        has_obj = False
                    else:
                        self._goal_xy = tgt_xy_arr
                        if self._near_goal(self._goal_proximity_thresh):
                            return {
                                "action": "stop",
                                "map_viz": self._visualize_map(),
                                "value_map_viz": self._vmap.visualize(self._markers_for_viz(), obstacle_map=self._omap),
                                "det_seg": det_seg,
                                "nav_goal": (self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None),
                                "stop_called": True,
                                "target_detected": has_obj,
                            }
                        return self._pointnav_step(
                            depth,
                            self._goal_xy,
                            extra={
                                "det_seg": det_seg,
                                "target_detected": has_obj,
                                "value_map_viz": self._vmap.visualize(self._markers_for_viz(), obstacle_map=self._omap),
                            },
                        )
            if not has_obj:
                self._phase = "explore"
                self._goal_xy = None
                self._should_reset_pointnav = True

        if self._goal_xy is not None and self._goal_is_occupied():
            self._ban_point(self._goal_xy)
            self._goal_xy = None
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)
            self._should_reset_pointnav = True
            self._last_goal = None

        prev = self._goal_xy.copy() if isinstance(self._goal_xy, np.ndarray) else None
        self._choose_goal_with_trace()
        if prev is not None and self._goal_xy is not None:
            self._should_reset_pointnav = self._distance(prev, self._goal_xy) > self._same_goal_thresh

        align_action = self._maybe_align_view(det_seg, has_obj)
        if align_action is not None:
            align_action["value_map_viz"] = self._vmap.visualize(self._markers_for_viz(), obstacle_map=self._omap)
            return align_action

        if self._goal_xy is None:
            return {
                "action": "stop",
                "map_viz": self._visualize_map(),
                "value_map_viz": self._vmap.visualize(self._markers_for_viz(), obstacle_map=self._omap),
                "det_seg": det_seg,
                "nav_goal": None,
                "stop_called": True,
                "target_detected": has_obj,
            }

        return self._pointnav_step(
            depth,
            self._goal_xy,
            extra={
                "det_seg": det_seg,
                "target_detected": has_obj,
                "value_map_viz": self._vmap.visualize(self._markers_for_viz(), obstacle_map=self._omap),
            },
        )
