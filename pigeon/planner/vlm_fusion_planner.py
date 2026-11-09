from typing import Any, Dict, Optional

import numpy as np

from .random_fusion_planner import RandomFusionPlanner
from .vlm_object_planner import VLMObjectPlanner
from ..mapping.fusion_pointcloud_map import FusionPointCloudMap

# TODO: This planner is currently not reliable. It's performance needs to be notably better than vlm object planner.


class VLMFusionPlanner(VLMObjectPlanner, RandomFusionPlanner):
    """Fusion-aware VLM planner: fusion detections + VLM frontier selection with suspect bias."""

    models = ["pointnav_rnn", "yolov7", "owl_v2", "mobile_sam"]

    def __init__(self) -> None:
        super().__init__()
        # Swap in fusion map and keep VLM confirmation hooked up.
        self._objmap = FusionPointCloudMap(
            erosion_size=5,
            explored_area=self._omap.explored_area,
            steppable_explored_area=self._omap.steppable_explored_area,
            episode_pixel_origin=self._omap._episode_pixel_origin,
            pixels_per_meter=self._omap.pixels_per_meter,
            obstacle_map=self._omap,
        )
        self._objmap.set_confirmation_client(self._vlm)
        self._bind_confirmation_viz()
        self._bind_downstair_approval()
        self.candidate_prompt = None
        self._goal_view_facing: Optional[np.ndarray] = None

    def reset(self, reset_stairs=True) -> None:
        super().reset(reset_stairs)
        self._objmap.set_confirmation_client(self._vlm)
        self._bind_confirmation_viz()
        self._bind_downstair_approval()
        self._goal_view_facing = None
        self.candidate_prompt = None

    def _ban_point(self, point_xy, reason="unspecified"):
        if self._goal_view_facing is not None:
            self._objmap.mark_view_inaccessible(self._goal_xy)
        return super()._ban_point(point_xy, reason=reason)

    def _choose_goal_vlm(self, allow_spin=True) -> None:
        # Prefer suspected viewpoints; otherwise fall back to VLM selection.
        suspects, centers = self._objmap.get_suspected_view_positions(self._target_class, self._curr_xy)
        if isinstance(suspects, np.ndarray) and suspects.size >= 2:
            candidates = [(p, c) for p, c in zip(suspects, centers) if not self.is_banned(p)]
            if candidates:
                if self._goal_xy is not None and not self.is_banned(self._goal_xy) and self._goal_view_facing is not None:
                    self._last_goal_selection_source = "keep_suspected_viewpoint"
                    return
                self._goal_view_facing = None
                pos, center = min(candidates, key=lambda pc: self._distance(pc[0], self._curr_xy))
                # pos, center = candidates[int(np.random.randint(len(candidates)))]
                self._goal_xy = np.asarray(pos, dtype=np.float32)
                if isinstance(center, np.ndarray) and center.size >= 2 and not np.any(np.isnan(center)):
                    self._goal_view_facing = center.astype(np.float32)
                if hasattr(self._omap, "set_selected_frontier"):
                    self._omap.set_selected_frontier(self._goal_xy)
                self._last_goal_selection_source = "suspected_object_viewpoint"
                return
        self._goal_view_facing = None
        num_frontiers = len(self._frontiers) + len([m for m in self._frontier_mem if m.get("valid", False)])
        if num_frontiers > 0:
            self._last_goal_selection_source = "vlm_frontier_or_fallback"
            super()._choose_goal_vlm(allow_spin=False)
        else:
            self._last_goal_selection_source = "no_frontier_no_target"
            tgt_xy = self._objmap.get_best_object(self._target_class, self._curr_xy, True)
            if tgt_xy is not None and len(tgt_xy) == 2:
                goal_xy = np.asarray(tgt_xy, dtype=np.float32)
                # If the nearest object point lies in a banned region,
                # treat it as unreachable and do not enter target phase.
                if self.is_banned(goal_xy):
                    return
                if hasattr(self._omap, "set_selected_frontier"):
                    self._omap.set_selected_frontier(goal_xy)
                if self._goal_xy is None:
                    self._goal_xy = goal_xy
                elif self._distance(goal_xy, self._goal_xy) > self._same_goal_thresh:
                    self._should_reset_pointnav = True
                    self._goal_xy = goal_xy
                self._phase = "target"
                self._last_goal_selection_source = "reliable_object_fallback"

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        result = super().act(obs)
        if result.get("stop_called", False):
            return result

        has_obj = bool(self._target_class) and self._objmap.has_object(self._target_class)
        align_action = self._maybe_align_view(result.get("det_seg"), has_obj)
        if align_action is None:
            return result

        # Keep VLM panels visible while aligning in place.
        if "value_map_viz" in result:
            align_action["value_map_viz"] = result.get("value_map_viz")
        if "right_panel_viz" in result:
            align_action["right_panel_viz"] = result.get("right_panel_viz")
        if "right_panel_text" in result:
            align_action["right_panel_text"] = result.get("right_panel_text")
        return align_action
