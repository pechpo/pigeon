from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .random_object_planner import RandomObjectPlanner
from ..mapping.value_map import ValueMap
from ..utils.geometry_utils import closest_point_within_threshold
from ..utils.acyclic_enforcer import AcyclicEnforcer


class ValueMapObjectPlanner(RandomObjectPlanner):
    """Frontier exploration guided by a learned value map with object pursuit.

    - Inherits detection + target pursuit from RandomObjectPlanner
    - During exploration, selects frontiers by ValueMap scores (BLIP2-ITM cosine)
    - Returns ValueMap visualization for integration with utils/viz.py
    """

    tmpl = "Seems like there is a target_object ahead."
    models = ["pointnav_rnn", "yolov7", "mobile_sam"]

    def __init__(self) -> None:
        super().__init__()
        # Value map uses one channel (target-object relevance)
        # Sync explored-area mask with obstacle map for consistent visualization.
        # Use weighted-average fusion for smoother value maps (matches VLFM eval)
        self._vmap = ValueMap(value_channels=1, obstacle_map=self._omap, use_max_confidence=False)
        # BLIP2-ITM is served in GPU manager; use our GPU client to query.
        # Keep soft memory of last chosen frontier
        self._last_goal: Optional[np.ndarray] = None
        # Waypoint aggregation radius for value lookup (meters)
        self._value_radius_m: float = 0.5
        # Minimal gain needed to switch goals (prevents thrashing)
        self._goal_hysteresis: float = 0.01
        # Corresponds to the detection confidence threshold in VLFM
        self._last_value = float("-inf")
        self._acyclic_enforcer = AcyclicEnforcer()
        self._downstair_detection = True
        self._upstair_detection = True

    def set_yolo_conf_thresh(self) -> None:
        # ValueMap uses a stricter threshold when relying on YOLO
        # and a slightly lower one when the target is handled by OWLv2.
        target_conf = 0.50 if self._target_uses_owl() else 0.80
        self._yolo_conf_thresh = [target_conf]  # target object
        if self._upstair_detection:
            self._yolo_conf_thresh.append(0.30)  # stairs
    
    def reset(self, reset_stairs=True) -> None:
        super().reset(reset_stairs)
        self._vmap.reset()
        self._last_goal = None
        self._last_value = float("-inf")
        self._acyclic_enforcer = AcyclicEnforcer()

    # ------------------------------ Value Map Logic ------------------------------
    def _update_value_map(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        """Compute BLIP2-ITM cosine and fuse into value map.

        Aligns with vlfm ITMPolicyV2 by using a fixed
        prompt template and single-channel value fusion.
        """
        if rgb is None or depth is None:
            return
        # Build the BLIP prompt using V2 template
        tgt = self._target_class or ""
        if not isinstance(tgt, str) or len(tgt) == 0:
            return
        try:
            # ITMPolicyV2 uses a template: "Seems like there is a target_object ahead."
            # Replace placeholder with the concrete target name. Guard against
            # '|' in names which may be used as separators in other contexts.
            prompt = self.tmpl.replace("target_object", tgt.replace("|", "/"))
            cosine = float(self._gpu.itm_cosine(rgb, prompt))
        except Exception:
            return
        values = np.array([cosine], dtype=np.float32)
        self._vmap.update_map(
            values=values,
            depth=depth,
            tf_camera_to_episodic=self._tf_camera_to_episodic,
            min_depth=float(self._min_depth),
            max_depth=float(self._max_depth),
            fov=float(self._camera_fov_rad),
        )
        # Keep agent trajectory in sync for visualization
        self._vmap.update_agent_traj(self._curr_xy, self._curr_yaw)

    def _choose_goal_by_value(self) -> None:
        """Pick the best non-banned, non-cyclic frontier by value; update _goal_xy.

        Changes vs. previous version:
        - Add loop suppression using AcyclicEnforcer (prevents oscillations).
        - If last frontier disappears or its value drops notably, switch promptly.
        - Keep minimal hysteresis to avoid thrashing when the old goal is still good.
        """
        fr = self._frontiers
        if not isinstance(fr, np.ndarray) or fr.size < 2:
            self._goal_xy = None
            return

        # Sort frontiers by local value around each waypoint
        sorted_pts, sorted_vals = self._vmap.sort_waypoints(fr, radius=self._value_radius_m)

        # Prepare context for loop suppression and stickiness
        robot_xy = self._curr_xy if isinstance(self._curr_xy, np.ndarray) else None
        top_two_values = tuple(float(v) for v in list(sorted_vals[:2]))

        # Try to keep pursuing previous goal only if it is still competitive
        chosen_idx: Optional[int] = None
        if self._last_goal is not None:
            # Find exact match or a nearby frontier corresponding to the last goal
            curr_index = None
            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_goal):
                    curr_index = idx
                    break
            if curr_index is None:
                ci = closest_point_within_threshold(sorted_pts, self._last_goal, threshold=0.5)
                if int(ci) != -1:
                    curr_index = int(ci)

            if curr_index is not None:
                curr_val = float(sorted_vals[curr_index])
                # Stick to last goal only if its current value hasn't dropped much
                if curr_val + float(self._goal_hysteresis) > float(self._last_value):
                    chosen_idx = int(curr_index)

        # Otherwise, pick the first non-banned, non-cyclic best frontier
        if chosen_idx is None:
            for idx, (pt, val) in enumerate(zip(sorted_pts, sorted_vals)):
                if self.is_banned(pt):
                    continue
                # Loop suppression: avoid repeating same (state, action) pairs
                if robot_xy is not None and self._acyclic_enforcer.check_cyclic(robot_xy, pt, top_two_values):
                    # Suppress cyclic frontier and continue searching
                    continue
                chosen_idx = idx
                break

        # Fallback: if everything was filtered out as cyclic/banned, choose the closest 
        if chosen_idx is None:
            dists = np.linalg.norm(fr - self._curr_xy.reshape(1, 2), axis=1)
            chosen_idx = int(np.argmin(dists))

        # Finalize selection and update memory for next step
        chosen = sorted_pts[int(chosen_idx)].astype(np.float32)
        chosen_val = float(sorted_vals[int(chosen_idx)])
        self._goal_xy = chosen
        self._last_goal = chosen.copy()
        self._last_value = chosen_val
        if robot_xy is not None:
            # Record this decision to discourage cycles in subsequent steps
            self._acyclic_enforcer.add_state_action(robot_xy, chosen, top_two_values)

    # ------------------------------ Act ------------------------------
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._start_layer_step()
        rgb = obs.get("rgb")
        depth = obs.get("depth")
        gps = obs.get("gps")
        yaw = obs.get("yaw")

        # Precompute agent pose and transform once per step.
        self._set_agent_pose(gps, yaw)

        # Update obstacle map/frontiers
        self._update_mapping(depth, rgb)

        # If the current goal has been banned (e.g., due to repeated STOPs or
        # manual banning), immediately clear it so that we replan using a new
        # frontier/value-based target instead of continuing toward a bad goal.
        if self._goal_xy is not None and self._should_enforce_goal_ban() and self.is_banned(self._goal_xy):
            self._goal_xy = None
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)

        # Update object map pruning w.r.t. explored area
        self._objmap.update_explored(self._tf_camera_to_episodic, float(self._max_depth), float(self._camera_fov_rad))

        # Update value map from BLIP2-ITM
        self._update_value_map(rgb, depth)

        # Try detection each step; may switch to target phase
        det_seg = self._detect_and_update(rgb, depth)
        self._refresh_detection_viz(
            det_seg=det_seg,
            value_map_viz=self._vmap.visualize(self._markers_for_viz(), obstacle_map=self._omap),
        )
        has_obj = bool(self._target_class) and self._objmap.has_object(self._target_class)

        # Initial spin
        if self._phase == "init" and self._init_steps < self._max_init_steps and self._phase != "target":
            self._init_steps += 1
            obs_viz = self._visualize_map()
            vmap_viz = self._vmap.visualize(self._markers_for_viz(), obstacle_map=self._omap)
            out = {
                "action": "turn_left",
                "map_viz": obs_viz,
                "value_map_viz": vmap_viz,
                "det_seg": det_seg,
                "nav_goal": (self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None),
                "stop_called": False,
                "target_detected": has_obj,
            }
            if self._init_steps >= self._max_init_steps:
                self._phase = "explore"
            return out

        # Target phase: pursue object map goal
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

        # If the current goal becomes an obstacle in the obstacle map
        # (e.g., due to downstairs protection), drop and ban it so that
        # we replan instead of treating an obstacle as a valid goal.
        if self._goal_xy is not None and self._goal_is_occupied():
            self._ban_point(self._goal_xy)
            self._goal_xy = None
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)
            self._should_reset_pointnav = True
            self._last_goal = None

        # Exploration: always re-evaluate by value each step for timely switching.
        # This allows switching when the previous goal disappears or its value drops.
        prev = self._goal_xy.copy() if isinstance(self._goal_xy, np.ndarray) else None
        self._choose_goal_by_value()
        if prev is not None and self._goal_xy is not None:
            # Reset RNN only when the goal meaningfully changes
            self._should_reset_pointnav = (self._distance(prev, self._goal_xy) > self._same_goal_thresh)

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

        # PointNav toward exploration goal
        return self._pointnav_step(
            depth,
            self._goal_xy,
            extra={
                "det_seg": det_seg,
                "target_detected": has_obj,
                "value_map_viz": self._vmap.visualize(self._markers_for_viz(), obstacle_map=self._omap),
            },
        )

    # ------------------------------ Visualization Helpers ------------------------------
    def _markers_for_viz(self) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        markers: List[Tuple[np.ndarray, Dict[str, Any]]] = []
        # Draw all frontiers in blue
        fr = self._frontiers
        if isinstance(fr, np.ndarray) and fr.size >= 2:
            for f in fr:
                markers.append(
                    (
                        f[:2],
                        {
                            "radius": 5,
                            "thickness": 2,
                            "color": (200, 0, 0),  # BGR
                        },
                    )
                )
        # Highlight selected goal in red
        if isinstance(self._goal_xy, np.ndarray) and self._goal_xy.size == 2:
            markers.append(
                (
                    self._goal_xy,
                    {
                        "radius": 7,
                        "thickness": -1,
                        "color": (0, 0, 200),  # BGR
                    },
                )
            )
        return markers
