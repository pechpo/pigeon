from typing import Any, Dict, Optional

import numpy as np
import networkx as nx
import json

from .random_fusion_planner import RandomFusionPlanner
from .vlm_fusion_planner import VLMFusionPlanner
from .valuemap_object_planner import ValueMapObjectPlanner
from ..mapping.fusion_pointcloud_map import FusionPointCloudMap
from ..mapping.value_map import ValueMap
from ..utils.acyclic_enforcer import AcyclicEnforcer
from ..utils.geometry_utils import xyz_yaw_to_tf_matrix, closest_point_within_threshold

# TODO: This planner is currently not reliable. It's performance needs to be notably better than vlm object planner.

REFERENCE_ROOMS = [
    "bathroom",
    "bedroom",
    "dining_room",
    "garage",
    "hall",
    "kitchen",
    "laundry_room",
    "living_room",
    "office",
    "rec_room",
]


class VLMFusionValuePlanner(VLMFusionPlanner, ValueMapObjectPlanner):
    """Fusion-aware VLM planner: fusion detections + VLM frontier selection with suspect bias."""

    models = ["pointnav_rnn", "yolov7", "mobile_sam"]

    _frontier_selection_prompt_template_template = (
        """You are a robot and after exploration, you have the given images.\n"""
        """Every image represents a possible direction to find the {goal_object}.\n"""
        """Based on the numerical markers in the image and the prior probabilities between room type and goal object, select one of these numbers to move next.\n"""
        """If you're not confident in moving to the marker to find the {goal_object}, you should return <answer> 0 </answer>.\n"""
        """Below is the prior probabilities between Room Type and Goal Object\n"""
        """{prior}\n"""
        """Please note all closed doors cannot be opened.\n"""
        """Please follow the format like this, without bracket --\n"""
        """<thought> [put your step-by-step thinking process here] </thought>\n"""
        """<answer> [put a single marker id or 0 here] </answer>\n"""
    )

    def __init__(self) -> None:
        super().__init__()
        # Swap in fusion map and keep VLM confirmation hooked up.
        self._value_radius_m: float = 0.5
        # Minimal gain needed to switch goals (prevents thrashing)
        self._goal_hysteresis: float = 0.01
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
        self._vmap = ValueMap(value_channels=1, obstacle_map=self._omap, use_max_confidence=False)
        self._last_goal: Optional[np.ndarray] = None
        # Corresponds to the detection confidence threshold in VLFM
        self._last_value = float("-inf")
        self._acyclic_enforcer = AcyclicEnforcer()
        with open("statistic_priors/knowledge_graph.json", "r") as f:
            self.knowledge_graph = nx.node_link_graph(json.load(f))

    def reset(self, reset_stairs=True) -> None:
        super().reset(reset_stairs)
        self._objmap.set_confirmation_client(self._vlm)
        self._bind_confirmation_viz()
        self._bind_downstair_approval()
        self._goal_view_facing = None
        self.candidate_prompt = None
        self._vmap.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_goal = None
        self._last_value = float("-inf")

    def set_target_object(self, name: str) -> None:
        # for object map without fusion, it only needs two classes: target and stairs
        super().set_target_object(name)
        room_probabilities = self.get_room_probs()
        sorted_rooms = sorted(room_probabilities.items(), key=lambda x: (-x[1], x[0])) 
        probability_strings = [f'"{room.capitalize()}": {prob:.1f}%' for room, prob in sorted_rooms]
        prob_entries = ",\n".join(probability_strings)
        self._frontier_selection_prompt_template = self._frontier_selection_prompt_template_template.format(
            prior=prob_entries
        )

    def get_room_probs(self):
        synonym_mapping = {
            "couch": ["sofa"],
            "sofa": ["couch"],
        }

        target_categories = [self._target_class] + synonym_mapping.get(self._target_class, [])

        if not any(category in self.knowledge_graph for category in target_categories):
            return {}

        room_probabilities = {}
        for room in REFERENCE_ROOMS:
            for category in target_categories:
                if self.knowledge_graph.has_edge(category, room):
                    room_probabilities[room] = round(self.knowledge_graph[category][room]["weight"] * 100, 1)
                    break  
            else:
                room_probabilities[room] = 0.0
        return room_probabilities

    def _ban_point(self, point_xy):
        if self._goal_view_facing is not None:
            self._objmap.mark_view_inaccessible(self._goal_xy)
        return super()._ban_point(point_xy)

    def _choose_goal_vlm(self) -> None:
        # Prefer suspected viewpoints; otherwise fall back to VLM selection.
        suspects, centers = self._objmap.get_suspected_view_positions(self._target_class, self._curr_xy)
        if isinstance(suspects, np.ndarray) and suspects.size >= 2:
            candidates = [(p, c) for p, c in zip(suspects, centers) if not self.is_banned(p)]
            if candidates:
                if (
                    self._goal_xy is not None
                    and not self.is_banned(self._goal_xy)
                    and self._goal_view_facing is not None
                ):
                    return
                self._goal_view_facing = None
                pos, center = min(candidates, key=lambda pc: self._distance(pc[0], self._curr_xy))
                # pos, center = candidates[int(np.random.randint(len(candidates)))]
                self._goal_xy = np.asarray(pos, dtype=np.float32)
                if isinstance(center, np.ndarray) and center.size >= 2 and not np.any(np.isnan(center)):
                    self._goal_view_facing = center.astype(np.float32)
                if hasattr(self._omap, "set_selected_frontier"):
                    self._omap.set_selected_frontier(self._goal_xy)
                return
        self._goal_view_facing = None

        fr = self._frontiers
        if not isinstance(fr, np.ndarray) or fr.size < 2:
            self._goal_xy = None
            return

        # Sort frontiers by local value around each waypoint
        sorted_pts, sorted_vals = self._vmap.sort_waypoints(fr, radius=self._value_radius_m)

        if (
            np.sqrt(np.var(sorted_vals)) < 0.03
            or np.max(sorted_vals) / np.mean(sorted_vals) < 1.2
            or closest_point_within_threshold(sorted_pts, self._curr_xy, threshold=4) == -1
        ):
            super()._choose_goal_vlm(allow_spin=False)
            return
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

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rgb = obs.get("rgb")
        depth = obs.get("depth")
        gps = obs.get("gps")
        yaw = obs.get("yaw")

        # Precompute agent pose and transform once per step
        x, y = float(gps[0]), float(gps[1])
        camera_position = np.array([x, -y, self._camera_height], dtype=np.float32)
        self._curr_xy = camera_position[:2].copy()
        self._curr_yaw = float(yaw)
        self._tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, self._curr_yaw)

        self._update_value_map(rgb, depth)
        result = super().act(obs)
        result["value_map_viz"] = self._vmap.visualize(self._markers_for_viz(), obstacle_map=self._omap)
        return result
