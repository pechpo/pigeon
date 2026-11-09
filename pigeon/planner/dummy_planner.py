import os
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
from time import sleep
from copy import deepcopy

import cv2
import cupy as cp
import numpy as np
from PIL import Image
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.core.env import Env
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from .random_object_planner import RandomObjectPlanner
from .vlm_object_planner import VLMObjectPlanner
from .vlm_fusion_planner import VLMFusionPlanner
from ..mapping.fusion_pointcloud_map import FusionPointCloudMap, _DEFAULT_TEXT_PROMPT
from ..mapping.obstacle_map import ObstacleMap
from ..models.vlm import VLMClient
from ..utils.viz import build_vlm_panel, mark_image_bgr
from planning_utils_cpp import a_star_range

DEBUG = False


class Coordinate:
    xy: np.ndarray
    gps: np.ndarray
    world: np.ndarray

    def __init__(self, xy: np.ndarray = None, gps: np.ndarray = None, world: np.ndarray = None, start: Tuple = None):
        self.start_position, self.start_rotation = start
        if xy is not None:
            self.xy = xy
            self.gps = np.array([xy[0], -xy[1]], dtype=np.float32)
            self.world = self.xy_to_world_coord(self.xy)
        elif gps is not None:
            self.xy = np.array([gps[0], -gps[1]], dtype=np.float32)
            self.gps = gps
            self.world = self.xy_to_world_coord(self.xy)
        elif world is not None:
            self.xy = self.world_coord_to_xy(world)
            self.gps = np.array([self.xy[0], -self.xy[1]], dtype=np.float32)
            self.world = world

    def gps_to_world_coord(self, gps):
        origin = np.array(self.start_position, dtype=np.float32)
        # origin[1] = 0
        rotation_world_start = self.start_rotation
        gps = np.array([gps[-1], 0, -gps[0]])
        agent_position = quaternion_rotate_vector(rotation_world_start, gps) + origin
        return agent_position

    def xy_to_world_coord(self, xy):
        xy = xy.copy()
        xy[1] = -xy[1]
        return self.gps_to_world_coord(xy)

    def world_coord_to_gps(self, world_coord):
        origin = np.array(self.start_position, dtype=np.float32)
        rotation_world_start = self.start_rotation
        position = quaternion_rotate_vector(rotation_world_start.inverse(), world_coord - origin)
        return np.array([-position[2], position[0]], dtype=np.float32)

    def world_coord_to_xy(self, world_coord):
        gps = self.world_coord_to_gps(world_coord)
        gps[1] = -gps[1]
        return gps


@dataclass
class Path:
    goal: Coordinate
    path: np.ndarray
    dist: float


def map_to_world(pixels, map_shape, lower_bound, upper_bound):
    """
    pixels: numpy array of shape (N, 2), where each row is (py, px)
    map_shape: (height, width) of the topdown map
    lower_bound, upper_bound: from pathfinder.get_bounds()
    """
    h, w = map_shape
    min_x, _, min_z = lower_bound
    max_x, _, max_z = upper_bound

    py = pixels[:, 0]
    px = pixels[:, 1]

    u = px / w
    v = py / h

    x = min_x + u * (max_x - min_x)
    z = min_z + v * (max_z - min_z)

    world_coords = np.stack([x, np.zeros_like(x), z], axis=1)
    return world_coords


# This class should inherit VLMFusionPlanner. However, as VLMFusionPlanner is not implemented yet, we inherit VLMObjectPlanner for now. Thus confirmation problems are not generated now.
class DummyPlanner(VLMFusionPlanner):
    def __init__(self, data_queue, env):
        super().__init__()
        self.hssd = False
        self.data_queue = data_queue
        self.env = env
        self._min_goal_dist = int(self._omap.pixels_per_meter * self._proximity_threshold * 3)

        self.goals_initialized = False
        self.goals = []
        self.paths = []
        self.unoccupied = None
        self._last_agent_xy_for_confirm = None
        self._objmap.make_dummy()
        self.recorded_xy = np.zeros((0, 2), dtype=np.float32)
        self._target_object_mask = []

    def reset(self):
        super().reset()
        self.goals_initialized = False
        self.goals = []
        self.paths = []
        self.unoccupied = None
        self._last_agent_xy_for_confirm = None
        self.recorded_xy = np.zeros((0, 2), dtype=np.float32)
        self._target_object_mask = []

    def convert_map(self, map):
        converted_map = np.zeros_like(self._omap._map)
        lower_bound, upper_bound = self.env._sim.pathfinder.get_bounds()
        coords = np.stack(np.indices(map.shape), axis=-1).reshape(-1, 2)
        xy = [
            Coordinate(world=c, start=self.start_state).xy
            for c in map_to_world(coords, map.shape, lower_bound, upper_bound)
        ]
        obs_map_coords = self._omap._xy_to_px(np.array(xy))
        valid_mask = np.all(obs_map_coords < map.shape[0], axis=-1) & np.all(obs_map_coords >= 0, axis=-1)
        obs_map_coords = obs_map_coords[valid_mask]
        map = map.reshape(-1)[valid_mask]
        converted_map[obs_map_coords[:, 1], obs_map_coords[:, 0]] = ~map
        return converted_map

    def reset_groundtruth_map(self):
        self.start_state = (
            self.env._sim.get_agent_state().position,
            self.env._sim.get_agent_state().rotation,
        )
        gt_map = self.env._sim.pathfinder.get_topdown_view(
            1 / self._omap.pixels_per_meter, self.env._sim.get_agent_state().position[1]
        )
        self.unoccupied = cv2.dilate(
            self.convert_map(~gt_map).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
        ).astype(bool)
        if DEBUG:
            self.gt_map = deepcopy(self._omap)
            self.gt_map._map = self.convert_map(gt_map).astype(bool)
            self.gt_map._navigable_map = 1 - cv2.dilate(
                self.gt_map._map.astype(np.uint8),
                self.gt_map._navigable_kernel,
                iterations=1,
            ).astype(bool)

    def _astar(self, goal: np.ndarray, start: np.ndarray = None) -> np.ndarray:
        explored = self.unoccupied
        start_xy = self._curr_xy if start is None else start
        agent_px = self._omap._xy_to_px(start_xy.reshape(1, 2))[0]

        if not explored[agent_px[1], agent_px[0]]:
            y_idxs, x_idxs = np.nonzero(explored)
            if len(x_idxs) == 0:
                return np.array([]), np.array([])
            distances = np.sqrt((x_idxs - agent_px[0]) ** 2 + (y_idxs - agent_px[1]) ** 2)
            closest_idx = np.argmin(distances)
            agent_px = np.array([x_idxs[closest_idx], y_idxs[closest_idx]])

        goal_px = self._omap._xy_to_px(goal.reshape(1, 2))[0]
        obstcl_kernel_size = self._omap._navigable_kernel.shape[0]
        for i in range(3, -1, -1):
            min_goal_dist = self._min_goal_dist // (2**i)
            best_path_px = a_star_range(
                explored, explored, tuple(agent_px[::-1]), tuple(goal_px[::-1]), obstcl_kernel_size, min_goal_dist
            )
            if best_path_px != []:
                break
        if best_path_px == []:
            return np.array([]), np.array([])
        best_path_px = np.array(best_path_px)[:, ::-1]
        best_path = self._omap._px_to_xy(best_path_px)
        return best_path, best_path_px

    def get_path_astar(self, goal, start=None):
        p, _ = self._astar(goal, start)
        dist = np.sum(np.linalg.norm(p[1:] - p[:-1], axis=-1)) if len(p) else 100
        return Path(goal=Coordinate(xy=goal, start=self.start_state), path=p, dist=dist)

    def _initialize_goals(self):
        for goal in self.env.current_episode.goals:
            # filter goals not in the same floor
            if abs(goal.position[1] - self.env._sim.get_agent_state().position[1]) > 2:
                continue
            # should we use a navigable point?
            self.goals.append(Coordinate(world=goal.position, start=self.start_state))

    def generate_answer_by_distance(self, candidates):
        if len(candidates) == 1:
            return 0, 0, []
        if len(self.paths[self.nearest_goal_index].path) == 0:
            return np.random.randint(len(candidates)), None, []

        path_to_nearest_goal = self.paths[self.nearest_goal_index]
        distance = path_to_nearest_goal.dist
        # if the distance is invalid, fallback to euclidean distance
        if distance == 100:
            distance = np.linalg.norm(self._curr_xy - path_to_nearest_goal.goal.xy) * 1.2
        distance += 1e-3

        probability_gt = min(0.6, 3 / distance)
        probability = np.ones(len(candidates)) * (1 - probability_gt) / (len(candidates) - 1)
        d = 1000
        best_frontier_id = None
        dist = []
        ids = []
        for idx, candidate in enumerate(candidates):
            d1 = [self.get_path_astar(goal.xy, candidate["xy"]).dist for goal in self.goals]
            d1 = min(d1)
            dist.append(d1)
            ids.append(idx + 1)
            if d1 < d:
                best_frontier_id = idx
                d = d1
        probability[best_frontier_id] = probability_gt
        selected_id = np.random.choice(np.arange(len(candidates)), p=probability)
        return (
            selected_id,
            best_frontier_id,
            dist,
        )

    def _vlm_select(self, candidates):
        if len(candidates) == 0:
            return None, None
        imgs_rgb, _ = self._build_marked_candidates(candidates)
        prompt = self._frontier_selection_prompt_template.format(goal_object=self._vlm_target_name())
        # For visualization, keep the right panel in sync with the candidate set.
        self._vlm_grid_bgr = build_vlm_panel(imgs_rgb, text=None)
        self._vlm_bottom_text = f"Prompt: {prompt}"
        images = imgs_rgb
        selected_id, answer, dist = self.generate_answer_by_distance(candidates)
        images = [Image.fromarray(rgb).convert("RGBA") for rgb in images]
        min_dist = np.min(np.linalg.norm(self.recorded_xy - self._curr_xy, axis=-1)) if len(self.recorded_xy) else 10
        if min_dist >= 1:
            self.recorded_xy = np.concatenate((self.recorded_xy, self._curr_xy.reshape(1, 2)), axis=0)
            if answer is not None and len(candidates) > 1:
                self.data_queue.put(
                    {
                        "images": images,
                        "problem": "<image>" * len(images) + prompt,
                        "answer": str(answer + 1),
                        "distance": dist,
                    }
                )

        reply_text = ""
        self._vlm_grid_bgr = build_vlm_panel(imgs_rgb, text=None)
        self._vlm_bottom_text = f"VLM: {reply_text}"
        chosen = candidates[selected_id]
        return chosen["xy"].astype(np.float32), self._vlm_grid_bgr

    def _detect_and_update(self, depth, rgb):
        return None

    def act(self, obs):
        # we must put it here as the policy is reset before env.reset()
        if self.unoccupied is None:
            self.reset_groundtruth_map()
            self.gen_target_obj_mask()
        gps = obs.get("gps")
        yaw = obs.get("yaw")

        # Precompute agent pose and transform once per step
        x, y = float(gps[0]), float(gps[1])
        camera_position = np.array([x, -y, self._camera_height], dtype=np.float32)
        self._curr_xy = camera_position[:2].copy()
        self._curr_yaw = float(yaw)

        if not self.goals_initialized:
            self.goals_initialized = True
            self._initialize_goals()
            for goal in self.goals:
                self.paths.append(self.get_path_astar(goal.xy))
            if len(self.paths) == 0:
                return self._call_stop()

        self.nearest_goal_index = np.argmin([path.dist for path in self.paths])
        ret = super().act(obs)
        self._maybe_gen_confirm_problem(obs["rgb"])
        if DEBUG:
            self.gt_map.update_agent_traj(self._curr_xy, self._curr_yaw)
            ret["map_viz"] = self.gt_map.visualize()
        return ret

    def _maybe_gen_confirm_problem(self, rgb):
        target_object_mask = self._target_object_mask[1]

        # since the mask is often larger than the actual object, and the margin of the mask
        # tend to fall on the ground, we filter out points with low height
        valid_pts_mask = self._omap.pcd[:, 2] > 0.1
        filtered_pcd_px = self._omap._xy_to_px(self._omap.pcd[valid_pts_mask, :2])
        filtered_pcd_px = filtered_pcd_px[np.all(filtered_pcd_px < target_object_mask.shape[0], axis=-1), :]
        in_sight_vol = cp.sum(target_object_mask[filtered_pcd_px[:, 1], filtered_pcd_px[:, 0]])
        filtered_goal_in_sight = in_sight_vol > 2000
        if (
            self._last_agent_xy_for_confirm is None
            or np.linalg.norm(self._curr_xy - self._last_agent_xy_for_confirm, axis=-1) > 2
        ) and np.random.uniform(0, 1) > (0.95 if filtered_goal_in_sight else 0.99):
            problem = {
                "images": [Image.fromarray(rgb).convert("RGBA")],
                "problem": "<image>" + self._objmap._confirmation_prompt,
                "answer": "True" if filtered_goal_in_sight else "False",
                "distance": [
                    cp.sum(tom[filtered_pcd_px[:, 1], filtered_pcd_px[:, 0]]) for tom in self._target_object_mask
                ],  # actually volume, not distance
            }
            self.data_queue.put(problem)
            self._last_agent_xy_for_confirm = self._curr_xy

        if filtered_goal_in_sight and self._phase != "target":
            goal_indices = np.array(np.nonzero(target_object_mask)).T[:, ::-1]
            nearest_idx = np.argmin(
                np.linalg.norm(goal_indices - self._omap._xy_to_px(self._curr_xy[None, :]), axis=-1)
            )
            tgt_xy = self._omap._px_to_xy(goal_indices[nearest_idx][None, :]).reshape(-1)
            if tgt_xy is not None and len(tgt_xy) == 2:
                goal_xy = np.asarray(tgt_xy, dtype=np.float32)
                if hasattr(self._omap, "set_selected_frontier"):
                    self._omap.set_selected_frontier(goal_xy)
                if self._goal_xy is None:
                    self._goal_xy = goal_xy
                elif self._distance(goal_xy, self._goal_xy) > self._same_goal_thresh:
                    self._should_reset_pointnav = True
                    self._goal_xy = goal_xy
                self._phase = "target"
                self._objmap.make_dummy(goal_xy)

    def gen_target_obj_mask(self):
        centers, sizes = [], []
        if self.hssd:
            for goal in self.env.current_episode.goals:
                obj = self.env._sim.get_rigid_object_manager().get_object_by_id(goal.object_id)
                if abs(goal.position[1] - self.env._sim.get_agent_state().position[1]) > 1:
                    continue
                centers.append(goal.position)
                sizes.append(np.abs(obj.root_scene_node.cumulative_bb.max) * 2)
        else:
            sem_scene = self.env._sim.semantic_annotations()
            for goal in self.env.current_episode.goals:
                if abs(goal.position[1] - self.env._sim.get_agent_state().position[1]) > 2:
                    continue
                centers.append(goal.position)
                sizes.append(sem_scene.objects[goal.object_id].aabb.sizes)
        for i in range(4):
            mask = np.zeros(self._omap._map.shape, dtype=np.uint8)
            for center, size in zip(centers, sizes):
                size = size * (1 - i / 10)
                x_len, _, z_len = size / 2.0
                x_len = max(x_len, 0.1)
                z_len = max(z_len, 0.1)
                world_coord = np.array(
                    [
                        center + np.array([x, 0, z])
                        for x, z in [(-x_len, -z_len), (x_len, -z_len), (x_len, z_len), (-x_len, z_len)]
                    ]
                )
                xy = np.array([Coordinate(world=wc, start=self.start_state).xy for wc in world_coord])
                px = self._omap._xy_to_px(xy)
                cv2.fillPoly(mask, px.astype(np.int32).reshape(1, -1, 2), color=255)
            self._target_object_mask.append(mask.astype(bool))

        if DEBUG:
            self.gt_map.explored_area = self._target_object_mask[1].copy()
            self.gt_map._map = self.gt_map._map & (~self._target_object_mask[1])

    def _call_stop(self):
        return {
            "action": "stop",
            "map_viz": self._visualize_map(),
            "right_panel_viz": self._vlm_grid_bgr,
            "right_panel_text": self._get_bottom_text(),
            "det_seg": None,
            "nav_goal": self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None,
            "stop_called": True,
            "target_detected": False,
        }
