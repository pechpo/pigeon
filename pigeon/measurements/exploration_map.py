from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np
from habitat import EmbodiedTask, registry
from habitat.core.embodied_task import Measure
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import HeadingSensor, NavigationEpisode, TopDownMap
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat.utils.visualizations import maps
from habitat.config.default_structured_configs import TopDownMapMeasurementConfig
from frontier_exploration.utils.general_utils import habitat_to_xyz


@registry.register_measure
class ExplorationMap(TopDownMap):
    """TopDownMap extension adding target bbox mask and feasibility info.

    Exposes additional fields on the measurement dict:
    - original_map: raw occupancy map before colorization
    - target_bboxes_mask: 2D mask of target object AABBs (same size as map)
    - upper_bound/lower_bound: sim bounds in meters (x,z)
    - grid_resolution: (H, W) of the topdown map grid
    - tf_episodic_to_global: 4x4 transform from episode frame to sim global
    - origin_in_pixel_coord: agent map coord at reset
    - is_feasible: episode feasible without stairs (bool)
    """

    def __init__(
        self,
        sim: HabitatSim,
        config,  # DictConfig
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(sim, config, *args, **kwargs)
        self._is_feasible: bool = True
        self._static_metrics: Dict[str, Any] = {}
        self._task = task
        self._floor_ref_y: float = 0.0
        self._recent_agent_heights = []
        self._recent_agent_planar_positions = []
        self._floor_height_threshold: float = 0.2
        self._floor_stable_window: int = 10
        self._floor_stable_tolerance: float = 0.1
        self._floor_planar_progress_threshold: float = 0.2

    def reset_metric(self, episode: NavigationEpisode, *args: Any, **kwargs: Any) -> None:
        self._static_metrics = {}
        super().reset_metric(episode, *args, **kwargs)
        agent_pos = np.asarray(self._sim.get_agent_state().position, dtype=np.float32)
        self._floor_ref_y = float(agent_pos[1])
        self._recent_agent_heights = [self._floor_ref_y]
        self._recent_agent_planar_positions = [self._get_agent_planar_position(agent_pos)]

        map_ori = self.get_original_map()
        self._draw_target_bbox_mask(episode)

        # Expose info for drawing 3D points on map
        lower_bound, upper_bound = self._sim.pathfinder.get_bounds()
        episodic_start_yaw = HeadingSensor._quat_to_xy_heading(  # type: ignore
            None, quaternion_from_coeff(episode.start_rotation).inverse()
        )[0]
        # Convert Habitat world coordinates to the shared xyz frame used by
        # the planner/mappers and xyz_to_habitat. This keeps episodic/global
        # transforms consistent with how gps and point clouds are defined.
        x, y, z = habitat_to_xyz(np.array(episode.start_position))
        self._static_metrics["original_map"] = map_ori
        self._static_metrics["upper_bound"] = (upper_bound[0], upper_bound[2])
        self._static_metrics["lower_bound"] = (lower_bound[0], lower_bound[2])
        self._static_metrics["grid_resolution"] = self._metric["map"].shape[:2]
        self._static_metrics["tf_episodic_to_global"] = np.array(
            [
                [np.cos(episodic_start_yaw), -np.sin(episodic_start_yaw), 0, x],
                [np.sin(episodic_start_yaw), np.cos(episodic_start_yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):  # type: ignore[override]
        agent_pos = np.asarray(self._sim.get_agent_state().position, dtype=np.float32)
        agent_y = float(agent_pos[1])
        agent_planar_pos = self._get_agent_planar_position(agent_pos)
        self._recent_agent_heights.append(agent_y)
        self._recent_agent_planar_positions.append(agent_planar_pos)
        if len(self._recent_agent_heights) > self._floor_stable_window:
            self._recent_agent_heights.pop(0)
        if len(self._recent_agent_planar_positions) > self._floor_stable_window:
            self._recent_agent_planar_positions.pop(0)
        if self._should_refresh_floor_map(agent_y):
            self._floor_ref_y = agent_y
            self._recent_agent_heights = [agent_y]
            self._recent_agent_planar_positions = [agent_planar_pos]
            self._refresh_floor_map(episode)
        super().update_metric(episode, action, *args, **kwargs)
        new_map = self._metric["map"].copy()
        self._metric["map"] = new_map
        self._metric["is_feasible"] = self._is_feasible
        if "origin_in_pixel_coord" not in self._static_metrics:
            self._static_metrics["origin_in_pixel_coord"] = self._metric["agent_map_coord"]
        self._metric.update(self._static_metrics)

    def _should_refresh_floor_map(self, agent_y: float) -> bool:
        if abs(agent_y - self._floor_ref_y) <= self._floor_height_threshold:
            return False
        if len(self._recent_agent_heights) < self._floor_stable_window:
            return False
        if len(self._recent_agent_planar_positions) < self._floor_stable_window:
            return False
        stable_height = (
            max(self._recent_agent_heights) - min(self._recent_agent_heights)
        ) < self._floor_stable_tolerance
        if not stable_height:
            return False
        planar_progress = float(
            np.linalg.norm(self._recent_agent_planar_positions[-1] - self._recent_agent_planar_positions[0])
        )
        return planar_progress >= self._floor_planar_progress_threshold

    @staticmethod
    def _get_agent_planar_position(agent_pos: np.ndarray) -> np.ndarray:
        return np.asarray(agent_pos[[0, 2]], dtype=np.float32)

    def _refresh_floor_map(self, episode: NavigationEpisode) -> None:
        self._top_down_map = self.get_original_map()
        self._previous_xy_location = [None for _ in range(len(self._sim.habitat_config.agents))]
        if hasattr(episode, "goals"):
            TopDownMap._draw_goals_view_points(self, episode)
            self._draw_goals_aabb(episode)
            self._draw_goals_positions(episode)
        self._draw_target_bbox_mask(episode)

    @staticmethod
    def _semantic_object_from_goal(sem_scene: Any, object_id: Any) -> Any:
        try:
            return sem_scene.objects[object_id]
        except (IndexError, KeyError, TypeError):
            pass

        object_id_str = str(object_id)
        for obj in sem_scene.objects:
            if obj is not None and obj.id == object_id_str:
                return obj
        return None

    def _draw_goals_aabb(self, episode):
        if not self._config.draw_goal_aabbs:
            return
        sem_scene = self._sim.semantic_annotations()
        for goal in episode.goals:
            try:
                sem_obj = self._semantic_object_from_goal(sem_scene, goal.object_id)
                if sem_obj is None:
                    continue
                center = sem_obj.aabb.center
                x_len, _, z_len = sem_obj.aabb.sizes / 2.0
                corners = [
                    center + np.array([x, 0, z])
                    for x, z in [
                        (-x_len, -z_len),
                        (-x_len, z_len),
                        (x_len, z_len),
                        (x_len, -z_len),
                        (-x_len, -z_len),
                    ]
                    if self._is_on_same_floor(center[1])
                ]
                map_corners = [
                    maps.to_grid(
                        p[2],
                        p[0],
                        (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                        sim=self._sim,
                    )
                    for p in corners
                ]
                maps.draw_path(
                    self._top_down_map,
                    map_corners,
                    maps.MAP_TARGET_BOUNDING_BOX,
                    self.line_thickness,
                )
            except AttributeError:
                pass

    def _draw_goals_view_points(self, episode):
        super()._draw_goals_view_points(episode)

        # Determine feasibility without stairs: check connected component containing start
        t_x, t_y = maps.to_grid(
            episode.start_position[2],
            episode.start_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        valid_with_viewpoints = self._top_down_map.copy()
        valid_with_viewpoints[valid_with_viewpoints == maps.MAP_VIEW_POINT_INDICATOR] = maps.MAP_VALID_POINT
        valid_with_viewpoints = cv2.dilate(valid_with_viewpoints, np.ones((3, 3), dtype=np.uint8))
        contours, _ = cv2.findContours(valid_with_viewpoints, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        min_dist = np.inf
        best_idx = 0
        for idx, cnt in enumerate(contours):
            dist = cv2.pointPolygonTest(cnt, (t_y, t_x), True)
            if dist >= 0:
                best_idx = idx
                break
            elif abs(dist) < min_dist:
                min_dist = abs(dist)
                best_idx = idx

        best_cnt = contours[best_idx]
        mask = np.zeros_like(valid_with_viewpoints)
        mask = cv2.drawContours(mask, [best_cnt], 0, 1, -1)  # type: ignore
        masked_values = self._top_down_map[mask.astype(bool)]
        values = set(masked_values.tolist())
        self._is_feasible = maps.MAP_VALID_POINT in values and maps.MAP_VIEW_POINT_INDICATOR in values

    def _draw_target_bbox_mask(self, episode: NavigationEpisode):
        if not isinstance(episode, ObjectGoalNavEpisode):
            return
        bbox_mask = np.zeros_like(self._top_down_map)
        sem_scene = self._sim.semantic_annotations()
        for goal in episode.goals:
            sem_obj = self._semantic_object_from_goal(
                sem_scene, goal.object_id  # type: ignore[attr-defined]
            )
            if sem_obj is None:
                continue
            center = sem_obj.aabb.center
            x_len, _, z_len = sem_obj.aabb.sizes / 2.0
            corners = [
                center + np.array([x, 0, z])
                for x, z in [(-x_len, -z_len), (x_len, z_len)]
                if self._is_on_same_floor(center[1])
            ]
            if not corners:
                continue
            map_corners = [
                maps.to_grid(
                    p[2], p[0], (self._top_down_map.shape[0], self._top_down_map.shape[1]), sim=self._sim
                )
                for p in corners
            ]
            (y1, x1), (y2, x2) = map_corners
            bbox_mask[y1:y2, x1:x2] = 1
        self._static_metrics["target_bboxes_mask"] = bbox_mask


@dataclass
class ExplorationMapMeasurementConfig(TopDownMapMeasurementConfig):
    type: str = ExplorationMap.__name__
    draw_waypoints: bool = True


# Register Hydra config node for Hydra-style defaults resolution
try:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(
        package="habitat.task.measurements.exploration_map",
        group="habitat/task/measurements",
        name="exploration_map",
        node=ExplorationMapMeasurementConfig,
    )
except Exception:
    pass
