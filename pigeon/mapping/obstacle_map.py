from typing import Any, Callable, List, Optional, Union

import cv2
import numpy as np
from scipy import ndimage
from frontier_exploration.frontier_detection import (
    detect_frontier_waypoints,
    detect_frontiers,
    filter_out_small_unexplored,
    frontier_waypoints,
)
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war, wrap_heading

from pigeon.utils.geometry_utils import extract_yaw, get_projected_min, rho_theta
from pigeon.utils.geometry_utils_gpu import get_point_cloud, transform_points, filter_points_by_height
from pigeon.utils.img_utils import fill_small_holes, reorient_rescale_map
from pigeon.utils.viz import mark_points_bgr
from pigeon.mapping.traj_visualizer import TrajectoryVisualizer


class ObstacleMap:
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    """

    _camera_positions: List[np.ndarray] = []
    _last_camera_yaw: float = 0.0
    _map_dtype: np.dtype = np.dtype(np.float32)
    _map_dtype: np.dtype = np.dtype(bool)
    _frontiers_px: np.ndarray = np.array([])
    frontiers: np.ndarray = np.array([])
    radius_padding_color: tuple = (100, 100, 100)
    _selected_frontier_px: Union[np.ndarray, None] = None
    _banned_frontiers_px: np.ndarray = np.array([])

    def __init__(
        self,
        min_height: float = 0.61,  # 0.66,
        max_height: float = 0.88,
        observe_min_height: Optional[float] = None,
        real_min_height: float = 0.10,
        agent_radius: float = 0.18,
        area_thresh: float = 1.5,  # square meters
        hole_area_thresh: int = 100000,  # square pixels
        size: int = 1000,
        pixels_per_meter: int = 20,
        staircase_protection: bool = True,
    ):
        self.pixels_per_meter = pixels_per_meter
        self.size = size
        self._episode_pixel_origin = np.array([size // 2, size // 2])
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)
        self.explored_area = np.zeros((size, size), dtype=bool)
        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self._observe_map = np.zeros((size, size), dtype=bool)
        self._observe_navigable_map = np.zeros((size, size), dtype=bool)
        self._real_map = np.zeros((size, size), dtype=bool)
        self._real_dilated_map = np.zeros((size, size), dtype=bool)
        self._real_navigable_map = np.zeros((size, size), dtype=bool)
        self._ignore_real_obstacles_for_exploration = False
        self._temporary_obstacle_mask = np.zeros((size, size), dtype=bool)
        self._steppable_map = np.zeros((size, size), dtype=bool)
        self._nonsteppable_map = np.zeros((size, size), dtype=bool)
        self.steppable_explored_area = np.zeros((size, size), dtype=bool)
        self.last_update_frustum_mask = np.zeros((size, size), dtype=bool)
        self._frustum_slack_m = 0.2
        self._view_mask_revision = 0
        self._observe_obstacle_labels: Optional[np.ndarray] = None
        self._observe_obstacle_labels_revision = -1
        self.exposed_exploration_boundary = np.zeros((size, size), dtype=bool)
        self._boundary_exclusion_mask = np.zeros((size, size), dtype=bool)
        self.explored_area_pixels = 0
        self.exposed_boundary_pixels = 0
        self.exploration_degree = 0.0
        # Track geometrically detected downstairs/drop-off candidates separately from
        # VLM-approved downstairs regions.
        self._pending_downstair_map = np.zeros((size, size), dtype=bool)
        self._downstair_map = np.zeros((size, size), dtype=bool)
        self._banned_downstair_map = np.zeros((size, size), dtype=bool)
        # Track detected upstairs regions (via semantics) for viz/logging
        self._upstair_map = np.zeros((size, size), dtype=bool)
        self._banned_upstair_map = np.zeros((size, size), dtype=bool)
        self._upstair_height_map = np.full((size, size), np.nan, dtype=np.float32)
        # When banning a stair component, tolerate small gaps so fragmented stair
        # regions get removed together instead of leaving behind tiny unreachable islands.
        self._stair_component_ban_dilation_px = max(1, int(round(0.15 * float(self.pixels_per_meter))))
        # Ignore tiny nearby stair fragments when choosing a descent component,
        # but fall back if no larger component is available yet.
        self._stair_component_min_area_px = 0.36 * float(self.pixels_per_meter**2)
        # Hard-filter very small stair seed components before frontier flooding so
        # isolated noise does not grow into a staircase candidate.
        self._stair_preflood_min_area_px = 0.04 * float(self.pixels_per_meter**2)
        self._stair_component_flood_dilation_m = 0.3
        # Encourage pointnav descent goals that are both lower and farther along
        # the stair component instead of latching onto a single noisy low pixel.
        self._downstair_descent_distance_weight = 0.35
        self._max_height = max_height
        self._min_height = self._clamp_height(min_height)
        self._real_min_height = self._clamp_height(real_min_height)
        self._observe_min_height = self._clamp_observe_min_height(observe_min_height)
        self._downstair_height = -0.4
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        self._stair_boundary_min_length_m = 0.5 #0.2
        kernel_size = self.pixels_per_meter * agent_radius * 2
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Optional: mark downstairs regions as obstacles to prevent falls
        self._staircase_protection = bool(staircase_protection)
        self._block_downstair_regions = True
        self._downstair_detected: bool = False
        self._upstair_detected: bool = False
        self._latest_agent_xy: Optional[np.ndarray] = None
        self.pending_downstair_frontier = np.array([])
        self.pending_downstair_view_facing = None
        self.downstair_frontier = np.array([])
        self.upstair_frontier = np.array([])
        self.downstair_view_facing = None
        self.upstair_view_facing = None
        self._stair_frontier_records: dict[str, list[dict[str, Any]]] = {}


    def reset(self) -> None:
        self._block_downstair_regions = True
        self._map.fill(0)
        self._camera_positions = []
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)
        self._navigable_map.fill(0)
        self._observe_map.fill(0)
        self._observe_navigable_map.fill(0)
        self._real_map.fill(0)
        self._real_dilated_map.fill(0)
        self._real_navigable_map.fill(0)
        self._ignore_real_obstacles_for_exploration = False
        self._temporary_obstacle_mask.fill(0)
        self._steppable_map.fill(0)
        self._nonsteppable_map.fill(0)
        self.steppable_explored_area.fill(0)
        self._pending_downstair_map.fill(0)
        self._downstair_map.fill(0)
        self._banned_downstair_map.fill(0)
        self._upstair_map.fill(0)
        self._banned_upstair_map.fill(0)
        self._upstair_height_map.fill(np.nan)
        self._downstair_detected = False
        self._upstair_detected = False
        self._latest_agent_xy = None
        self.explored_area.fill(0)
        self.last_update_frustum_mask.fill(0)
        self._view_mask_revision = 0
        self._observe_obstacle_labels = None
        self._observe_obstacle_labels_revision = -1
        self.exposed_exploration_boundary.fill(0)
        self._boundary_exclusion_mask.fill(0)
        self.explored_area_pixels = 0
        self.exposed_boundary_pixels = 0
        self.exploration_degree = 0.0
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self._selected_frontier_px = None
        self._banned_frontiers_px = np.array([])
        self.pending_downstair_frontier = np.array([])
        self.pending_downstair_view_facing = None
        self.downstair_frontier = np.array([])
        self.upstair_frontier = np.array([])
        self.downstair_view_facing = None
        self.upstair_view_facing = None
        self._stair_frontier_records = {}
        self.additional_vis_info = {}

    def _clamp_height(self, min_height: float) -> float:
        upper = max(0.0, float(self._max_height) - 1e-3)
        return float(np.clip(float(min_height), 0.0, upper))

    def update_agent_traj(self, robot_xy: np.ndarray, robot_heading: float) -> None:
        self._latest_agent_xy = np.asarray(robot_xy, dtype=np.float32).reshape(2).copy()
        self._camera_positions.append(robot_xy)
        self._last_camera_yaw = robot_heading

    def _clamp_observe_min_height(self, observe_min_height: Optional[float]) -> float:
        if observe_min_height is None:
            return float(self._min_height)
        return self._clamp_height(observe_min_height)

    def set_observe_min_height(self, observe_min_height: Optional[float]) -> None:
        self._observe_min_height = self._clamp_observe_min_height(observe_min_height)

    def set_min_height(self, min_height: float) -> None:
        self._min_height = self._clamp_height(min_height)

    def set_real_min_height(self, real_min_height: float) -> None:
        self._real_min_height = self._clamp_height(real_min_height)

    def _compute_navigable_mask(self, obstacle_mask: np.ndarray) -> np.ndarray:
        dilated = cv2.dilate(
            obstacle_mask.astype(np.uint8),
            self._navigable_kernel,
            iterations=1,
        ).astype(bool)
        return ~dilated

    def _get_nonblocking_downstair_mask(self) -> np.ndarray:
        if self._block_downstair_regions:
            return np.zeros_like(self._downstair_map, dtype=bool)
        return self._downstair_map & ~self._banned_downstair_map

    def _get_effective_obstacle_mask(self) -> np.ndarray:
        return np.asarray(self._map, dtype=bool) & ~self._get_nonblocking_downstair_mask()

    def _get_effective_real_obstacle_mask(self) -> np.ndarray:
        return np.asarray(self._real_map, dtype=bool) & ~self._get_nonblocking_downstair_mask()

    def _get_exploration_real_obstacle_mask(self) -> np.ndarray:
        if self._ignore_real_obstacles_for_exploration:
            return np.zeros_like(self._real_map, dtype=bool)
        return self._get_effective_real_obstacle_mask()

    def _get_exploration_navigable_map(self) -> np.ndarray:
        if self._ignore_real_obstacles_for_exploration:
            return self._navigable_map
        return self._real_navigable_map

    def set_ignore_real_obstacles_for_exploration(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._ignore_real_obstacles_for_exploration == enabled:
            return
        prev_explored = self.explored_area.copy()
        self._ignore_real_obstacles_for_exploration = enabled
        self._refresh_real_obstacle_padding()
        self._prune_explored_area_from_obstacles()
        self._restore_explored_area_if_collapsed(prev_explored)
        self._refresh_frontiers()
        self._update_exploration_degree()

    def set_downstair_obstacle_blocking(self, blocking: bool) -> None:
        blocking = bool(blocking)
        if self._block_downstair_regions == blocking:
            return
        self._block_downstair_regions = blocking
        self._navigable_map = self._compute_navigable_mask(self._get_effective_obstacle_mask())
        self._refresh_real_obstacle_padding()
        self._prune_explored_area_from_obstacles()
        self._refresh_frontiers()
        self._update_exploration_degree()

    @property
    def view_mask_revision(self) -> int:
        return int(self._view_mask_revision)

    def _invalidate_view_mask_cache(self) -> None:
        self._view_mask_revision += 1
        self._observe_obstacle_labels = None
        self._observe_obstacle_labels_revision = -1

    def _get_observe_obstacle_labels(self) -> np.ndarray:
        if (
            isinstance(self._observe_obstacle_labels, np.ndarray)
            and self._observe_obstacle_labels.shape == self._observe_navigable_map.shape
            and self._observe_obstacle_labels_revision == self._view_mask_revision
        ):
            return self._observe_obstacle_labels
        obstacle_mask = ~self._observe_navigable_map.astype(bool, copy=False)
        labels, _ = ndimage.label(obstacle_mask)
        self._observe_obstacle_labels = labels.astype(np.int32, copy=False)
        self._observe_obstacle_labels_revision = self._view_mask_revision
        return self._observe_obstacle_labels

    def _sample_ray_points(self, start_xy: np.ndarray, angle_rad: float, length_px: float) -> np.ndarray:
        num_samples = max(2, int(np.ceil(float(length_px) * 2.0)) + 1)
        end_xy = start_xy.astype(np.float32) + np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32) * float(
            length_px
        )
        xs = np.rint(np.linspace(float(start_xy[0]), float(end_xy[0]), num_samples)).astype(np.int32)
        ys = np.rint(np.linspace(float(start_xy[1]), float(end_xy[1]), num_samples)).astype(np.int32)
        points = np.stack([xs, ys], axis=1)
        if len(points) <= 1:
            return points
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.any(points[1:] != points[:-1], axis=1)
        return points[keep]

    def _apply_ray_slack_to_frustum_mask(
        self,
        base_mask: np.ndarray,
        current_point: np.ndarray,
        current_angle: float,
        fov_deg: float,
        max_line_len: float,
    ) -> np.ndarray:
        if self._frustum_slack_m <= 0.0 or not np.any(base_mask) or not np.any(self._observe_navigable_map):
            return base_mask
        obstacle_labels = self._get_observe_obstacle_labels()
        if not np.any(obstacle_labels):
            return base_mask

        slack_px = float(self._frustum_slack_m) * float(self.pixels_per_meter)
        if slack_px <= 0.0:
            return base_mask

        h, w = base_mask.shape
        curr_pt_cv2 = current_point[::-1].astype(np.int32)
        angle_cv2_deg = float(np.rad2deg(wrap_heading(-float(current_angle) + np.pi / 2)))
        num_rays = max(32, int(np.ceil(float(max_line_len + slack_px) * np.deg2rad(float(fov_deg)))))
        ray_angles = np.deg2rad(np.linspace(angle_cv2_deg - fov_deg / 2.0, angle_cv2_deg + fov_deg / 2.0, num_rays))

        slack_mask = base_mask.astype(bool, copy=True)
        start_xy = curr_pt_cv2.astype(np.float32)
        for ray_angle in ray_angles:
            ray_points = self._sample_ray_points(start_xy, float(ray_angle), float(max_line_len + slack_px))
            if ray_points.size == 0:
                continue
            in_bounds = (
                (ray_points[:, 0] >= 0)
                & (ray_points[:, 0] < w)
                & (ray_points[:, 1] >= 0)
                & (ray_points[:, 1] < h)
            )
            ray_points = ray_points[in_bounds]
            if ray_points.size == 0:
                continue
            labels = obstacle_labels[ray_points[:, 1], ray_points[:, 0]]
            hit_indices = np.flatnonzero(labels > 0)
            if len(hit_indices) == 0:
                continue
            hit_idx = int(hit_indices[0])
            hit_label = int(labels[hit_idx])
            hit_xy = ray_points[hit_idx].astype(np.float32)
            for point_xy, label in zip(ray_points[hit_idx:], labels[hit_idx:]):
                if int(label) != hit_label:
                    break
                if np.linalg.norm(point_xy.astype(np.float32) - hit_xy) > slack_px + 1e-6:
                    break
                slack_mask[point_xy[1], point_xy[0]] = True
        return slack_mask

    def _build_view_frustum_mask(
        self,
        top_down_map: np.ndarray,
        current_point: np.ndarray,
        current_angle: float,
        topdown_fov: float,
        max_depth: float,
        apply_slack: bool = True,
    ) -> np.ndarray:
        frustum_area = reveal_fog_of_war(
            top_down_map=top_down_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=current_point,
            current_angle=current_angle,
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        ).astype(bool)
        if apply_slack:
            frustum_area = self._apply_ray_slack_to_frustum_mask(
                frustum_area,
                current_point=current_point,
                current_angle=current_angle,
                fov_deg=float(np.rad2deg(topdown_fov)),
                max_line_len=float(max_depth) * float(self.pixels_per_meter),
            )
        return cv2.dilate(frustum_area.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)

    def recompute_view_frustum_mask(
        self,
        tf_camera_to_episodic: np.ndarray,
        topdown_fov: float,
        max_depth: float,
    ) -> np.ndarray:
        agent_xy_location = np.asarray(tf_camera_to_episodic[:2, 3], dtype=np.float32).reshape(1, 2)
        agent_pixel_location = self._xy_to_px(agent_xy_location)[0]
        return self._build_view_frustum_mask(
            self._observe_navigable_map,
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            topdown_fov=topdown_fov,
            max_depth=max_depth,
            apply_slack=True,
        )

    def _refresh_real_obstacle_padding(self) -> None:
        combined_real_obstacles = self._get_exploration_real_obstacle_mask() | self._temporary_obstacle_mask
        self._real_dilated_map = cv2.dilate(
            combined_real_obstacles.astype(np.uint8),
            self._navigable_kernel,
            iterations=1,
        ).astype(bool)
        self._real_navigable_map = ~self._real_dilated_map

    def _prune_explored_area_from_obstacles(self) -> None:
        self.explored_area[self._navigable_map == 0] = 0
        self.explored_area[self._real_dilated_map] = 0

    def _restore_explored_area_if_collapsed(self, prev_explored: np.ndarray) -> None:
        prev_pixels = int(np.count_nonzero(prev_explored))
        if prev_pixels == 0:
            return
        if 2 * int(np.count_nonzero(self.explored_area)) < prev_pixels:
            self.explored_area[...] = np.asarray(prev_explored, dtype=bool)

    def _restrict_exploration_by_real_obstacles(
        self, explored_mask: np.ndarray, agent_pixel_location: np.ndarray
    ) -> np.ndarray:
        candidate = np.asarray(explored_mask, dtype=bool)
        if not np.any(candidate):
            return np.zeros_like(candidate, dtype=bool)
        candidate &= ~self._real_dilated_map
        seed_x, seed_y = int(agent_pixel_location[0]), int(agent_pixel_location[1])
        h, w = candidate.shape
        if seed_x < 0 or seed_x >= w or seed_y < 0 or seed_y >= h:
            return np.zeros_like(candidate, dtype=bool)
        if not candidate[seed_y, seed_x]:
            radius_px = max(1, int(round(0.5 * float(self.pixels_per_meter))))
            x0, x1 = max(0, seed_x - radius_px), min(w, seed_x + radius_px + 1)
            y0, y1 = max(0, seed_y - radius_px), min(h, seed_y + radius_px + 1)
            ys, xs = np.where(candidate[y0:y1, x0:x1])
            if len(xs) == 0:
                return np.zeros_like(candidate, dtype=bool)
            d2 = (xs + x0 - seed_x) ** 2 + (ys + y0 - seed_y) ** 2
            idx = int(np.argmin(d2))
            seed_x = int(xs[idx] + x0)
            seed_y = int(ys[idx] + y0)
        seed = np.zeros_like(candidate, dtype=bool)
        seed[seed_y, seed_x] = True
        return ndimage.binary_propagation(
            seed,
            mask=candidate,
            structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        )

    def update_map(
        self,
        depth: Union[np.ndarray, Any],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        explore: bool = True,
        update_obstacles: bool = True,
        gpu_worker: Any = None,
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """
        # Precompute agent pixel location (used by optional staircase protection)
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        self._latest_agent_xy = np.asarray(agent_xy_location, dtype=np.float32).reshape(2).copy()
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]

        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            # Valid depth mask
            mask = scaled_depth < max_depth

            # 1) Always start with a dense HxW point cloud in camera frame (CuPy)
            dense_points_cam = get_point_cloud(scaled_depth, None, fx, fy, preserve_hw=True)
            # 2) Shared transform: transform dense cloud once, preserving HxW shape (CuPy)
            self.dense_points_cam = dense_points_cam
            dense_points_epi = transform_points(tf_camera_to_episodic, dense_points_cam)
            self.dense_points_epi = dense_points_epi  # for reuse in object fusion
            self.out_of_range_mask = scaled_depth > max_depth * 0.95  # for reuse in object fusion
            # 3) Selected points for obstacle processing (N x 3) using a CuPy mask
            import cupy as cp

            sel_points_epi = dense_points_epi[cp.asarray(mask)]
            # Height-filtered obstacles (CuPy)
            obstacle_cloud = filter_points_by_height(sel_points_epi, self._min_height, self._max_height)
            observe_obstacle_cloud = filter_points_by_height(sel_points_epi, self._observe_min_height, self._max_height)
            real_obstacle_cloud = filter_points_by_height(sel_points_epi, self._real_min_height, self._max_height)
            nonsteppable_obstacle_cloud = filter_points_by_height(obstacle_cloud, 0.2, self._max_height).get()
            dense_points_epi = dense_points_epi.get()
            sel_points_epi = sel_points_epi.get()
            obstacle_cloud = obstacle_cloud.get()
            observe_obstacle_cloud = observe_obstacle_cloud.get()
            real_obstacle_cloud = real_obstacle_cloud.get()

            self.pcd = sel_points_epi  # for reuse in object fusion

            if self._staircase_protection:
                # Build an HxW mask of low-height points in episodic frame and clean small regions
                low_mask_inv = np.ones_like(mask, dtype=np.uint8)
                low_mask_raw = (dense_points_epi[..., 2] < self._downstair_height) & mask
                if np.any(low_mask_raw):
                    low_mask_inv[low_mask_raw] = 0
                    # Fill small low regions and keep only large consistent drops
                    low_mask = (1 - fill_small_holes(low_mask_inv, 2000)).astype(bool)  # 1000
                    # Align to selected points
                    low_sel = low_mask[mask]
                    if np.any(low_sel) and sel_points_epi.shape[0] > 0:
                        sel_px = self._xy_to_px(sel_points_epi[:, :2])
                        c, v = get_projected_min(sel_px[low_sel], sel_points_epi[low_sel, 2])
                        deep_drop = v <= self._downstair_height
                        if np.any(deep_drop):
                            self.mark_stairs_unreachable(tf_camera_to_episodic, c[deep_drop], v[deep_drop])

            # Populate topdown map with obstacle locations
            xy_points = obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            pixel_points = pixel_points[np.all(pixel_points < self._map.shape[0], axis=-1), :]
            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            # Update the navigable area, which is an inverse of the obstacle map after a
            # dilation operation to accommodate the robot's radius.
            self._navigable_map = self._compute_navigable_mask(self._get_effective_obstacle_mask())

            xy_points = observe_obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            pixel_points = pixel_points[np.all(pixel_points < self._map.shape[0], axis=-1), :]
            self._observe_map[pixel_points[:, 1], pixel_points[:, 0]] = 1
            self._observe_navigable_map = self._compute_navigable_mask(self._observe_map)
            self._invalidate_view_mask_cache()

            xy_points = real_obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            pixel_points = pixel_points[np.all(pixel_points < self._map.shape[0], axis=-1), :]
            self._real_map[pixel_points[:, 1], pixel_points[:, 0]] = 1
            self._refresh_real_obstacle_padding()

            xy_points = nonsteppable_obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            pixel_points = pixel_points[np.all(pixel_points < self._map.shape[0], axis=-1), :]
            self._nonsteppable_map[pixel_points[:, 1], pixel_points[:, 0]] = 1
            self._steppable_map = self._compute_navigable_mask(self._nonsteppable_map)

        # Build a frustum mask using the current sensing depth so history uses the
        # same range as the depth sensor instead of a fixed visualization distance.
        self.last_update_frustum_mask = self._build_view_frustum_mask(
            self._observe_navigable_map,
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            topdown_fov=topdown_fov,
            max_depth=max_depth,
            apply_slack=True,
        )

        if not explore:
            return

        # Update the explored area using the actual sensing depth.
        prev_explored = self.explored_area.copy()
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        new_explored_area = self._restrict_exploration_by_real_obstacles(new_explored_area > 0, agent_pixel_location)
        self.explored_area[new_explored_area] = 1
        self._prune_explored_area_from_obstacles()
        self.steppable_explored_area[new_explored_area] = 1
        self.steppable_explored_area[self._steppable_map == 0] = 0
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
                if dist >= 0:
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
            # we must fill it in place here, as explored_area is shared with objmap
            self.explored_area[...] = new_area.astype(bool)
        self._prune_explored_area_from_obstacles()
        self._restore_explored_area_if_collapsed(prev_explored)

        self._refresh_frontiers()
        self._update_exploration_degree()

    def mark_stairs_unreachable(
        self, tf_camera_to_episodic: np.ndarray, coords: np.ndarray, values: np.ndarray
    ) -> None:
        """Accumulate likely downstairs/drop-off regions as unconfirmed candidates.

        Args:
            agent_pixel_loc: Agent pixel location (x, y) on the map.
            coords: Pixel coordinates along the low-edge front (N, 2).
            values: Corresponding world z values (N,) used to scale the unsafe corridor.
        """
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_loc = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]

        mask = np.zeros(self._map.shape, dtype=bool)
        for coord, value in zip(coords, values):
            # Scale the unsafe corridor length based on depth drop magnitude
            ratio = self._max_height / (self._max_height + (-float(value)))
            line_points = np.linspace(
                (agent_pixel_loc * (1 - ratio) + coord * ratio).astype(int),
                coord,
                num=500,
            ).astype(int)
            line_points = np.unique(line_points, axis=0)
            # Avoid immediate vicinity of the agent
            line_points = line_points[np.linalg.norm(line_points - agent_pixel_loc, axis=-1) > 8]
            # Only mark within explored area to reduce false positives
            valid = self.explored_area[line_points[:, 1], line_points[:, 0]]
            if np.any(valid):
                lp = line_points[valid]
                mask[lp[:, 1], lp[:, 0]] = 1
        if np.any(mask):
            self._pending_downstair_map |= mask
            self.pending_downstair_frontier, self.pending_downstair_view_facing = self._summarize_stair_mask(
                self._pending_downstair_map
            )

    def _update_stair_height_map(self, height_map: np.ndarray, px: np.ndarray, heights: np.ndarray) -> None:
        curr = height_map[px[:, 1], px[:, 0]]
        update = ~np.isfinite(curr) | (np.abs(heights) < np.abs(curr))
        height_map[px[update, 1], px[update, 0]] = heights[update]

    def _largest_stair_component(self, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return mask
        labels, num = ndimage.label(mask)
        if num <= 1:
            return mask
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        return labels == int(np.argmax(sizes))

    def _stair_other_obstacle_dilated_mask(self, seed_mask: np.ndarray) -> np.ndarray:
        """Dilated obstacles excluding the current stair seeds themselves."""
        obstacle_mask = self._map if self._ignore_real_obstacles_for_exploration else self._real_map
        other_obstacles = (obstacle_mask | self._temporary_obstacle_mask) & ~np.asarray(seed_mask, dtype=bool)
        return cv2.dilate(
            other_obstacles.astype(np.uint8),
            self._navigable_kernel,
            iterations=1,
        ).astype(bool)

    def _flood_stair_components(self, seed_mask: np.ndarray, radius_px: int) -> np.ndarray:
        """Expand stair seeds without crossing other obstacles or their padding."""
        if not np.any(seed_mask):
            return np.zeros_like(seed_mask, dtype=bool)
        if radius_px <= 0:
            return np.asarray(seed_mask, dtype=bool).copy()
        blocked = self._stair_other_obstacle_dilated_mask(seed_mask)
        allowed = (~blocked) | np.asarray(seed_mask, dtype=bool)
        structure = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        )
        return ndimage.binary_dilation(
            np.asarray(seed_mask, dtype=bool),
            structure=structure,
            iterations=int(radius_px),
            mask=allowed,
        )

    def _get_active_stair_mask(self, kind: str) -> np.ndarray:
        stair_kind = "down" if str(kind).startswith("down") else "up"
        if stair_kind == "down":
            return self._downstair_map & ~self._banned_downstair_map
        return self._upstair_map & ~self._banned_upstair_map

    def _largest_bannable_stair_component(self, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return mask
        dilation_px = int(getattr(self, "_stair_component_ban_dilation_px", 0))
        if dilation_px <= 0:
            return self._largest_stair_component(mask)
        expanded_mask = self._flood_stair_components(mask, dilation_px)
        expanded_component = self._largest_stair_component(expanded_mask)
        return mask & expanded_component

    def _filter_preflood_stair_components(self, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        min_area_px = float(getattr(self, "_stair_preflood_min_area_px", 0.0))
        if min_area_px <= 0.0:
            return np.asarray(mask, dtype=bool).copy()
        labels, num = ndimage.label(mask)
        if num <= 0:
            return np.zeros_like(mask, dtype=bool)
        filtered = np.zeros_like(mask, dtype=bool)
        for label in range(1, num + 1):
            component = labels == label
            if int(np.count_nonzero(component)) < min_area_px:
                continue
            filtered |= component
        return filtered

    def _default_stair_component_flood_radius_px(self, component_dilation_m: Optional[float] = None) -> int:
        dilation_m = self._stair_component_flood_dilation_m if component_dilation_m is None else component_dilation_m
        return max(1, int(round(float(dilation_m) * float(self.pixels_per_meter))))

    def _compute_flooded_stair_components(
        self,
        kind: str,
        component_dilation_m: Optional[float] = None,
    ) -> list[dict[str, np.ndarray]]:
        active_mask = self._get_active_stair_mask(kind)
        filtered_mask = self._filter_preflood_stair_components(active_mask)
        seed_mask = filtered_mask if np.any(filtered_mask) else active_mask
        if not np.any(seed_mask):
            return []

        if np.any(filtered_mask):
            expanded_mask = self._flood_stair_components(
                seed_mask,
                self._default_stair_component_flood_radius_px(component_dilation_m),
            )
        else:
            expanded_mask = np.asarray(seed_mask, dtype=bool).copy()
        labels, num = ndimage.label(expanded_mask)
        if num <= 0:
            return []

        components = []
        for label in range(1, num + 1):
            expanded_component = labels == label
            component_seed_mask = seed_mask & expanded_component
            if not np.any(component_seed_mask) and not np.any(expanded_component):
                continue
            components.append(
                {
                    "seed_mask": component_seed_mask,
                    "expanded_mask": expanded_component,
                }
            )
        return components


    def _get_flooded_stair_components(
        self,
        kind: str,
        component_dilation_m: Optional[float] = None,
    ) -> list[dict[str, np.ndarray]]:
        components = self._compute_flooded_stair_components(kind, component_dilation_m=component_dilation_m)
        return components

    def _select_flooded_stair_component(
        self,
        kind: str,
        near_xy: Optional[np.ndarray] = None,
        component_dilation_m: Optional[float] = None,
    ) -> Optional[dict[str, np.ndarray]]:
        components = self._get_flooded_stair_components(kind, component_dilation_m=component_dilation_m)
        if not components:
            return None
        if near_xy is None:
            return max(components, key=lambda record: int(np.count_nonzero(record["expanded_mask"])))

        near_px = self._xy_to_px(np.asarray(near_xy, dtype=np.float32).reshape(1, 2))[0].astype(np.float32)
        best_component = None
        fallback_component = None
        best_dist = np.inf
        fallback_dist = np.inf
        min_area_px = int(getattr(self, "_stair_component_min_area_px", 0))
        for record in components:
            seed_mask = record["seed_mask"]
            expanded_mask = record["expanded_mask"]
            area_mask = seed_mask if np.any(seed_mask) else expanded_mask
            rows, cols = np.where(area_mask)
            if len(rows) == 0:
                continue
            px = np.stack([cols, rows], axis=1).astype(np.float32)
            dist = float(np.min(np.linalg.norm(px - near_px.reshape(1, 2), axis=1)))
            if dist < fallback_dist:
                fallback_dist = dist
                fallback_component = record
            if int(np.count_nonzero(area_mask)) < min_area_px:
                continue
            if dist < best_dist:
                best_dist = dist
                best_component = record
        return best_component or fallback_component

    def _refresh_stair_summary(self, kind: str) -> None:
        active_mask = self._get_active_stair_mask(kind)
        frontier, view_facing = self._summarize_stair_mask(active_mask)
        if str(kind).startswith("down"):
            self.downstair_frontier = frontier
            self.downstair_view_facing = view_facing
            self._downstair_detected = bool(np.any(active_mask))
        else:
            self.upstair_frontier = frontier
            self.upstair_view_facing = view_facing
            self._upstair_detected = bool(np.any(active_mask))

    def ban_current_stair_component(self, kind: str) -> bool:
        active_mask = self._get_active_stair_mask(kind)
        if not np.any(active_mask):
            return False
        component = self._largest_bannable_stair_component(active_mask)
        if not np.any(component):
            return False
        if str(kind).startswith("down"):
            self._banned_downstair_map |= component
        else:
            self._banned_upstair_map |= component
        self._refresh_stair_summary(kind)
        return True

    def _stair_component_explored_mask(
        self,
        component_mask: np.ndarray,
        seed_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        explored_mask = self.explored_area & self._get_exploration_navigable_map()
        if seed_mask is not None and np.any(seed_mask):
            explored_mask &= ~self._stair_other_obstacle_dilated_mask(seed_mask)
        explored_mask &= ~component_mask
        return explored_mask

    def _stair_frontier_is_long_enough(self, frontier: np.ndarray) -> bool:
        min_length_m = float(getattr(self, "_stair_boundary_min_length_m", 0.0))
        if min_length_m <= 0.0:
            return len(frontier) > 1
        min_length_px = max(1, int(np.ceil(min_length_m * float(self.pixels_per_meter))))
        pts = np.asarray(frontier, dtype=np.float32).reshape(-1, 2)
        if len(pts) <= 1:
            return False
        seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return float(np.sum(seg_lengths)) >= float(min_length_px)

    def get_stair_frontier_candidates(
        self,
        kind: str,
        component_dilation_m: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._stair_frontier_records[str(kind)] = []
        components = self._get_flooded_stair_components(kind, component_dilation_m=component_dilation_m)
        if not components:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

        frontier_px_list = []
        view_px_list = []
        frontier_records = []
        for record in components:
            component = record["expanded_mask"]
            seed_component = record["seed_mask"]
            rows, cols = np.where(seed_component if np.any(seed_component) else component)
            if len(rows) == 0:
                continue
            explored_mask = self._stair_component_explored_mask(component, seed_mask=seed_component)
            if not np.any(explored_mask):
                continue
            frontiers = detect_frontiers(
                (component | explored_mask).astype(np.uint8, copy=False),
                explored_mask.astype(np.uint8, copy=True),
                -1,
            )
            frontiers = [frontier for frontier in frontiers if self._stair_frontier_is_long_enough(frontier)]
            if len(frontiers) == 0:
                continue
            frontier_px = frontier_waypoints(frontiers)
            frontier_px = np.rint(np.asarray(frontier_px, dtype=np.float32)).astype(np.int32)
            center_px = np.array([[int(np.rint(cols.mean())), int(np.rint(rows.mean()))]], dtype=np.int32)
            seed_mask = seed_component if np.any(seed_component) else component
            for pt, frontier in zip(frontier_px, frontiers):
                frontier_mask = np.zeros_like(component, dtype=bool)
                frontier_pts = np.rint(np.asarray(frontier, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
                frontier_mask[frontier_pts[:, 1], frontier_pts[:, 0]] = True
                frontier_px_list.append(pt.reshape(1, 2))
                view_px_list.append(center_px.copy())
                frontier_records.append(
                    {
                        "kind": str(kind),
                        "frontier_px": pt.copy(),
                        "seed_mask": seed_mask.copy(),
                        "component_mask": component.copy(),
                        "explored_mask": explored_mask.copy(),
                        "frontier_mask": frontier_mask,
                    }
                )

        if not frontier_px_list:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        frontier_px = np.vstack(frontier_px_list)
        view_px = np.vstack(view_px_list)
        self._stair_frontier_records[str(kind)] = frontier_records
        return self._px_to_xy(frontier_px), self._px_to_xy(view_px)

    def _stair_frontier_record(
        self,
        frontier_xy: np.ndarray,
        kind: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        frontier_px = tuple(int(v) for v in self._xy_to_px(np.asarray(frontier_xy, dtype=np.float32).reshape(1, 2))[0])
        if kind is not None:
            records = self._stair_frontier_records.get(str(kind), [])
            for rec in records:
                if tuple(int(v) for v in rec["frontier_px"]) == frontier_px:
                    return rec
            return None
        for records in self._stair_frontier_records.values():
            for rec in records:
                if tuple(int(v) for v in rec["frontier_px"]) == frontier_px:
                    return rec
        return None


    def _select_stair_component(self, mask: np.ndarray, near_xy: Optional[np.ndarray] = None) -> np.ndarray:
        if not np.any(mask):
            return mask
        if near_xy is None:
            return self._largest_stair_component(mask)
        labels, num = ndimage.label(mask)
        if num <= 1:
            return mask
        near_px = self._xy_to_px(np.asarray(near_xy, dtype=np.float32).reshape(1, 2))[0].astype(np.float32)
        best_component = np.zeros_like(mask, dtype=bool)
        fallback_component = np.zeros_like(mask, dtype=bool)
        best_dist = np.inf
        fallback_dist = np.inf
        min_area_px = int(getattr(self, "_stair_component_min_area_px", 0))
        for label in range(1, num + 1):
            component = labels == label
            rows, cols = np.where(component)
            if len(rows) == 0:
                continue
            px = np.stack([cols, rows], axis=1).astype(np.float32)
            dist = float(np.min(np.linalg.norm(px - near_px.reshape(1, 2), axis=1)))
            if dist < fallback_dist:
                fallback_dist = dist
                fallback_component = component
            if len(rows) < min_area_px:
                continue
            if dist < best_dist:
                best_dist = dist
                best_component = component
        if np.any(best_component):
            return best_component
        if np.any(fallback_component):
            return fallback_component
        return self._largest_stair_component(mask)

    def _summarize_stair_mask(self, mask: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if not np.any(mask):
            return np.array([]), None

        component = self._largest_stair_component(mask)
        rows, cols = np.where(component)
        px = np.stack([cols, rows], axis=1).astype(np.float32)
        view_facing = self._px_to_xy(np.rint(px.mean(axis=0, keepdims=True)).astype(int))

        agent_xy = self._latest_agent_xy
        if agent_xy is None and len(self._camera_positions) > 0:
            agent_xy = np.asarray(self._camera_positions[-1], dtype=np.float32).reshape(2)
        if agent_xy is not None:
            agent_px = self._xy_to_px(agent_xy.reshape(1, 2))[0].astype(np.float32)
            dist = np.linalg.norm(px - agent_px.reshape(1, 2), axis=1)
            frontier_px = np.rint(px[int(np.argmin(dist))].reshape(1, 2)).astype(int)
        else:
            frontier_px = np.rint(px.mean(axis=0, keepdims=True)).astype(int)
        return self._px_to_xy(frontier_px), view_facing

    def get_downstair_descent_component_masks(
        self,
        entry_xy: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        record = self._select_flooded_stair_component("down", near_xy=entry_xy)
        if record is None:
            shape = self._downstair_map.shape
            return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
        return record["expanded_mask"].copy(), record["seed_mask"].copy()

    def get_downstair_descent_component_mask(self, entry_xy: Optional[np.ndarray] = None) -> np.ndarray:
        component_mask, _ = self.get_downstair_descent_component_masks(entry_xy)
        return component_mask


    def get_downstair_overlay_mask(self, include_pending: bool = True) -> np.ndarray:
        overlay_mask = self._downstair_map & ~self._banned_downstair_map
        if include_pending:
            overlay_mask |= self._pending_downstair_map
        return overlay_mask

    def approve_pending_downstair(self) -> bool:
        """Promote the visible portion of the downstairs candidate to a blocking region."""
        if not np.any(self._pending_downstair_map):
            return False

        frustum_mask = getattr(self, "last_update_frustum_mask", None)
        if not isinstance(frustum_mask, np.ndarray) or frustum_mask.shape != self._pending_downstair_map.shape:
            return False

        approved_mask = self._pending_downstair_map & frustum_mask.astype(bool, copy=False)
        if not np.any(approved_mask):
            return False

        prev_explored = self.explored_area.copy()
        self._map[approved_mask] = 1
        self._real_map[approved_mask] = 1
        self._downstair_map[approved_mask] = 1
        self._refresh_stair_summary("down")

        self._pending_downstair_map &= ~approved_mask
        if np.any(self._pending_downstair_map):
            pending_frontier, pending_view_facing = self._summarize_stair_mask(self._pending_downstair_map)
            self.pending_downstair_frontier = pending_frontier
            self.pending_downstair_view_facing = pending_view_facing
        else:
            self.pending_downstair_frontier = np.array([])
            self.pending_downstair_view_facing = None
        self._navigable_map = self._compute_navigable_mask(self._get_effective_obstacle_mask())
        self._refresh_real_obstacle_padding()
        self._prune_explored_area_from_obstacles()
        self._restore_explored_area_if_collapsed(prev_explored)
        self._refresh_frontiers()
        self._update_exploration_degree()
        return True

    def _get_frontiers(self) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(
            self._get_exploration_navigable_map().astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,
        )
        if len(frontiers) == 0 or not np.any(self._boundary_exclusion_mask):
            return frontiers
        keep_mask = ~self._boundary_exclusion_mask[frontiers[:, 1].astype(int), frontiers[:, 0].astype(int)]
        return frontiers[keep_mask]

    def _refresh_frontiers(self) -> None:
        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
            return
        self.frontiers = self._px_to_xy(self._frontiers_px)

    def _update_exploration_degree(self) -> None:
        """Measure exploration progress as explored-area / exposed-boundary.

        The exposed boundary is the portion of the explored-region boundary that is
        adjacent to still-unexplored navigable space, i.e. not blocked by obstacles.
        """
        self.explored_area_pixels = int(np.count_nonzero(self.explored_area))
        if self.explored_area_pixels == 0:
            self.exposed_exploration_boundary.fill(0)
            self.exposed_boundary_pixels = 0
            self.exploration_degree = 0.0
            return

        exploration_navigable_map = self._get_exploration_navigable_map()
        frontier_explored_area = filter_out_small_unexplored(
            exploration_navigable_map.astype(np.uint8),
            self.explored_area.astype(np.uint8),
            self._area_thresh_in_pixels,
        ).astype(bool)
        unknown_navigable = (~frontier_explored_area) & exploration_navigable_map
        if not np.any(unknown_navigable):
            self.exposed_exploration_boundary.fill(0)
            self.exposed_boundary_pixels = 0
            self.exploration_degree = float("inf")
            return

        boundary_kernel = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )
        exposed_boundary = frontier_explored_area & cv2.dilate(
            unknown_navigable.astype(np.uint8),
            boundary_kernel,
            iterations=1,
        ).astype(bool)
        if np.any(self._boundary_exclusion_mask):
            exposed_boundary &= ~self._boundary_exclusion_mask
        self.exposed_exploration_boundary[...] = exposed_boundary
        self.exposed_boundary_pixels = int(np.count_nonzero(exposed_boundary))
        if self.exposed_boundary_pixels == 0:
            self.exploration_degree = float("inf")
            return
        self.exploration_degree = float(self.explored_area_pixels) / float(self.exposed_boundary_pixels)

    def set_exposed_boundary_exclusion(self, xy_points: Optional[np.ndarray], radius_m: float = 0.0) -> None:
        """Exclude circular regions from exposed-boundary accounting.

        Args:
            xy_points: World (x, y) meter coordinates, shape (N, 2). None or empty clears.
            radius_m: Radius in meters for each exclusion region.
        """
        self._boundary_exclusion_mask.fill(0)
        if xy_points is None:
            self._refresh_frontiers()
            self._update_exploration_degree()
            return
        arr = np.asarray(xy_points, dtype=np.float32)
        if arr.size == 0 or radius_m <= 0.0:
            self._refresh_frontiers()
            self._update_exploration_degree()
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, 2)
        radius_px = max(1, int(np.ceil(float(radius_m) * float(self.pixels_per_meter))))
        exclusion_mask = np.zeros_like(self._boundary_exclusion_mask, dtype=np.uint8)
        for px in self._xy_to_px(arr):
            cv2.circle(exclusion_mask, tuple(int(v) for v in px), radius_px, 1, thickness=-1)
        self._boundary_exclusion_mask[...] = exclusion_mask.astype(bool)
        self._refresh_frontiers()
        self._update_exploration_degree()

    def set_temporary_obstacles(self, xy_points: Optional[np.ndarray], radius_m: float = 0.0) -> None:
        """Overlay temporary obstacle regions without modifying the sensed obstacle map."""
        prev_explored = self.explored_area.copy()
        self._temporary_obstacle_mask.fill(0)
        if xy_points is not None:
            arr = np.asarray(xy_points, dtype=np.float32)
            if arr.size > 0 and radius_m > 0.0:
                if arr.ndim == 1:
                    arr = arr.reshape(1, 2)
                radius_px = max(1, int(np.ceil(float(radius_m) * float(self.pixels_per_meter))))
                temp_mask = np.zeros_like(self._temporary_obstacle_mask, dtype=np.uint8)
                for px in self._xy_to_px(arr):
                    cv2.circle(temp_mask, tuple(int(v) for v in px), radius_px, 1, thickness=-1)
                self._temporary_obstacle_mask[...] = temp_mask.astype(bool)
        self._refresh_real_obstacle_padding()
        self._prune_explored_area_from_obstacles()
        self._restore_explored_area_if_collapsed(prev_explored)
        self._refresh_frontiers()
        self._update_exploration_degree()


    def set_additional_vis_info(
        self,
        view_pos: np.ndarray = np.array([]),
        candidate_view_pos: np.ndarray = np.array([]),
        view_ray: np.ndarray = np.array([]),
    ):
        view_pos = np.asarray(view_pos)
        candidate_view_pos = np.asarray(candidate_view_pos)
        self.additional_vis_info["view_pos"] = (
            self._xy_to_px(view_pos.reshape(-1, 2)) if view_pos.size >= 2 else None
        )
        self.additional_vis_info["candidate_view_pos"] = (
            self._xy_to_px(candidate_view_pos.reshape(-1, 2)) if candidate_view_pos.size >= 2 else None
        )
        ray = np.asarray(view_ray)
        self.additional_vis_info["view_ray"] = (
            self._xy_to_px(ray.reshape(-1, 2)) if ray.size >= 4 else None
        )

    def _display_px_after_flip(self, px: np.ndarray) -> np.ndarray:
        pts = np.asarray(px, dtype=np.int32).copy()
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        pts[:, 1] = self._map.shape[0] - 1 - pts[:, 1]
        return pts

    def _render_base_map(self, include_last_frustum: bool = True) -> np.ndarray:
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        vis_img[self.explored_area] = (200, 255, 200)
        vis_img[~self._navigable_map] = self.radius_padding_color
        vis_img[self._real_dilated_map & self._navigable_map] = (130, 130, 130)
        if not self._ignore_real_obstacles_for_exploration:
            vis_img[self._real_map & ~self._map] = (60, 60, 60)
        vis_img[self._map] = (0, 0, 0)
        if np.any(self._temporary_obstacle_mask):
            vis_img[self._temporary_obstacle_mask] = (0, 140, 255)
        if np.any(self._pending_downstair_map):
            vis_img[self._pending_downstair_map] = (215, 205, 255)

        active_downstair_map = self._downstair_map & ~self._banned_downstair_map
        if np.any(active_downstair_map):
            vis_img[active_downstair_map] = (255, 0, 255)

        active_upstair_map = self._upstair_map & ~self._banned_upstair_map
        if np.any(active_upstair_map):
            vis_img[active_upstair_map] = (255, 255, 0)

        if include_last_frustum and np.any(self.last_update_frustum_mask):
            overlay_mask = self.last_update_frustum_mask.astype(bool, copy=False)
            alpha = 0.38
            base = vis_img[overlay_mask].astype(np.float32)
            overlay_color = np.array([30.0, 110.0, 255.0], dtype=np.float32)
            vis_img[overlay_mask] = np.clip(
                base * (1.0 - alpha) + overlay_color[None, :] * alpha,
                0,
                255,
            ).astype(np.uint8)

        return vis_img

    def visualize(self, labeled_frontiers_xy: Optional[np.ndarray] = None) -> np.ndarray:
        """Visualizes the map."""
        vis_img = self._render_base_map()
        # Draw frontiers in blue (BGR: 200, 0, 0)
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        # Highlight the currently selected frontier in red (BGR: 0, 0, 200)
        if self._selected_frontier_px is not None and len(self._selected_frontier_px) == 2:
            sf = tuple(int(i) for i in self._selected_frontier_px)
            cv2.circle(vis_img, sf, 7, (0, 0, 200), thickness=-1)

        # Show only the currently selected object's viewpoint and its ray.  The
        # previous overlay drew every object's candidate points, which looked
        # like a ring of blue navigation goals and did not describe the active
        # decision.
        view_ray = self.additional_vis_info.get("view_ray")
        if isinstance(view_ray, np.ndarray) and view_ray.shape[0] >= 2:
            start = tuple(int(i) for i in view_ray[0])
            end = tuple(int(i) for i in view_ray[-1])
            h, w = vis_img.shape[:2]
            start_in_bounds = 0 <= start[0] < w and 0 <= start[1] < h
            end_in_bounds = 0 <= end[0] < w and 0 <= end[1] < h
            if start_in_bounds or end_in_bounds:
                cv2.line(vis_img, start, end, (200, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        view_pos = self.additional_vis_info.get("view_pos")
        if isinstance(view_pos, np.ndarray) and view_pos.size >= 2:
            for vp in view_pos[:1]:
                sf = tuple(int(i) for i in vp)
                if 0 <= sf[0] < vis_img.shape[1] and 0 <= sf[1] < vis_img.shape[0]:
                    cv2.circle(vis_img, sf, 5, (200, 0, 0), thickness=-1)

        # Mark banned frontiers with an 'X' for clarity
        if isinstance(self._banned_frontiers_px, np.ndarray) and self._banned_frontiers_px.size >= 2:
            for bp in self._banned_frontiers_px:
                bx, by = int(bp[0]), int(bp[1])
                # Draw an 'X' centered at (bx, by)
                size = 7
                color = (0, 0, 0)  # black for high contrast
                thickness = 2
                cv2.line(vis_img, (bx - size, by - size), (bx + size, by + size), color, thickness)
                cv2.line(vis_img, (bx - size, by + size), (bx + size, by - size), color, thickness)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        if isinstance(labeled_frontiers_xy, np.ndarray) and labeled_frontiers_xy.size >= 2:
            labeled_px = self._xy_to_px(np.asarray(labeled_frontiers_xy, dtype=np.float32).reshape(-1, 2))
            mark_points_bgr(vis_img, self._display_px_after_flip(labeled_px))

        vis_img = reorient_rescale_map(vis_img)

        return vis_img

    def is_occupied_xy(self, xy: Union[np.ndarray, list, tuple], radius_m: float = -0.5) -> bool:
        """Return True if the given world (x, y) lies on the dilated real-obstacle mask."""
        arr = np.asarray(xy, dtype=np.float32).reshape(1, 2)
        px = self._xy_to_px(arr)[0]
        x, y = int(px[0]), int(px[1])
        h, w = self._real_map.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            # Treat out-of-bounds as occupied for safety.
            return True
        if radius_m <= 0:
            return bool(self._real_dilated_map[y, x])
        radius_p = int(np.ceil(float(radius_m) * self.pixels_per_meter))
        t, b, l, r = max(y - radius_p, 0), min(y + radius_p + 1, h), max(x - radius_p, 0), min(x + radius_p + 1, w)
        return bool(np.any(self._real_dilated_map[t:b, l:r]).item())

    @property
    def downstair_detected(self) -> bool:
        return self._downstair_detected

    @property
    def has_pending_downstair(self) -> bool:
        return bool(np.any(self._pending_downstair_map))

    @property
    def upstair_detected(self) -> bool:
        return self._upstair_detected

    def is_near_stairs(
        self, xy: Union[np.ndarray, list, tuple], radius_m: float = 0.2, include_upstairs: bool = True, dilate=True
    ) -> bool:
        """Return True if (x, y) is within radius_m of detected stair regions (down or up).

        Downstairs regions come from geometric drop-off detection; upstairs regions from semantics.
        Both masks are optionally dilated to catch small gaps/holes.
        """
        arr = np.asarray(xy, dtype=np.float32).reshape(1, 2)
        px = self._xy_to_px(arr)[0]
        x, y = int(px[0]), int(px[1])
        h, w = self._map.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            # Stay conservative for out-of-bounds goals
            return True

        stair_mask = np.zeros_like(self._map, dtype=bool)
        stair_mask |= self._downstair_map & ~self._banned_downstair_map
        if include_upstairs:
            stair_mask |= self._upstair_map & ~self._banned_upstair_map
        if not np.any(stair_mask):
            return False

        if dilate:
            radius_px = max(1, int(round(radius_m * float(self.pixels_per_meter))))
            kernel_size = radius_px * 2 + 1  # make it odd
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated = cv2.dilate(stair_mask.astype(np.uint8), kernel, iterations=1)
            return bool(dilated[y, x])
        else:
            return bool(stair_mask[y, x])

    def mark_semantic_obstacles(self, stairs_cloud: np.ndarray, kind: str = "") -> None:
        """Project semantic stair points onto the map.

        Upstairs detections are tracked as stair candidates only; they should not
        become hard obstacles because the geometry map already captures elevated
        stair surfaces and semantic masks can over-segment surrounding floor.
        """
        if stairs_cloud is None or len(stairs_cloud) == 0:
            return
        height = stairs_cloud[:, -1]
        max_height = np.max(height)
        if max_height < 0.6 or len(stairs_cloud) < 50:
            return
        stairs_cloud = stairs_cloud[:, :2]
        pts = np.asarray(stairs_cloud, dtype=np.float32)
        if pts.size == 0:
            return
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        pixel_points = self._xy_to_px(pts)
        h, w = self._map.shape
        valid = (
            (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < w) & (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < h)
        )
        if not np.any(valid):
            return
        pp = pixel_points[valid]
        height = height[valid].astype(np.float32, copy=False)
        self._update_stair_height_map(self._upstair_height_map, pp, height)
        is_upstairs = kind == "upstairs"
        if is_upstairs and hasattr(self, "_upstair_map"):
            self._upstair_map[pp[:, 1], pp[:, 0]] = 1
        self._refresh_stair_summary("up")
        if not is_upstairs:
            prev_explored = self.explored_area.copy()
            self._map[pp[:, 1], pp[:, 0]] = 1
            self._real_map[pp[:, 1], pp[:, 0]] = 1
            # Update navigable map to respect new obstacles
            self._navigable_map = self._compute_navigable_mask(self._get_effective_obstacle_mask())
            self._refresh_real_obstacle_padding()
            self._prune_explored_area_from_obstacles()
            self._restore_explored_area_if_collapsed(prev_explored)
        self._refresh_frontiers()
        self._update_exploration_degree()

    def set_selected_frontier(self, xy: Union[np.ndarray, None]) -> None:
        """Set the currently selected frontier for visualization.

        Args:
            xy: Frontier location in world (x, y) meters; None clears selection.
        """
        if xy is None:
            self._selected_frontier_px = None
            return
        arr = np.asarray(xy, dtype=np.float32).reshape(1, 2)
        px = self._xy_to_px(arr)[0]
        self._selected_frontier_px = px

    def set_banned_frontiers(self, xy_points: Optional[np.ndarray]) -> None:
        """Set banned frontiers for visualization (drawn as 'X' marks).

        Args:
            xy_points: Array of world (x, y) meters, shape (N, 2). None or empty clears.
        """
        if xy_points is None:
            self._banned_frontiers_px = np.array([])
            return
        arr = np.asarray(xy_points, dtype=np.float32)
        if arr.size == 0:
            self._banned_frontiers_px = np.array([])
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, 2)
        self._banned_frontiers_px = self._xy_to_px(arr)

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """Converts an array of (x, y) coordinates to pixel coordinates.

        Args:
            points: The array of (x, y) coordinates to convert.

        Returns:
            The array of (x, y) pixel coordinates.
        """
        px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        px[:, 0] = self._map.shape[0] - px[:, 0]
        return px.astype(int)

    def _px_to_xy(self, px: np.ndarray) -> np.ndarray:
        """Converts an array of pixel coordinates to (x, y) coordinates.

        Args:
            px: The array of pixel coordinates to convert.

        Returns:
            The array of (x, y) coordinates.
        """
        px_copy = px.copy()
        px_copy[:, 0] = self._map.shape[0] - px_copy[:, 0]
        points = (px_copy - self._episode_pixel_origin) / self.pixels_per_meter
        return points[:, ::-1]
