from typing import List, Optional

import cv2
import numpy as np

from .traj_visualizer import TrajectoryVisualizer
from ..utils.img_utils import reorient_rescale_map
from ..utils.viz import mark_points_bgr


class ColorTopDownMap:
    """Persistent colored BEV built by splatting RGB onto projected 3D points."""

    _trail_full_opacity_steps: int = 10
    _trail_min_opacity: float = 0.15
    _trail_path_thickness: int = 2
    _trail_dwell_min_steps: int = 2
    _trail_dwell_base_radius: int = 3
    _trail_dwell_radius_step: int = 1
    _trail_dwell_max_radius: int = 10

    def __init__(self, size: int, pixels_per_meter: int, episode_pixel_origin: np.ndarray) -> None:
        self.size = int(size)
        self.pixels_per_meter = int(pixels_per_meter)
        self._episode_pixel_origin = np.asarray(episode_pixel_origin, dtype=np.int32).copy()
        self._traj_vis = self._create_traj_visualizer()
        self.reset()

    def reset(self) -> None:
        self._color = np.full((self.size, self.size, 3), 255, dtype=np.uint8)
        self._valid = np.zeros((self.size, self.size), dtype=bool)
        self._surface_height = np.full((self.size, self.size), -np.inf, dtype=np.float32)
        self._camera_positions: List[np.ndarray] = []
        self._last_camera_yaw: float = 0.0
        self._step: int = 0
        self._traj_vis = self._create_traj_visualizer()

    def update_agent_traj(self, robot_xy: np.ndarray, robot_heading: float) -> None:
        self._camera_positions.append(np.asarray(robot_xy, dtype=np.float32).copy())
        self._last_camera_yaw = float(robot_heading)

    def update_map(
        self,
        rgb: np.ndarray,
        dense_points_episodic: Optional[np.ndarray],
        invalid_mask: Optional[np.ndarray] = None,
        max_height_m: Optional[float] = None,
    ) -> None:
        if rgb is None or dense_points_episodic is None:
            return
        if hasattr(dense_points_episodic, "get"):
            dense_points_episodic = dense_points_episodic.get()
        else:
            dense_points_episodic = np.asarray(dense_points_episodic)
        if dense_points_episodic.ndim != 3 or dense_points_episodic.shape[-1] != 3:
            return

        self._step += 1

        if invalid_mask is None:
            valid = np.ones(dense_points_episodic.shape[:2], dtype=bool)
        else:
            if hasattr(invalid_mask, "get"):
                invalid_mask = invalid_mask.get()
            valid = ~np.asarray(invalid_mask, dtype=bool)
        valid &= np.isfinite(dense_points_episodic).all(axis=-1)
        if max_height_m is not None:
            valid &= dense_points_episodic[..., 2] <= float(max_height_m)

        if not np.any(valid):
            return

        pts = dense_points_episodic[valid]
        # Habitat observations are RGB, while OpenCV visualization expects BGR.
        colors = rgb[valid][:, ::-1]
        px = self._xy_to_px(pts[:, :2])

        in_bounds = (
            (px[:, 0] >= 0)
            & (px[:, 0] < self.size)
            & (px[:, 1] >= 0)
            & (px[:, 1] < self.size)
        )
        if not np.any(in_bounds):
            return

        pts = pts[in_bounds]
        colors = colors[in_bounds]
        px = px[in_bounds]

        flat_idx = px[:, 1] * self.size + px[:, 0]
        heights = pts[:, 2].astype(np.float32, copy=False)

        # Keep the highest point per map cell within the current frame.
        order = np.lexsort((heights, flat_idx))
        flat_idx = flat_idx[order]
        heights = heights[order]
        colors = colors[order]
        keep = np.ones(flat_idx.shape[0], dtype=bool)
        keep[:-1] = flat_idx[:-1] != flat_idx[1:]
        flat_idx = flat_idx[keep]
        heights = heights[keep]
        colors = colors[keep]

        curr_height = self._surface_height.reshape(-1)
        update_mask = (~self._valid.reshape(-1)[flat_idx]) | (heights >= (curr_height[flat_idx] - 0.05))
        if not np.any(update_mask):
            return

        flat_idx = flat_idx[update_mask]
        heights = heights[update_mask]
        colors = colors[update_mask]

        color_flat = self._color.reshape(-1, 3)
        valid_flat = self._valid.reshape(-1)
        curr_height[flat_idx] = heights
        color_flat[flat_idx] = colors
        valid_flat[flat_idx] = True

    def _display_px_after_flip(self, px: np.ndarray) -> np.ndarray:
        pts = np.asarray(px, dtype=np.int32).copy()
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        pts[:, 1] = self.size - 1 - pts[:, 1]
        return pts

    def visualize(
        self,
        frontiers_xy: Optional[np.ndarray] = None,
        selected_frontier_xy: Optional[np.ndarray] = None,
        banned_xy: Optional[np.ndarray] = None,
        temporary_obstacle_xy: Optional[np.ndarray] = None,
        temporary_obstacle_radius_m: Optional[float] = None,
        labeled_frontiers_xy: Optional[np.ndarray] = None,
        deferred_xy: Optional[np.ndarray] = None,
        deferred_radius_m: Optional[float] = None,
        review_xy: Optional[np.ndarray] = None,
        review_radius_m: Optional[float] = None,
    ) -> np.ndarray:
        vis_img = np.full_like(self._color, 255)
        vis_img[self._valid] = self._color[self._valid]
        return self._render_map(
            vis_img,
            frontiers_xy=frontiers_xy,
            selected_frontier_xy=selected_frontier_xy,
            banned_xy=banned_xy,
            temporary_obstacle_xy=temporary_obstacle_xy,
            temporary_obstacle_radius_m=temporary_obstacle_radius_m,
            labeled_frontiers_xy=labeled_frontiers_xy,
            deferred_xy=deferred_xy,
            deferred_radius_m=deferred_radius_m,
            review_xy=review_xy,
            review_radius_m=review_radius_m,
            include_navigation_annotations=True,
            include_trajectory=True,
        )

    def _render_map(
        self,
        vis_img: np.ndarray,
        *,
        frontiers_xy: Optional[np.ndarray] = None,
        selected_frontier_xy: Optional[np.ndarray] = None,
        banned_xy: Optional[np.ndarray] = None,
        temporary_obstacle_xy: Optional[np.ndarray] = None,
        temporary_obstacle_radius_m: Optional[float] = None,
        labeled_frontiers_xy: Optional[np.ndarray] = None,
        deferred_xy: Optional[np.ndarray] = None,
        deferred_radius_m: Optional[float] = None,
        review_xy: Optional[np.ndarray] = None,
        review_radius_m: Optional[float] = None,
        include_navigation_annotations: bool = True,
        include_trajectory: bool = True,
    ) -> np.ndarray:

        if include_navigation_annotations:
            if isinstance(frontiers_xy, np.ndarray) and frontiers_xy.size >= 2:
                for frontier in self._xy_to_px(frontiers_xy):
                    if self._in_bounds(frontier):
                        cv2.circle(vis_img, tuple(int(i) for i in frontier), 5, (200, 0, 0), 2)

            if selected_frontier_xy is not None:
                sf = self._xy_to_px(np.asarray(selected_frontier_xy, dtype=np.float32).reshape(1, 2))[0]
                if self._in_bounds(sf):
                    cv2.circle(vis_img, tuple(int(i) for i in sf), 7, (0, 0, 200), thickness=-1)

            if isinstance(banned_xy, np.ndarray) and banned_xy.size >= 2:
                for bp in self._xy_to_px(banned_xy):
                    if not self._in_bounds(bp):
                        continue
                    bx, by = int(bp[0]), int(bp[1])
                    cv2.line(vis_img, (bx - 7, by - 7), (bx + 7, by + 7), (0, 0, 0), 2)
                    cv2.line(vis_img, (bx - 7, by + 7), (bx + 7, by - 7), (0, 0, 0), 2)

            if isinstance(temporary_obstacle_xy, np.ndarray) and temporary_obstacle_xy.size >= 2:
                radius_px = max(3, int(round(float(temporary_obstacle_radius_m or 0.0) * self.pixels_per_meter)))
                for tp in self._xy_to_px(temporary_obstacle_xy):
                    if not self._in_bounds(tp):
                        continue
                    center = tuple(int(i) for i in tp)
                    cv2.circle(vis_img, center, radius_px, (0, 140, 255), thickness=2)
                    cv2.circle(vis_img, center, 2, (0, 140, 255), thickness=-1)

            if isinstance(deferred_xy, np.ndarray) and deferred_xy.size >= 2:
                radius_px = max(3, int(round(float(deferred_radius_m or 0.0) * self.pixels_per_meter)))
                for dp in self._xy_to_px(deferred_xy):
                    if self._in_bounds(dp):
                        cv2.circle(vis_img, tuple(int(i) for i in dp), radius_px, (0, 165, 255), 2)

            if review_xy is not None:
                rp = self._xy_to_px(np.asarray(review_xy, dtype=np.float32).reshape(1, 2))[0]
                if self._in_bounds(rp):
                    radius_px = max(3, int(round(float(review_radius_m or 0.0) * self.pixels_per_meter)))
                    cv2.circle(vis_img, tuple(int(i) for i in rp), radius_px, (0, 215, 255), 3)

        vis_img = cv2.flip(vis_img, 0)
        if include_trajectory and len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(vis_img, self._camera_positions, self._last_camera_yaw)
        if include_navigation_annotations and isinstance(labeled_frontiers_xy, np.ndarray) and labeled_frontiers_xy.size >= 2:
            labeled_px = self._xy_to_px(np.asarray(labeled_frontiers_xy, dtype=np.float32).reshape(-1, 2))
            mark_points_bgr(vis_img, self._display_px_after_flip(labeled_px))
        return reorient_rescale_map(vis_img)

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, 2)
        px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        px[:, 0] = self.size - px[:, 0]
        return px.astype(np.int32)

    def _in_bounds(self, point_px: np.ndarray) -> bool:
        return bool(0 <= point_px[0] < self.size and 0 <= point_px[1] < self.size)

    def _create_traj_visualizer(self) -> TrajectoryVisualizer:
        return TrajectoryVisualizer(
            self._episode_pixel_origin,
            self.pixels_per_meter,
            use_opacity_decay=True,
            full_opacity_steps=self._trail_full_opacity_steps,
            min_path_opacity=self._trail_min_opacity,
            path_thickness=self._trail_path_thickness,
            highlight_dwell=True,
            dwell_min_steps=self._trail_dwell_min_steps,
            dwell_base_radius=self._trail_dwell_base_radius,
            dwell_radius_step=self._trail_dwell_radius_step,
            dwell_max_radius=self._trail_dwell_max_radius,
        )
