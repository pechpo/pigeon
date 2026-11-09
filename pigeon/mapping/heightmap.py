from typing import List, Optional

import matplotlib.cm as cm
import numpy as np

from .color_topdown_map import ColorTopDownMap


def _build_coolwarm_lut_bgr() -> np.ndarray:
    colors = (cm.get_cmap("coolwarm")(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)
    return colors[:, ::-1]


_COOLWARM_LUT_BGR = _build_coolwarm_lut_bgr()
_EXTRA_COLD_LOW_BGR = np.array([255, 0, 160], dtype=np.uint8)


def _colorize_heights_bgr(values: np.ndarray, clip_height_m: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    clip_height_m = max(float(clip_height_m), 1e-6)
    extra_cold_min_m = -3.0 * clip_height_m
    clipped = np.clip(values, extra_cold_min_m, clip_height_m)

    colors = np.empty((*clipped.shape, 3), dtype=np.uint8)
    standard_mask = clipped >= -clip_height_m
    if np.any(standard_mask):
        standard_values = clipped[standard_mask]
        idx = np.rint((standard_values + clip_height_m) * (255.0 / (2.0 * clip_height_m))).astype(np.uint8)
        colors[standard_mask] = _COOLWARM_LUT_BGR[idx]

    extra_cold_mask = ~standard_mask
    if np.any(extra_cold_mask):
        extra_cold_values = clipped[extra_cold_mask]
        alpha = (extra_cold_values - extra_cold_min_m) / (2.0 * clip_height_m)
        alpha = np.clip(alpha, 0.0, 1.0).reshape(-1, 1)
        cold_start = _EXTRA_COLD_LOW_BGR.astype(np.float32)
        cold_end = _COOLWARM_LUT_BGR[0].astype(np.float32)
        colors[extra_cold_mask] = np.rint(cold_start + alpha * (cold_end - cold_start)).astype(np.uint8)

    return colors


class HeightMap(ColorTopDownMap):
    def reset(self) -> None:
        self._valid = np.zeros((self.size, self.size), dtype=bool)
        self._height = np.full((self.size, self.size), -np.inf, dtype=np.float32)
        self._last_update_mask = np.zeros((self.size, self.size), dtype=bool)
        self._camera_positions: List[np.ndarray] = []
        self._last_camera_yaw: float = 0.0
        self._step: int = 0
        self._traj_vis = self._create_traj_visualizer()

    def update_map(
        self,
        dense_points_episodic: Optional[np.ndarray],
        invalid_mask: Optional[np.ndarray] = None,
        max_height_m: Optional[float] = None,
        height_offset_m: float = 0.0,
    ) -> None:
        self._last_update_mask.fill(False)
        if dense_points_episodic is None:
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
        px = px[in_bounds]

        flat_idx = px[:, 1] * self.size + px[:, 0]
        heights = pts[:, 2].astype(np.float32, copy=False)
        if height_offset_m != 0.0:
            heights = heights + np.float32(height_offset_m)

        order = np.lexsort((heights, flat_idx))
        flat_idx = flat_idx[order]
        heights = heights[order]
        keep = np.ones(flat_idx.shape[0], dtype=bool)
        keep[:-1] = flat_idx[:-1] != flat_idx[1:]
        flat_idx = flat_idx[keep]
        heights = heights[keep]

        curr_height = self._height.reshape(-1)
        update_mask = (~self._valid.reshape(-1)[flat_idx]) | (heights >= (curr_height[flat_idx] - 0.05))
        if not np.any(update_mask):
            return

        flat_idx = flat_idx[update_mask]
        heights = heights[update_mask]

        curr_height[flat_idx] = heights
        self._valid.reshape(-1)[flat_idx] = True
        self._last_update_mask.reshape(-1)[flat_idx] = True

    def get_local_median_height(self, center_xy: Optional[np.ndarray], radius_m: float = 0.75) -> Optional[float]:
        if center_xy is None:
            return None
        center = np.asarray(center_xy, dtype=np.float32).reshape(-1)
        if center.size != 2:
            return None
        center_px = self._xy_to_px(center.reshape(1, 2))[0]
        radius_px = max(1, int(round(float(radius_m) * float(self.pixels_per_meter))))
        x0 = max(0, int(center_px[0]) - radius_px)
        x1 = min(self.size, int(center_px[0]) + radius_px + 1)
        y0 = max(0, int(center_px[1]) - radius_px)
        y1 = min(self.size, int(center_px[1]) + radius_px + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        valid = self._valid[y0:y1, x0:x1] & np.isfinite(self._height[y0:y1, x0:x1])
        if not np.any(valid):
            return None
        return float(np.median(self._height[y0:y1, x0:x1][valid]))

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
        height_overlay_mask: Optional[np.ndarray] = None,
        height_overlay_values: Optional[np.ndarray] = None,
        clip_height_m: float = 1.5,
    ) -> np.ndarray:
        vis_img = np.full((self.size, self.size, 3), 255, dtype=np.uint8)
        if np.any(self._valid):
            vis_img[self._valid] = _colorize_heights_bgr(self._height[self._valid], clip_height_m)
        self._apply_height_overlay(vis_img, clip_height_m, height_overlay_mask, height_overlay_values)
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
            include_navigation_annotations=False,
            include_trajectory=False,
        )

    def _apply_height_overlay(
        self,
        vis_img: np.ndarray,
        clip_height_m: float,
        overlay_mask: Optional[np.ndarray],
        overlay_values: Optional[np.ndarray],
    ) -> None:
        if overlay_mask is None or overlay_values is None:
            return
        mask = np.asarray(overlay_mask, dtype=bool)
        values = np.asarray(overlay_values, dtype=np.float32)
        if mask.shape != (self.size, self.size) or values.shape != (self.size, self.size):
            return
        valid = mask & np.isfinite(values)
        if not np.any(valid):
            return
        vis_img[valid] = _colorize_heights_bgr(values[valid], clip_height_m)

    def get_valid_height_points(self, mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        valid = self._valid & np.isfinite(self._height)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != valid.shape:
                return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)
            valid &= mask
        if not np.any(valid):
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)
        rows, cols = np.where(valid)
        px = np.stack([cols, rows], axis=1).astype(np.int32)
        return px, self._height[rows, cols].astype(np.float32, copy=False)

    def get_height_overlay(self, mask: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        overlay_mask = np.zeros((self.size, self.size), dtype=bool)
        overlay_height = np.full((self.size, self.size), np.nan, dtype=np.float32)
        if mask is None:
            return overlay_mask, overlay_height
        px, heights = self.get_valid_height_points(mask)
        if px.size == 0:
            return overlay_mask, overlay_height
        overlay_mask[px[:, 1], px[:, 0]] = True
        overlay_height[px[:, 1], px[:, 0]] = heights
        return overlay_mask, overlay_height
