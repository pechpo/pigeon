from typing import Any, List, Tuple, Union

import cv2
import numpy as np


class TrajectoryVisualizer:
    _num_drawn_points: int = 1
    _cached_path_mask: Union[np.ndarray, None] = None
    _origin_in_img: Union[np.ndarray, None] = None
    _pixels_per_meter: Union[float, None] = None
    agent_line_length: int = 10
    agent_line_thickness: int = 3
    path_color: tuple = (0, 255, 0)
    path_thickness: int = 3
    scale_factor: float = 1.0

    def __init__(
        self,
        origin_in_img: np.ndarray,
        pixels_per_meter: float,
        use_opacity_decay: bool = False,
        full_opacity_steps: int = 10,
        min_path_opacity: float = 0.15,
        path_thickness: int = 3,
        highlight_dwell: bool = False,
        dwell_min_steps: int = 2,
        dwell_base_radius: int = 2,
        dwell_radius_step: int = 1,
        dwell_max_radius: int = 10,
    ):
        self._origin_in_img = origin_in_img
        self._pixels_per_meter = pixels_per_meter
        self._use_opacity_decay = use_opacity_decay
        self._full_opacity_steps = max(1, int(full_opacity_steps))
        self._min_path_opacity = float(np.clip(min_path_opacity, 0.0, 1.0))
        self.path_thickness = int(path_thickness)
        self._highlight_dwell = highlight_dwell
        self._dwell_min_steps = max(2, int(dwell_min_steps))
        self._dwell_base_radius = max(1, int(dwell_base_radius))
        self._dwell_radius_step = max(1, int(dwell_radius_step))
        self._dwell_max_radius = max(self._dwell_base_radius, int(dwell_max_radius))
        self.reset()

    def reset(self) -> None:
        self._num_drawn_points = 1
        self._cached_path_mask = None

    def draw_trajectory(
        self,
        img: np.ndarray,
        camera_positions: Union[np.ndarray, List[np.ndarray]],
        camera_yaw: float,
    ) -> np.ndarray:
        """Draws the trajectory on the image and returns it"""
        img = self._draw_path(img, camera_positions)
        img = self._draw_agent(img, camera_positions[-1], camera_yaw)
        return img

    def _draw_path(self, img: np.ndarray, camera_positions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Draws the path on the image and returns it"""
        if len(camera_positions) >= 2:
            if self._use_opacity_decay:
                img = self._draw_path_with_opacity_decay(img, camera_positions)
            else:
                if self._cached_path_mask is not None:
                    path_mask = self._cached_path_mask.copy()
                else:
                    path_mask = np.zeros(img.shape[:2], dtype=np.uint8)

                for i in range(self._num_drawn_points - 1, len(camera_positions) - 1):
                    path_mask = self._draw_line(path_mask, camera_positions[i], camera_positions[i + 1])

                img[path_mask == 255] = self.path_color

                self._cached_path_mask = path_mask
                self._num_drawn_points = len(camera_positions)

        if self._highlight_dwell:
            self._draw_dwell_points(img, camera_positions)
        return img

    def _draw_path_with_opacity_decay(
        self, img: np.ndarray, camera_positions: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """Draw the full path each frame so older segments can fade by age."""
        num_segments = len(camera_positions) - 1
        for segment_idx in range(num_segments):
            alpha = self._get_segment_opacity(segment_idx, num_segments)
            if alpha <= 0.0:
                continue
            if alpha >= 1.0:
                img = self._draw_line(
                    img,
                    camera_positions[segment_idx],
                    camera_positions[segment_idx + 1],
                    self.path_color,
                )
                continue

            path_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            path_mask = self._draw_line(path_mask, camera_positions[segment_idx], camera_positions[segment_idx + 1])
            self._blend_color_mask(img, path_mask, alpha)

        self._num_drawn_points = len(camera_positions)
        self._cached_path_mask = None
        return img

    def _draw_line(
        self,
        img: np.ndarray,
        pt_a: np.ndarray,
        pt_b: np.ndarray,
        color: Union[int, Tuple[int, int, int]] = 255,
    ) -> np.ndarray:
        """Draws a line between two points and returns it"""
        # Convert metric coordinates to pixel coordinates
        px_a = self._metric_to_pixel(pt_a)
        px_b = self._metric_to_pixel(pt_b)

        if np.array_equal(px_a, px_b):
            return img

        cv2.line(
            img,
            tuple(px_a[::-1]),
            tuple(px_b[::-1]),
            color,
            int(self.path_thickness * self.scale_factor),
        )

        return img

    def _blend_color_mask(self, img: np.ndarray, color_mask: np.ndarray, alpha: float) -> None:
        """Blend the path color onto existing pixels without flattening the BEV colors."""
        mask = color_mask == 255
        if not np.any(mask):
            return
        base_pixels = img[mask].astype(np.float32)
        path_color = np.asarray(self.path_color, dtype=np.float32)
        blended = np.rint(base_pixels * (1.0 - alpha) + path_color * alpha).astype(np.uint8)
        img[mask] = blended

    def _get_segment_opacity(self, segment_idx: int, num_segments: int) -> float:
        """Keep the most recent steps fully opaque and fade the older history."""
        return self._get_history_opacity(segment_idx, num_segments)

    def _get_history_opacity(self, item_idx: int, num_items: int) -> float:
        steps_from_end = num_items - item_idx
        if steps_from_end <= self._full_opacity_steps:
            return 1.0

        older_item_count = num_items - self._full_opacity_steps
        relative_rank = item_idx + 1
        opacity = self._min_path_opacity + (1.0 - self._min_path_opacity) * (
            relative_rank / (older_item_count + 1)
        )
        return float(np.clip(opacity, self._min_path_opacity, 0.999))

    def _draw_dwell_points(self, img: np.ndarray, camera_positions: Union[np.ndarray, List[np.ndarray]]) -> None:
        if len(camera_positions) < self._dwell_min_steps:
            return

        px_positions = [self._metric_to_pixel(pos) for pos in camera_positions]
        run_start = 0
        while run_start < len(px_positions):
            run_end = run_start + 1
            while run_end < len(px_positions) and np.array_equal(px_positions[run_end], px_positions[run_start]):
                run_end += 1

            dwell_steps = run_end - run_start
            if dwell_steps >= self._dwell_min_steps:
                radius = min(
                    self._dwell_base_radius + (dwell_steps - self._dwell_min_steps) * self._dwell_radius_step,
                    self._dwell_max_radius,
                )
                alpha = self._get_history_opacity(run_end - 1, len(px_positions)) if self._use_opacity_decay else 1.0
                self._draw_dwell_circle(img, px_positions[run_start], radius, alpha)

            run_start = run_end

    def _draw_dwell_circle(self, img: np.ndarray, center_px: np.ndarray, radius: int, alpha: float) -> None:
        if alpha <= 0.0:
            return
        if not self._point_in_bounds(center_px, img.shape[:2]):
            return
        center = tuple(center_px[::-1])
        if alpha >= 1.0:
            cv2.circle(img, center, radius, self.path_color, -1)
            return

        circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask, center, radius, 255, -1)
        self._blend_color_mask(img, circle_mask, alpha)

    def _point_in_bounds(self, point_px: np.ndarray, img_shape: Tuple[int, int]) -> bool:
        return bool(0 <= point_px[0] < img_shape[0] and 0 <= point_px[1] < img_shape[1])

    def _draw_agent(self, img: np.ndarray, camera_position: np.ndarray, camera_yaw: float) -> np.ndarray:
        """Draws the agent on the image and returns it"""
        px_position = self._metric_to_pixel(camera_position)
        cv2.circle(
            img,
            tuple(px_position[::-1]),
            int(8 * self.scale_factor),
            (255, 192, 15),
            -1,
        )
        heading_end_pt = (
            int(px_position[0] - self.agent_line_length * self.scale_factor * np.cos(camera_yaw)),
            int(px_position[1] - self.agent_line_length * self.scale_factor * np.sin(camera_yaw)),
        )
        cv2.line(
            img,
            tuple(px_position[::-1]),
            tuple(heading_end_pt[::-1]),
            (0, 0, 0),
            int(self.agent_line_thickness * self.scale_factor),
        )

        return img

    def draw_circle(self, img: np.ndarray, position: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Draws the point as a circle on the image and returns it"""
        px_position = self._metric_to_pixel(position)
        cv2.circle(img, tuple(px_position[::-1]), **kwargs)

        return img

    def _metric_to_pixel(self, pt: np.ndarray) -> np.ndarray:
        """Converts a metric coordinate to a pixel coordinate"""
        # Need to flip y-axis because pixel coordinates start from top left
        px = pt * self._pixels_per_meter * np.array([-1, -1]) + self._origin_in_img
        px = px.astype(np.int32)
        return px
