from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .viz import compose_frame, overlay_status_bar

_UNSET = object()


def _screen_size() -> Optional[tuple[int, int]]:
    try:
        import tkinter as tk  # type: ignore

        root = tk.Tk()
        root.withdraw()
        width, height = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(width), int(height)
    except Exception:
        return None


class VizManager:
    def __init__(self, window_name: str, enable_display: bool = False) -> None:
        self.window_name = window_name
        self.enable_display = enable_display
        self._window_sized = False
        self._window_created = False
        self._rendered_frames: list[np.ndarray] = []
        self._state: Dict[str, Any] = {
            "rgb": None,
            "info": {},
            "depth": None,
            "planner_viz": None,
            "det_seg": None,
            "value_map_viz": None,
            "aux_map_viz": None,
            "height_map_viz": None,
            "right_panel_viz": None,
            "bottom_text": None,
            "steps": 0,
            "max_steps": 0,
            "ep_text": None,
        }

    def update_observation(
        self,
        *,
        rgb: Optional[np.ndarray],
        info: Dict[str, Any],
        depth: Optional[np.ndarray],
        steps: int,
        max_steps: int,
        ep_text: Optional[str],
    ) -> None:
        self._state["rgb"] = rgb
        self._state["info"] = info
        self._state["depth"] = depth
        # Detection overlays are frame-specific. Clear the previous frame's
        # annotations before any mid-step render occurs on the new observation.
        self._state["det_seg"] = None
        self._state["steps"] = steps
        self._state["max_steps"] = max_steps
        self._state["ep_text"] = ep_text
        self._rendered_frames.clear()

    def update_panels(
        self,
        *,
        planner_viz: Any = _UNSET,
        det_seg: Any = _UNSET,
        value_map_viz: Any = _UNSET,
        aux_map_viz: Any = _UNSET,
        height_map_viz: Any = _UNSET,
        right_panel_viz: Any = _UNSET,
        bottom_text: Any = _UNSET,
        render: bool = False,
    ) -> Optional[np.ndarray]:
        if planner_viz is not _UNSET:
            self._state["planner_viz"] = planner_viz
        if det_seg is not _UNSET:
            self._state["det_seg"] = det_seg
        if value_map_viz is not _UNSET:
            self._state["value_map_viz"] = value_map_viz
        if aux_map_viz is not _UNSET:
            self._state["aux_map_viz"] = aux_map_viz
        if height_map_viz is not _UNSET:
            self._state["height_map_viz"] = height_map_viz
        if right_panel_viz is not _UNSET:
            self._state["right_panel_viz"] = right_panel_viz
        if bottom_text is not _UNSET:
            self._state["bottom_text"] = bottom_text
        frame = self.render() if render else None
        if frame is not None:
            self._rendered_frames.append(frame.copy())
        return frame

    def drain_rendered_frames(self) -> list[np.ndarray]:
        frames = self._rendered_frames
        self._rendered_frames = []
        return frames

    def render(self) -> Optional[np.ndarray]:
        frame = compose_frame(
            self._state["rgb"],
            self._state["info"],
            planner_viz=self._state["planner_viz"],
            depth=self._state["depth"],
            det_seg=self._state["det_seg"],
            value_map_viz=self._state["value_map_viz"],
            aux_map_viz=self._state["aux_map_viz"],
            height_map_viz=self._state["height_map_viz"],
            right_panel_viz=self._state["right_panel_viz"],
            bottom_text=self._state["bottom_text"],
        )
        if frame is None:
            return None

        annotated = overlay_status_bar(
            frame,
            int(self._state["steps"]),
            int(self._state["max_steps"]),
            extra_text=self._state["ep_text"],
        )
        if self.enable_display:
            self._show(annotated)
        return annotated

    def _show(self, frame_bgr: np.ndarray) -> None:
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            if not self._window_sized:
                screen = _screen_size()
                if screen is not None:
                    _, screen_h = screen
                    target_h = max(200, screen_h // 2)
                    height, width = frame_bgr.shape[:2]
                    target_w = max(200, int(round(width * (target_h / max(1, height)))))
                    try:
                        cv2.resizeWindow(self.window_name, target_w, target_h)
                    except Exception:
                        pass
                self._window_sized = True
            cv2.imshow(self.window_name, frame_bgr)
            cv2.waitKey(1)
            self._window_created = True
        except Exception:
            pass

    def close(self) -> None:
        if not self._window_created:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
