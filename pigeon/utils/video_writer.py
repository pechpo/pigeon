from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class EpisodeVideoWriter:
    def __init__(self, path: Path, fps: int = 10) -> None:
        self.path = path
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._size = None

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            self._size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self._size)
        else:
            # OpenCV VideoWriter requires all frames be the same size as
            # the size used at initialization. If a later frame differs,
            # resize it to the writer's size to keep the stream valid.
            h, w = frame_bgr.shape[:2]
            if (w, h) != self._size:
                frame_bgr = cv2.resize(frame_bgr, self._size, interpolation=cv2.INTER_AREA)
        # Write frame and release references ASAP to avoid accumulation
        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


@dataclass
class VideoWriterFactory:
    out_dir: str
    enable_debug: bool = False

    def create(self, episode_id: str) -> Optional[EpisodeVideoWriter]:
        if not self.enable_debug:
            return None
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        return EpisodeVideoWriter(Path(self.out_dir) / f"{episode_id}.mp4")
