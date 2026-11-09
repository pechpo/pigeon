"""Compact per-episode JSONL traces for offline failure analysis."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _jsonable(value: Any) -> Any:
    """Convert common numpy/CuPy/scalar values without serializing large arrays."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "get"):
        try:
            return _jsonable(value.get())
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class EpisodeTraceLogger:
    """Append-only JSONL logger; one file is maintained for each episode."""

    def __init__(self, trace_dir: Optional[str], episode_id: str) -> None:
        self.path: Optional[Path] = None
        self._file = None
        if not trace_dir:
            return
        self.path = Path(trace_dir) / f"episode_{episode_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def write(self, event: str, **payload: Any) -> None:
        if self._file is None:
            return
        record: Dict[str, Any] = {"event": event}
        record.update(payload)
        self._file.write(
            json.dumps(_jsonable(record), ensure_ascii=True, separators=(",", ":"))
            + "\n"
        )
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
