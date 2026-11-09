from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from habitat import registry
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Simulator
from habitat.config.default_structured_configs import MeasurementConfig
from omegaconf import DictConfig


@registry.register_measure
class TraveledDistance(Measure):
    """Accumulates agent's traveled distance within an episode.

    Metric key: "traveled_distance" (float, meters)
    By default measures planar distance on the ground plane (x, z).
    Set config "include_vertical=True" to accumulate full 3D distance.
    """

    cls_uuid: str = "traveled_distance"

    def __init__(self, sim: Simulator, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        self._include_vertical: bool = bool(getattr(config, "include_vertical", False))
        self._prev_pos: Optional[np.ndarray] = None
        self._total: float = 0.0
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return TraveledDistance.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._prev_pos = None
        self._total = 0.0
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        pos = np.asarray(self._sim.get_agent_state().position, dtype=np.float32)
        if not self._include_vertical:
            # Only accumulate on ground plane (x, z)
            pos = np.array([pos[0], pos[2]], dtype=np.float32)

        if self._prev_pos is None:
            self._prev_pos = pos
            self._metric = float(self._total)
            return

        step_dist = float(np.linalg.norm(pos - self._prev_pos))
        # Guard against rare NaNs
        if np.isfinite(step_dist):
            self._total += step_dist
        self._prev_pos = pos
        self._metric = float(self._total)


@dataclass
class TraveledDistanceMeasurementConfig(MeasurementConfig):
    type: str = TraveledDistance.__name__
    include_vertical: bool = False


# Register Hydra config node so YAML defaults can reference it
try:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(
        package="habitat.task.measurements.traveled_distance",
        group="habitat/task/measurements",
        name="traveled_distance",
        node=TraveledDistanceMeasurementConfig,
    )
except Exception:
    # Hydra not present; safe to ignore
    pass

