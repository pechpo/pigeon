from dataclasses import dataclass
from typing import Any, List

import numpy as np
from habitat import registry
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Simulator
from habitat.config.default_structured_configs import MeasurementConfig
from omegaconf import DictConfig


@registry.register_measure
class TraveledStairs(Measure):
    cls_uuid: str = "traveled_stairs"

    def __init__(self, sim: Simulator, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        self._history: List[float] = []
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return TraveledStairs.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._history = []
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        curr_z = float(self._sim.get_agent_state().position[1])
        self._history.append(curr_z)
        # 1 if vertical travel exceeds 0.9m within episode
        self._metric = int(np.ptp(self._history) > 0.9)


@dataclass
class TraveledStairsMeasurementConfig(MeasurementConfig):
    type: str = TraveledStairs.__name__


# Register Hydra config node so the defaults list can reference it when using Hydra-style YAML
try:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(
        package="habitat.task.measurements.traveled_stairs",
        group="habitat/task/measurements",
        name="traveled_stairs",
        node=TraveledStairsMeasurementConfig,
    )
except Exception:
    # Hydra not present; safe to ignore as long as YAML is explicit
    pass
