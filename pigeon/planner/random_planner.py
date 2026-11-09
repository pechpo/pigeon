import random
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RandomPlanner:
    actions = ("move_forward", "turn_left", "turn_right", "stop")
    models = []

    def reset(self) -> None:
        pass

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        a = random.choice(self.actions[:-1])
        return {"action": a}

