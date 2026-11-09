import numpy as np
from .random_object_planner import RandomObjectPlanner

class ClosestObjectPlanner(RandomObjectPlanner):
    def _choose_goal(self, passby: bool = True) -> None:
        frontiers = self._frontiers
        if frontiers.size >= 2:
            dists = np.linalg.norm(frontiers - np.asarray(self._curr_xy, dtype=np.float32), axis=1)
            nearest_idx = int(np.argmin(dists))
            self._goal_xy = frontiers[nearest_idx].astype(np.float32)
        else:
            self._goal_xy = None
        # Update visualization highlight for the selected frontier
        if hasattr(self._omap, "set_selected_frontier"):
            self._omap.set_selected_frontier(self._goal_xy)
