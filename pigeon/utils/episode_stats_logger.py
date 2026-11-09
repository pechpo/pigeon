import os
from typing import Any, Dict, Tuple, Optional

import cv2
import numpy as np
from frontier_exploration.utils.general_utils import xyz_to_habitat
from pigeon.utils.geometry_utils import transform_points as cpu_transform_points

def episodic_xy_to_habitat_xy(
    xy: Any,
    infos: Dict[str, Any],
    gpu: Optional[Any] = None,
) -> Optional[Tuple[float, float]]:
    """Convert an episodic-frame (x, y) point to habitat world-frame (x, z).

    Mirrors the transform chain in `was_false_positive` so that logged
    positions live in the same frame as the dataset's goal view_points.
    Returns None when the required map metadata is unavailable.
    """
    if xy is None or not isinstance(xy, (list, tuple, np.ndarray)):
        return None
    top_down_map = infos.get("top_down_map")
    if not isinstance(top_down_map, dict) or "tf_episodic_to_global" not in top_down_map:
        return None

    xy_arr = np.asarray(xy, dtype=np.float64).reshape(-1)
    if xy_arr.size < 2:
        return None
    episodic_xyz = np.array([xy_arr[0], xy_arr[1], 0.0]).reshape(1, 3)
    tf = top_down_map["tf_episodic_to_global"]

    try:
        if gpu is not None and hasattr(gpu, "transform_points"):
            global_xyz = gpu.transform_points(tf, episodic_xyz)
        else:
            global_xyz = cpu_transform_points(tf, episodic_xyz)
    except Exception:
        global_xyz = cpu_transform_points(tf, episodic_xyz)

    habitat_xyz = xyz_to_habitat(global_xyz)
    # xyz_to_habitat yields (x, y, z); the floor plane is (x, z), matching the
    # [2, 0] indexing used by was_false_positive.
    return float(habitat_xyz[0, 0]), float(habitat_xyz[0, 2])


def sim_xy_to_grid_xy(
    upper_bound: Tuple[int, int],
    lower_bound: Tuple[int, int],
    grid_resolution: Tuple[int, int],
    sim_xy: np.ndarray,
    remove_duplicates: bool = True,
) -> np.ndarray:
    """Converts simulation coordinates to grid coordinates.

    Args:
        upper_bound (Tuple[int, int]): The upper bound of the grid.
        lower_bound (Tuple[int, int]): The lower bound of the grid.
        grid_resolution (Tuple[int, int]): The resolution of the grid.
        sim_xy (np.ndarray): A numpy array of 2D simulation coordinates.
        remove_duplicates (bool): Whether to remove duplicate grid coordinates.

    Returns:
        np.ndarray: A numpy array of 2D grid coordinates.
    """
    grid_size = np.array(
        [
            abs(upper_bound[1] - lower_bound[1]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        ]
    )
    # Clip points to be within the grid bounds
    sim_xy[:, 0] = np.clip(sim_xy[:, 0], lower_bound[1], upper_bound[1] - 1)
    sim_xy[:, 1] = np.clip(sim_xy[:, 1], lower_bound[0], upper_bound[0] - 1)
    grid_xy = ((sim_xy - lower_bound[::-1]) / grid_size).astype(int)

    if remove_duplicates:
        grid_xy = np.unique(grid_xy, axis=0)

    return grid_xy

def determine_failure_cause(infos: Dict, gpu: Optional[Any] = None) -> str:
    """Using the info and policy_info dicts, determine the cause of failure.

    Args:
        infos: The info dict from the environment after update with policy info.

    Returns:
        A string describing the cause of failure.
    """
    infos = _use_pre_forced_stop_policy_info(infos)
    if infos["target_detected"]:  # examine if the target is true
        if was_false_positive(infos, gpu):
            return "false_positive"
        else:
            if infos["stop_called"]:
                return "bad_stop_true_positive"
            else:
                return "timeout_true_positive"
    else:  # examine if the target was ever seen
        if was_target_seen(infos):
            return "false_negative"
        else:
            if infos["traveled_stairs"]:
                cause = "never_saw_target_traveled_stairs"
            else:
                cause = "never_saw_target_did_not_travel_stairs"
            if not infos["top_down_map"]["is_feasible"]:
                return cause + "_likely_infeasible"
            else:
                return cause + "_feasible"


def add_success_feasibility_suffix(cause: str, infos: Dict[str, Any]) -> str:
    """Append feasibility to the success label when map metadata is available."""
    if cause != "did_not_fail":
        return cause

    top_down_map = infos.get("top_down_map")
    if not isinstance(top_down_map, dict) or "is_feasible" not in top_down_map:
        return cause

    suffix = "feasible" if bool(top_down_map["is_feasible"]) else "likely_infeasible"
    return f"{cause}_{suffix}"


def _use_pre_forced_stop_policy_info(infos: Dict[str, Any]) -> Dict[str, Any]:
    if not infos.get("forced_stop") or infos.get("forced_stop_reason") != "last_step":
        return infos
    pre_forced_stop = infos.get("pre_forced_stop")
    if not isinstance(pre_forced_stop, dict):
        return infos
    adjusted = dict(infos)
    for key in ("nav_goal", "target_detected", "stop_called"):
        if key in pre_forced_stop:
            adjusted[key] = pre_forced_stop[key]
    return adjusted


def was_target_seen(infos: Dict[str, Any]) -> bool:
    top_down = infos["top_down_map"]
    # Ensure required masks exist
    if "target_bboxes_mask" not in top_down or "fog_of_war_mask" not in top_down:
        return False
    target_bboxes_mask = infos["top_down_map"]["target_bboxes_mask"]
    explored_area = infos["top_down_map"]["fog_of_war_mask"]
    # Dilate the target_bboxes_mask by 10 pixels to add a margin of error
    target_bboxes_mask = cv2.dilate(target_bboxes_mask, np.ones((10, 10)))
    target_explored = bool(np.any(np.logical_and(explored_area, target_bboxes_mask)))
    return target_explored


def was_false_positive(infos: Dict[str, Any], gpu: Optional[Any] = None) -> bool:
    """Return whether the point goal target is within a bounding box."""
    target_bboxes_mask = infos["top_down_map"]["target_bboxes_mask"]
    nav_goal_episodic_xy = infos.get("nav_goal")
    if nav_goal_episodic_xy is None or not isinstance(nav_goal_episodic_xy, (list, tuple, np.ndarray)):
        # No goal available; treat as false positive to be conservative
        return True
    nav_goal_episodic_xyz = np.array([nav_goal_episodic_xy[0], nav_goal_episodic_xy[1], 0]).reshape(1, 3)

    upper_bound = infos["top_down_map"]["upper_bound"]
    lower_bound = infos["top_down_map"]["lower_bound"]
    grid_resolution = infos["top_down_map"]["grid_resolution"]
    tf_episodic_to_global = infos["top_down_map"]["tf_episodic_to_global"]

    # Prefer GPU transform if available; fallback to CPU
    try:
        if gpu is not None and hasattr(gpu, "transform_points"):
            nav_goal_global_xyz = gpu.transform_points(tf_episodic_to_global, nav_goal_episodic_xyz)
        else:
            nav_goal_global_xyz = cpu_transform_points(tf_episodic_to_global, nav_goal_episodic_xyz)
    except Exception:
        nav_goal_global_xyz = cpu_transform_points(tf_episodic_to_global, nav_goal_episodic_xyz)
    nav_goal_global_habitat = xyz_to_habitat(nav_goal_global_xyz)
    nav_goal_global_habitat_xy = nav_goal_global_habitat[:, [2, 0]]

    grid_xy = sim_xy_to_grid_xy(
        upper_bound,
        lower_bound,
        grid_resolution,
        nav_goal_global_habitat_xy,
        remove_duplicates=True,
    )

    try:
        return target_bboxes_mask[grid_xy[0, 0], grid_xy[0, 1]] == 0
    except IndexError:
        # If the point goal is outside the map, assume it is a false positive
        return True


def remove_numpy_arrays(d: Any) -> Dict:
    if not isinstance(d, dict):
        return d

    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            new_dict[key] = remove_numpy_arrays(value)
        elif not isinstance(value, np.ndarray):
            new_dict[key] = value

    return new_dict
