import math
from typing import Tuple, Optional

import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union

def merge_contour(contours):
    polys = [Polygon(c) for c in contours]
    merged = unary_union(polys)
    return np.array(merged.exterior.coords)

def quick_hull_simplify(points, angle_step_deg=10):
    center = np.mean(points, axis=0)
    vecs = points - center
    angles = np.arctan2(vecs[:,1], vecs[:,0])
    bins = np.floor((angles + np.pi) / np.deg2rad(angle_step_deg)).astype(int)

    simplified = []
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if len(idx) > 0:
            farthest_idx = idx[np.argmax(np.linalg.norm(vecs[idx], axis=1))]
            simplified.append(points[farthest_idx])

    return np.array(simplified), center

def closest_point_on_polygon(polygon, point):
    """
    polygon: np.ndarray of shape (N, 2) or (N, 1, 2) 
    point: (x, y) 
    """
    point = np.array(point, dtype=np.float32)
    if cv2.pointPolygonTest(polygon.reshape(-1, 1, 2), point, False) >=0:
        return point, 0

    min_dist = float('inf')
    closest_point = None

    if polygon.ndim == 3:
        polygon = polygon[:, 0, :]

    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        ab = b - a
        ap = point - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab

        dist = np.linalg.norm(point - proj)
        if dist < min_dist:
            min_dist = dist
            closest_point = proj

    return closest_point, min_dist

def get_submat(mat: np.ndarray, center, r):
    shape = mat.shape
    h, w = shape[:2]
    upper, lower = max(0, center[0] - r), min(h, center[0] + r)
    left, right = max(0, center[1] - r), min(w, center[1] + r)
    return mat[upper:lower, left:right]


def gen_mask(shape, pos, r):
    h, w = shape[:2]
    upper, lower = max(0, pos[0] - r), min(h, pos[0] + r)
    left, right = max(0, pos[1] - r), min(w, pos[1] + r)
    mask = np.zeros(shape, dtype=bool)
    mask[upper:lower, left:right] = 1
    return mask


def rho_theta(curr_pos: np.ndarray, curr_heading: float, curr_goal: np.ndarray) -> Tuple[float, float]:
    """Calculates polar coordinates (rho, theta) relative to a given position and
    heading to a given goal position. 'rho' is the distance from the agent to the goal,
    and theta is how many radians the agent must turn (to the left, CCW from above) to
    face the goal. Coordinates are in (x, y), where x is the distance forward/backwards,
    and y is the distance to the left or right (right is negative)

    Args:
        curr_pos (np.ndarray): Array of shape (2,) representing the current position.
        curr_heading (float): The current heading, in radians. It represents how many
            radians  the agent must turn to the left (CCW from above) from its initial
            heading to reach its current heading.
        curr_goal (np.ndarray): Array of shape (2,) representing the goal position.

    Returns:
        Tuple[float, float]: A tuple of floats representing the polar coordinates
            (rho, theta).
    """
    rotation_matrix = get_rotation_matrix(-curr_heading, ndims=2)
    local_goal = curr_goal - curr_pos
    local_goal = rotation_matrix @ local_goal

    rho = np.linalg.norm(local_goal)
    theta = np.arctan2(local_goal[1], local_goal[0])

    return float(rho), float(theta)


def get_rotation_matrix(angle: float, ndims: int = 2) -> np.ndarray:
    """Returns a 2x2 or 3x3 rotation matrix for a given angle; if 3x3, the z-axis is
    rotated."""
    if ndims == 2:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
    elif ndims == 3:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("ndims must be 2 or 3")


def wrap_heading(theta: float) -> float:
    """Wraps given angle to be between -pi and pi.

    Args:
        theta (float): The angle in radians.
    Returns:
        float: The wrapped angle in radians.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def calculate_vfov(hfov: float, width: int, height: int) -> float:
    """Calculates the vertical field of view (VFOV) based on the horizontal field of
    view (HFOV), width, and height of the image sensor.

    Args:
        hfov (float): The HFOV in radians.
        width (int): Width of the image sensor in pixels.
        height (int): Height of the image sensor in pixels.

    Returns:
        A float representing the VFOV in radians.
    """
    # Calculate the diagonal field of view (DFOV)
    dfov = 2 * math.atan(math.tan(hfov / 2) * math.sqrt((width**2 + height**2) / (width**2 + height**2)))

    # Calculate the vertical field of view (VFOV)
    vfov = 2 * math.atan(math.tan(dfov / 2) * (height / math.sqrt(width**2 + height**2)))

    return vfov


def within_fov_cone(
    cone_origin: np.ndarray,
    cone_angle: float,
    cone_fov: float,
    cone_range: float,
    points: np.ndarray,
) -> np.ndarray:
    """Checks if points are within a cone of a given origin, angle, fov, and range.

    Args:
        cone_origin (np.ndarray): The origin of the cone.
        cone_angle (float): The angle of the cone in radians.
        cone_fov (float): The field of view of the cone in radians.
        cone_range (float): The range of the cone.
        points (np.ndarray): The points to check.

    Returns:
        np.ndarray: The subarray of points that are within the cone.
    """
    directions = points[:, :3] - cone_origin
    dists = np.linalg.norm(directions, axis=1)
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    angle_diffs = np.mod(angles - cone_angle + np.pi, 2 * np.pi) - np.pi

    mask = np.logical_and(dists <= cone_range, np.abs(angle_diffs) <= cone_fov / 2)
    return points[mask]


def convert_to_global_frame(agent_pos: np.ndarray, agent_yaw: float, local_pos: np.ndarray) -> np.ndarray:
    """Converts a given position from the agent's local frame to the global frame.

    Args:
        agent_pos (np.ndarray): A 3D vector representing the agent's position in their
            local frame.
        agent_yaw (float): The agent's yaw in radians.
        local_pos (np.ndarray): A 3D vector representing the position to be converted in
            the agent's local frame.

    Returns:
        A 3D numpy array representing the position in the global frame.
    """
    # Append a homogeneous coordinate of 1 to the local position vector
    local_pos_homogeneous = np.append(local_pos, 1)

    # Construct the homogeneous transformation matrix
    transformation_matrix = xyz_yaw_to_tf_matrix(agent_pos, agent_yaw)

    # Perform the transformation using matrix multiplication
    global_pos_homogeneous = transformation_matrix.dot(local_pos_homogeneous)
    global_pos_homogeneous = global_pos_homogeneous[:3] / global_pos_homogeneous[-1]

    return global_pos_homogeneous


def extract_yaw(matrix: np.ndarray) -> float:
    """Extract the yaw angle from a 4x4 transformation matrix.

    Args:
        matrix (np.ndarray): A 4x4 transformation matrix.
    Returns:
        float: The yaw angle in radians.
    """
    assert matrix.shape == (4, 4), "The input matrix must be 4x4"
    rotation_matrix = matrix[:3, :3]

    # Compute the yaw angle
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return yaw


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    """Converts a given position and yaw angle to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians.
    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    transformation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw), np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )
    return transformation_matrix


def xyz_yaw_pitch_to_tf_matrix(xyz: np.ndarray, yaw: float, pitch_down: float) -> np.ndarray:
    """Converts position, yaw, and downward pitch to a 4x4 transformation matrix.

    The local camera frame follows the existing convention in this repo:
    x points forward, y points left, z points up. Positive ``pitch_down`` tilts
    the optical axis downward around the local y-axis.
    """
    x, y, z = xyz
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch_down), np.sin(pitch_down)

    rot_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rot_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float32,
    )

    transformation_matrix = np.eye(4, dtype=np.float32)
    transformation_matrix[:3, :3] = rot_z @ rot_y
    transformation_matrix[:3, 3] = np.asarray([x, y, z], dtype=np.float32)
    return transformation_matrix


def closest_point_within_threshold(points_array: np.ndarray, target_point: np.ndarray, threshold: float) -> int:
    """Find the point within the threshold distance that is closest to the target_point.

    Args:
        points_array (np.ndarray): An array of 2D points, where each point is a tuple
            (x, y).
        target_point (np.ndarray): The target 2D point (x, y).
        threshold (float): The maximum distance threshold.

    Returns:
        int: The index of the closest point within the threshold distance.
    """
    distances = np.sqrt((points_array[:, 0] - target_point[0]) ** 2 + (points_array[:, 1] - target_point[1]) ** 2)
    within_threshold = distances <= threshold

    if np.any(within_threshold):
        closest_index = np.argmin(distances)
        return int(closest_index)

    return -1


def transform_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Applies a 4x4 transform to points shaped (N, D) or (..., D) and preserves shape.

    Supports inputs where D is the last dimension (typically 3). If the input has
    rank > 2 (e.g., HxWx3), it is flattened, transformed, and reshaped back.
    """
    pts = np.asarray(points)
    original_shape = pts.shape
    d = original_shape[-1]
    flat = pts.reshape(-1, d)

    # Add homogeneous coordinate
    homogeneous_points = np.hstack((flat, np.ones((flat.shape[0], 1))))
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T

    # Remove the added homogeneous coordinate and divide by the last coordinate
    out = transformed_points[:, :d] / transformed_points[:, d:]
    return out.reshape(original_shape)


def get_point_cloud(
    depth_image: np.ndarray,
    mask: Optional[np.ndarray],
    fx: float,
    fy: float,
    preserve_hw: bool = False,
) -> np.ndarray:
    """Calculates the 3D coordinates (x, y, z) of points in the depth image based on
    the horizontal field of view (HFOV), the image width and height, the depth values,
    and the pixel x and y coordinates.

    Args:
        depth_image (np.ndarray): 2D depth image.
        mask (np.ndarray): 2D binary mask identifying relevant pixels.
        fx (float): Focal length in the x direction.
        fy (float): Focal length in the y direction.

    Returns:
        np.ndarray: If preserve_hw is False, returns (N, 3) array of selected points;
            otherwise returns a dense (H, W, 3) point cloud.
    """
    if preserve_hw:
        v, u = np.indices(depth_image.shape)
        z = depth_image
        x = (u - depth_image.shape[1] // 2) * z / fx
        y = (v - depth_image.shape[0] // 2) * z / fy
        cloud = np.stack((z, -x, -y), axis=-1)
        return cloud

    v, u = np.where(mask)
    z = depth_image[v, u]
    x = (u - depth_image.shape[1] // 2) * z / fx
    y = (v - depth_image.shape[0] // 2) * z / fy
    cloud = np.stack((z, -x, -y), axis=-1)

    return cloud


def get_fov(focal_length: float, image_height_or_width: int) -> float:
    """
    Given an fx and the image width, or an fy and the image height, returns the
    horizontal or vertical field of view, respectively.

    Args:
        focal_length: Focal length of the camera.
        image_height_or_width: Height or width of the image.

    Returns:
        Field of view in radians.
    """

    # Calculate the field of view using the formula
    fov = 2 * math.atan((image_height_or_width / 2) / focal_length)
    return fov


def pt_from_rho_theta(rho: float, theta: float) -> np.ndarray:
    """
    Given a rho and theta, computes the x and y coordinates.

    Args:
        rho: Distance from the origin.
        theta: Angle from the x-axis.

    Returns:
        numpy.ndarray: x and y coordinates.
    """
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)

    return np.array([x, y])


def get_projected_min(coords, values):
    """
    get_projected_min(coords: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    Compute per-coordinate minima after projecting a set of points onto a 2D grid defined by their (x, y) coordinates.
    """
    sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))
    sorted_coords = coords[sorted_indices]
    sorted_values = values[sorted_indices]
    diff = np.any(sorted_coords[1:] != sorted_coords[:-1], axis=1)
    group_starts = np.r_[0, np.where(diff)[0] + 1]
    min_values = np.minimum.reduceat(sorted_values, group_starts)
    unique_coords = sorted_coords[group_starts]
    return unique_coords, min_values


def dbscan(points: np.ndarray, eps: float = 0.2, min_samples: int = 100) -> np.ndarray:
    """CPU DBSCAN clustering to keep the largest non-noise cluster.

    Args:
        points: Point array of shape (N, D). If N==0, returns empty (0, D).
        eps: Neighborhood radius (meters).
        min_samples: Minimum samples per cluster.

    Returns:
        Points belonging to the largest cluster (float32). Empty if no clusters.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2:
        pts = pts.reshape(-1, pts.shape[-1])
    if pts.shape[0] == 0:
        return np.empty((0, pts.shape[1] if pts.ndim == 2 else 3), dtype=np.float32)
    from sklearn.cluster import DBSCAN as SKDBSCAN
    model = SKDBSCAN(eps=float(eps), min_samples=int(min_samples))
    labels = model.fit_predict(pts)
    valid = labels != -1
    if not np.any(valid):
        return np.empty((0, pts.shape[1]), dtype=np.float32)
    uniq, cnt = np.unique(labels[valid], return_counts=True)
    if uniq.size == 0:
        return np.empty((0, pts.shape[1]), dtype=np.float32)
    largest = uniq[np.argmax(cnt)]
    out = pts[labels == largest].astype(np.float32, copy=False)
    return out

def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]

def get_random_subarray(points: np.ndarray, size: int) -> np.ndarray:
    """
    This function returns a subarray of a given 3D points array. The size of the
    subarray is specified by the user. The elements of the subarray are randomly
    selected from the original array. If the size of the original array is smaller than
    the specified size, the function will simply return the original array.

    Args:
        points (numpy array): A numpy array of 3D points. Each element of the array is a
            3D point represented as a numpy array of size 3.
        size (int): The desired size of the subarray.

    Returns:
        numpy array: A subarray of the original points array.
    """
    if len(points) <= size:
        return points
    indices = np.random.choice(len(points), size, replace=False)
    return points[indices]
