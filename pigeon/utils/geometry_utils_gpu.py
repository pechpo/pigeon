"""
CuPy-accelerated geometry utilities.

Note: Each function imports `cupy as np` inside the function body to avoid EGL
issues during module import. All functions return CuPy arrays.
"""

from typing import Optional


def get_point_cloud(
    depth_image,
    mask: Optional[object],
    fx: float,
    fy: float,
    preserve_hw: bool = False,
):
    """GPU get_point_cloud using CuPy; returns CuPy array."""
    import cupy as np

    if preserve_hw:
        h, w = depth_image.shape[:2]
        v, u = np.indices((h, w))
        z = np.asarray(depth_image)
        x = (u - (w // 2)) * z / fx
        y = (v - (h // 2)) * z / fy
        cloud = np.stack((z, -x, -y), axis=-1)
        return cloud

    if mask is None:
        raise ValueError("mask must be provided when preserve_hw=False")
    h, w = depth_image.shape[:2]
    m = np.asarray(mask)
    v, u = np.where(m)
    z = np.asarray(depth_image)[v, u]
    if len(depth_image.shape) == 3 and depth_image.shape[-1] == 3:
        return z
    x = (u - (w // 2)) * z / fx
    y = (v - (h // 2)) * z / fy
    cloud = np.stack((z, -x, -y), axis=-1)
    return cloud


def transform_points(transformation_matrix, points):
    """GPU transform_points using CuPy; returns CuPy array."""
    import cupy as np

    pts = np.asarray(points)
    original_shape = pts.shape
    d = original_shape[-1]
    flat = pts.reshape(-1, d)

    tm = np.asarray(transformation_matrix)
    ones = np.ones((flat.shape[0], 1), dtype=flat.dtype)
    homogeneous = np.hstack((flat, ones))
    transformed = (tm @ homogeneous.T).T
    out = transformed[:, :d] / transformed[:, d:]
    return out.reshape(original_shape)


def filter_points_by_height(points, min_height: float, max_height: float):
    """GPU z-filter using CuPy; returns CuPy array."""
    import cupy as np

    pts = np.asarray(points)
    if pts.size == 0:
        cols = points.shape[1] if getattr(points, "ndim", 2) == 2 else 3
        return np.empty((0, cols), dtype=pts.dtype)
    mask = (pts[:, 2] >= min_height) & (pts[:, 2] <= max_height)
    out = pts[mask]
    return out


def dbscan(points, eps: float = 0.2, min_samples: int = 100):
    """GPU DBSCAN via cuML; returns CuPy array of largest cluster points."""
    import cupy as np
    from cuml.cluster import DBSCAN as CuDBSCAN  # type: ignore

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2:
        pts = pts.reshape(-1, pts.shape[-1])
    if pts.shape[0] == 0:
        return np.empty((0, pts.shape[1] if pts.ndim == 2 else 3), dtype=pts.dtype)

    model = CuDBSCAN(eps=float(eps), min_samples=int(min_samples))
    labels = model.fit_predict(pts)

    valid = labels != -1
    if not np.any(valid):
        return np.empty((0, pts.shape[1]), dtype=pts.dtype)
    uniq, cnt = np.unique(labels[valid], return_counts=True)
    if uniq.size == 0:
        return np.empty((0, pts.shape[1]), dtype=pts.dtype)
    largest = uniq[np.argmax(cnt)]
    sel = pts[labels == largest]
    return sel

def get_random_subarray(points, size: int):
    """CuPy random downsample; returns CuPy array."""
    import cupy as np
    if len(points) <= size:
        return np.asarray(points)
    pts = np.asarray(points)
    idx = np.random.choice(pts.shape[0], size, replace=False)
    sampled = pts[idx]
    return sampled
