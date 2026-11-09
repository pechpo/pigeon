from typing import Any, Dict, Optional, List

import cv2
import numpy as np
from habitat.utils.visualizations import maps
from frontier_exploration.utils.general_utils import xyz_to_habitat
from pigeon.utils.episode_stats_logger import sim_xy_to_grid_xy
from pigeon.utils.geometry_utils import transform_points as cpu_transform_points
from pigeon.utils.img_utils import generate_text_image
from ..models.detections import draw_bounding_box
from .img_utils import reorient_rescale_map, crop_white_border


def _select_topdown_map(info: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Pick the topdown map dict, preferring explicit override then exploration_map."""
    if override is not None:
        return override
    if not isinstance(info, dict):
        return None
    if "exploration_map" in info:
        return info.get("exploration_map")
    if "top_down_map" in info:
        return info.get("top_down_map")
    return None


def _cloud_to_grid(
    points_xyz: np.ndarray,
    tdm: Dict[str, Any],
    map_shape: tuple[int, int],
    remove_duplicates: bool = True,
) -> Optional[np.ndarray]:
    """Project episodic-frame XYZ points into topdown grid coordinates."""
    upper_bound = tdm.get("upper_bound", None)
    lower_bound = tdm.get("lower_bound", None)
    grid_resolution = tdm.get("grid_resolution", None)
    tf_episodic_to_global = tdm.get("tf_episodic_to_global", None)
    if (
        points_xyz is None
        or not isinstance(points_xyz, np.ndarray)
        or points_xyz.size == 0
        or upper_bound is None
        or lower_bound is None
        or grid_resolution is None
    ):
        return None
    pts = points_xyz[:, :3]
    if tf_episodic_to_global is not None:
        pts = cpu_transform_points(np.asarray(tf_episodic_to_global), pts)
    habitat_xyz = xyz_to_habitat(pts)
    habitat_xy = habitat_xyz[:, [2, 0]]
    grid_xy = sim_xy_to_grid_xy(
        upper_bound,
        lower_bound,
        grid_resolution,
        habitat_xy,
        remove_duplicates=remove_duplicates,
    )
    if not isinstance(grid_xy, np.ndarray) or grid_xy.size == 0:
        return None
    h, w = map_shape
    yy = np.clip(grid_xy[:, 0], 0, h - 1)
    xx = np.clip(grid_xy[:, 1], 0, w - 1)
    return np.stack([yy, xx], axis=1).astype(int)


def _overlay_fusion_objects(top_down_bgr: np.ndarray, tdm: Dict[str, Any], fusion: Dict[str, Any]) -> None:
    """Draw suspect/reliable fusion clouds on the topdown map in-place."""
    overlay_specs = [
        ("suspected", (0, 165, 255), "S"),
        ("reliable", (0, 180, 0), "R"),
    ]
    top_down_map = tdm.get("map")
    if top_down_map is None:
        return
    for key, color, prefix in overlay_specs:
        entries = fusion.get(key) or []
        for obj in entries:
            pts = obj.get("points", None)
            if pts is None:
                continue
            grid_coords = _cloud_to_grid(np.asarray(pts), tdm, top_down_map.shape, remove_duplicates=False)
            if grid_coords is None:
                continue
            yy, xx = grid_coords[:, 0], grid_coords[:, 1]
            top_down_bgr[yy, xx] = color
            cy = int(np.median(yy))
            cx = int(np.median(xx))
            label = str(obj.get("label", "") or prefix)
            conf = obj.get("conf", "")
            cv2.putText(
                top_down_bgr,
                f"{prefix}:{label} {conf:.3f}",
                (cx + 4, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3,
                cv2.LINE_AA,
            )


def _overlay_cloud_points(
    top_down_bgr: np.ndarray,
    tdm: Dict[str, Any],
    points_xyz: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """Draw a point cloud on the topdown map in-place."""
    top_down_map = tdm.get("map")
    if top_down_map is None:
        return
    pts_np = np.asarray(points_xyz)
    if pts_np.ndim != 2 or pts_np.shape[0] == 0 or pts_np.shape[1] < 3:
        return
    grid_coords = _cloud_to_grid(pts_np, tdm, top_down_map.shape, remove_duplicates=True)
    if grid_coords is None:
        return
    top_down_bgr[grid_coords[:, 0], grid_coords[:, 1]] = color


def _overlay_numeric_value(
    top_down_bgr: np.ndarray,
    grid_coords: np.ndarray,
    value: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a numeric annotation near the median of projected grid points."""
    if grid_coords.ndim != 2 or grid_coords.shape[0] == 0:
        return
    if not np.isfinite(value):
        return
    yy, xx = grid_coords[:, 0], grid_coords[:, 1]
    cy = int(np.median(yy))
    cx = int(np.median(xx))
    text = f"{float(value):.2f}"
    org = (cx + 4, cy - 4)
    cv2.putText(
        top_down_bgr,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        top_down_bgr,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _colorize_topdown_map(
    top_down_map: np.ndarray,
    fog_of_war_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Match Habitat's colors without allocating a float RGB image per frame."""
    colors = maps.TOP_DOWN_MAP_COLORS
    if fog_of_war_mask is None:
        return colors[top_down_map]

    # Habitat dims unseen valid cells by 0.5 and leaves invalid cells alone.
    # Apply that operation to the small palette instead of every map pixel.
    palettes = np.stack(((colors * 0.5).astype(np.uint8), colors))
    palettes[0, maps.MAP_INVALID_POINT] = colors[maps.MAP_INVALID_POINT]
    return palettes[fog_of_war_mask, top_down_map]


def build_topdown_viz(
    info: Dict[str, Any],
    target_height: int = 480,
    tdm_override: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    # Prefer exploration_map if present; otherwise fall back to top_down_map
    tdm = _select_topdown_map(info, tdm_override)
    if tdm is None:
        return None
    # Expected structure in habitat-lab: {"map": np.ndarray, "fog_of_war_mask": np.ndarray, "agent_map_coord": (y,x) or list, "agent_angle": float}
    top_down = tdm.get("map", None)
    if top_down is None:
        return None

    fog = tdm.get("fog_of_war_mask", None)
    top_down_rgb = _colorize_topdown_map(top_down, fog)
    # Optionally overlay object point cloud (episodic frame) onto the topdown map.
    cloud = info.get("object_point_cloud", None)
    if cloud is not None:
        cloud_np = np.asarray(cloud)
        if cloud_np.ndim == 2 and cloud_np.shape[0] > 0 and cloud_np.shape[1] >= 3:
            _overlay_cloud_points(top_down_rgb, tdm, cloud_np, (255, 0, 255))

    stair_cloud = info.get("stair_point_cloud", None)
    if stair_cloud is not None:
        stair_cloud_np = np.asarray(stair_cloud)
        if stair_cloud_np.ndim == 2 and stair_cloud_np.shape[0] > 0 and stair_cloud_np.shape[1] >= 3:
            _overlay_cloud_points(top_down_rgb, tdm, stair_cloud_np, (0, 255, 255))

    top_down = cv2.cvtColor(top_down_rgb, cv2.COLOR_RGB2BGR)

    stair_confidence = info.get("stair_confidence", None)
    if stair_cloud is not None and stair_confidence is not None:
        stair_cloud_np = np.asarray(stair_cloud)
        if stair_cloud_np.ndim == 2 and stair_cloud_np.shape[0] > 0 and stair_cloud_np.shape[1] >= 3:
            grid_coords = _cloud_to_grid(stair_cloud_np, tdm, top_down_rgb.shape[:2], remove_duplicates=True)
            if grid_coords is not None:
                try:
                    _overlay_numeric_value(top_down, grid_coords, float(stair_confidence), (255, 255, 0))
                except (TypeError, ValueError):
                    pass

    # Overlay suspected/reliable fusion objects when provided
    fusion = info.get("fusion_objects_viz", None)
    if isinstance(fusion, dict):
        _overlay_fusion_objects(top_down, tdm, fusion)

    # Support single or multi-agent entries
    agent_coords = tdm.get("agent_map_coord", [])
    agent_angles = tdm.get("agent_angle", [])
    if isinstance(agent_coords, (tuple, list)) and (len(agent_coords) == 2 and isinstance(agent_coords[0], (int, float))):
        agent_coords = [agent_coords]
        agent_angles = [tdm.get("agent_angle", 0.0)]

    for coord, ang in zip(agent_coords, agent_angles):
        top_down = maps.draw_agent(
            image=top_down,
            agent_center_coord=coord,
            agent_rotation=ang,
            agent_radius_px=max(1, min(top_down.shape[0:2]) // 32),
        )

    if top_down.shape[0] > top_down.shape[1]:
        top_down = np.rot90(top_down, 1)

    h, w = top_down.shape[:2]
    out_w = int(w * (float(target_height) / h))
    top_down = cv2.resize(top_down, (out_w, target_height), interpolation=cv2.INTER_CUBIC)
    return top_down


def _depth_to_bgr(depth: np.ndarray, target_height: int) -> np.ndarray:
    d = depth
    if d.ndim == 3:
        d = d[..., 0]
    d = d.astype(np.float32, copy=False)
    # Handle invalid values
    d[np.isnan(d)] = 0.0
    d[np.isinf(d)] = 0.0
    # Normalize robustly to [0, 255]
    pos = d[d > 0]
    if pos.size > 0:
        vmax = float(np.percentile(pos, 98))
        if vmax <= 0:
            vmax = float(pos.max())
    else:
        vmax = 1.0
    scaled = np.clip(d / max(vmax, 1e-6), 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_PLASMA)
    # Resize to requested height preserving aspect
    h, w = color.shape[:2]
    out_w = int(round(w * (float(target_height) / max(h, 1))))
    color = cv2.resize(color, (out_w, target_height), interpolation=cv2.INTER_AREA)
    return color


def _depth_to_grey_bgr(depth: np.ndarray, target_height: int) -> np.ndarray:
    """Greyscale depth (min-max) -> 3-channel BGR, resized to target height."""
    d = depth[..., 0] if depth.ndim == 3 else depth
    d = np.nan_to_num(d).astype(np.float32)
    gray = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    out_w = int(round(w * (float(target_height) / max(h, 1))))
    return cv2.resize(bgr, (out_w, target_height), interpolation=cv2.INTER_AREA)


def draw_det_set(
    depth_bgr: np.ndarray,
    det_seg: Dict[str, Any],
) -> np.ndarray:        
    # Multi-object visualization: draw each box and mask independently
    multi_bboxes = det_seg.get("bboxes_px") or []
    multi_scores = det_seg.get("scores") or []
    multi_classes = det_seg.get("classes") or []
    h, w = depth_bgr.shape[:2]
    for i, b in enumerate(multi_bboxes):
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        # If bbox appears normalized, scale to image size
        if max(x1, y1, x2, y2) <= 1.0:
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h
        x1i = int(max(0, min(w - 1, round(x1))))
        y1i = int(max(0, min(h - 1, round(y1))))
        x2i = int(max(0, min(w - 1, round(x2))))
        y2i = int(max(0, min(h - 1, round(y2))))

        cls_i = multi_classes[i] if i < len(multi_classes) else ""
        score_i = multi_scores[i] if i < len(multi_scores) else 0.0
        score_f = float(score_i) if isinstance(score_i, (int, float)) else 0.0

        depth_bgr = draw_bounding_box(
            image=depth_bgr,
            point1=(x1i, y1i),
            point2=(x2i, y2i),
            class_name=str(cls_i),
            score=score_f,
            image_rgb=False,
        )

    multi_masks = det_seg.get("masks") or []
    if isinstance(multi_masks, np.ndarray):
        if multi_masks.ndim == 3:
            masks_iter = [multi_masks[i] for i in range(multi_masks.shape[0])]
        else:
            masks_iter = [multi_masks]
    elif isinstance(multi_masks, (list, tuple)):
        masks_iter = multi_masks
    else:
        masks_iter = []
    for m in masks_iter:
        if m is None:
            continue
        m_arr = np.asarray(m)
        if m_arr.ndim == 3:
            m_arr = m_arr[..., 0]
        m_arr = (m_arr.astype(np.uint8) > 0).astype(np.uint8)
        if m_arr.max() == 0:
            continue
        cnts, _ = cv2.findContours(m_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(depth_bgr, cnts, -1, (0, 255, 0), 2)

    return depth_bgr


def _pad_width(img: np.ndarray, target_w: int, fill_value: int = 0) -> np.ndarray:
    if img.shape[1] == target_w:
        return img
    if img.shape[1] > target_w:
        return img[:, :target_w]
    pad = np.full((img.shape[0], target_w - img.shape[1], 3), fill_value, dtype=np.uint8)
    return np.hstack([img, pad])


def _prepare_panel(img: Optional[np.ndarray], target_height: int, keep_layout: bool = False) -> Optional[np.ndarray]:
    if img is None:
        return None
    panel = img
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    if not keep_layout:
        try:
            panel = reorient_rescale_map(panel)
        except Exception:
            try:
                panel = crop_white_border(panel)
            except Exception:
                pass
    if panel.shape[0] != target_height:
        new_w = int(round(panel.shape[1] * (target_height / max(panel.shape[0], 1))))
        panel = cv2.resize(panel, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return panel


def _stack_maps_column(
    top_img: Optional[np.ndarray],
    bottom_img: Optional[np.ndarray],
    panel_height: int,
) -> Optional[np.ndarray]:
    if top_img is None and bottom_img is None:
        return None
    top_panel = _prepare_panel(top_img, panel_height)
    bottom_panel = _prepare_panel(bottom_img, panel_height)
    if top_panel is None and bottom_panel is not None:
        top_panel = np.full_like(bottom_panel, 255)
    if bottom_panel is None and top_panel is not None:
        bottom_panel = np.full_like(top_panel, 255)
    if top_panel is None or bottom_panel is None:
        return None
    col_w = max(top_panel.shape[1], bottom_panel.shape[1])
    return np.vstack([
        _pad_width(top_panel, col_w, fill_value=255),
        _pad_width(bottom_panel, col_w, fill_value=255),
    ])


def _stack_panel_column(
    imgs: list[Optional[np.ndarray]],
    panel_height: int,
    *,
    fill_value: int = 0,
    keep_layout_flags: Optional[list[bool]] = None,
) -> Optional[np.ndarray]:
    if not imgs:
        return None
    if keep_layout_flags is None:
        keep_layout_flags = [False] * len(imgs)
    prepared: list[Optional[np.ndarray]] = []
    for img, keep_layout in zip(imgs, keep_layout_flags):
        prepared.append(_prepare_panel(img, panel_height, keep_layout=keep_layout))
    first_valid = next((panel for panel in prepared if panel is not None), None)
    if first_valid is None:
        return None
    col_w = max(panel.shape[1] for panel in prepared if panel is not None)
    blank = np.full((panel_height, col_w, 3), fill_value, dtype=np.uint8)
    rows = [
        _pad_width(panel if panel is not None else blank, col_w, fill_value=fill_value)
        for panel in prepared
    ]
    return np.vstack(rows)


def compose_frame(
    rgb: Optional[np.ndarray],
    info: Dict[str, Any],
    planner_viz: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    det_seg: Optional[Dict[str, Any]] = None,
    value_map_viz: Optional[np.ndarray] = None,
    aux_map_viz: Optional[np.ndarray] = None,
    height_map_viz: Optional[np.ndarray] = None,
    right_panel_viz: Optional[np.ndarray] = None,
    bottom_text: Optional[str] = None,
    bottom_text_min_height: int = 180,
) -> Optional[np.ndarray]:
    """Compose the main grid with optional side columns on the right."""
    if rgb is None:
        return None

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    top_h_target = bgr.shape[0]
    tdm = _select_topdown_map(info)

    top_right = build_topdown_viz(info, target_height=top_h_target, tdm_override=tdm)
    if top_right is None:
        top_right = np.zeros((top_h_target, bgr.shape[1], 3), dtype=np.uint8)

    if depth is not None:
        bottom_left = _depth_to_grey_bgr(depth, target_height=top_h_target)
        if isinstance(det_seg, dict) and det_seg:
            bottom_left = draw_det_set(bottom_left, det_seg)
    else:
        bottom_left = np.zeros_like(bgr)

    planner_panel = _prepare_panel(planner_viz, top_h_target)
    left_col = _stack_panel_column(
        [bgr, bottom_left, top_right],
        top_h_target,
        fill_value=0,
        keep_layout_flags=[True, True, True],
    )
    right_col = _stack_panel_column(
        [planner_panel, aux_map_viz, height_map_viz],
        top_h_target,
        fill_value=255,
        keep_layout_flags=[True, False, False],
    )
    if left_col is None:
        return None
    if right_col is None:
        right_col = np.full((left_col.shape[0], left_col.shape[1], 3), 255, dtype=np.uint8)
    base_grid = np.hstack([left_col, right_col])
    base_h = base_grid.shape[0]

    right_columns: list[np.ndarray] = []
    value_col = _prepare_panel(value_map_viz, base_h)
    if value_col is not None:
        right_columns.append(value_col)

    right_panel_col = _prepare_panel(right_panel_viz, base_h, keep_layout=True)
    if right_panel_col is not None:
        right_columns.append(right_panel_col)

    final_parts: list[np.ndarray] = [base_grid, *right_columns]
    final = np.hstack(final_parts)

    if bottom_text is not None:
        full_w = final.shape[1]
        if isinstance(bottom_text, str) and bottom_text.strip() != "":
            text_img = generate_text_image(full_w, bottom_text)
            if text_img.ndim == 2:
                text_img = cv2.cvtColor(text_img, cv2.COLOR_GRAY2BGR)
        else:
            text_img = np.full((max(1, bottom_text_min_height), full_w, 3), 255, dtype=np.uint8)
        if text_img.shape[0] < bottom_text_min_height:
            pad_h = bottom_text_min_height - text_img.shape[0]
            pad = np.full((pad_h, text_img.shape[1], 3), 255, dtype=np.uint8)
            text_img = np.vstack([text_img, pad])
        final = np.vstack([final, text_img])

    return final


def overlay_status_bar(frame_bgr: np.ndarray, steps: int, max_steps: int, extra_text: Optional[str] = None) -> np.ndarray:
    """Append a status bar below the frame (no overlay on content).

    The bar shows optional `extra_text` (e.g., episode name/index) and the step counter.
    Returns a new annotated frame; does not modify the input.
    """
    try:
        h, w = frame_bgr.shape[:2]
        bar_h = max(28, h // 24)
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar[:] = (16, 16, 16)

        base = f"step: {steps}/{max_steps}"
        text = (f"{extra_text} | {base}" if extra_text else base)
        cv2.putText(
            bar,
            text,
            (12, int(bar_h * 0.7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return np.vstack([frame_bgr, bar])
    except Exception:
        return frame_bgr


# --------------------------- VLM Right-Panel Helpers ---------------------------
def fit_to_slot(img_bgr: np.ndarray, slot_h: int = 480, slot_w: int = 640, pad_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Letterbox an image into a fixed slot (slot_h x slot_w), preserving aspect.

    Ensures every tile in the VLM panel is the same size for clean layout.
    """
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((slot_h, slot_w, 3), pad_color, dtype=np.uint8)
    scale = min(slot_w / float(w), slot_h / float(h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_AREA)
    canvas = np.full((slot_h, slot_w, 3), pad_color, dtype=np.uint8)
    y0 = (slot_h - resized.shape[0]) // 2
    x0 = (slot_w - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas


def _draw_numbered_marker_bgr(img_bgr: np.ndarray, label: int, x: int, y: int, radius: int = 9) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    cv2.circle(img_bgr, (x, y), radius, (100, 0, 0), 2)
    cv2.circle(img_bgr, (x, y), radius - 1, (0, 0, 255), -1)
    text = str(label)
    x_offset = [-3, 3] if len(text) == 2 else [0]
    for idx, ch in enumerate(text):
        ch_size = cv2.getTextSize(ch, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        ch_x = x - ch_size[0] // 2 + x_offset[idx]
        ch_y = y + ch_size[1] // 2
        cv2.putText(img_bgr, ch, (ch_x, ch_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img_bgr


def mark_points_bgr(
    img_bgr: np.ndarray,
    points_px: np.ndarray,
    labels: Optional[List[int]] = None,
    radius: int = 9,
) -> np.ndarray:
    """Draw numbered circular markers at pixel locations on a BGR image."""
    points = np.asarray(points_px)
    if points.size == 0:
        return img_bgr
    if points.ndim == 1:
        points = points.reshape(1, 2)
    if labels is None:
        labels = list(range(1, points.shape[0] + 1))
    for idx, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        label = int(labels[idx]) if idx < len(labels) else idx + 1
        _draw_numbered_marker_bgr(img_bgr, label, x, y, radius=radius)
    return img_bgr


def mark_image_bgr(img_bgr: np.ndarray, label: int, x: int, y: int, radius: int = 9) -> np.ndarray:
    """Draw a red circular marker with numeric label at (x,y) on a BGR image."""
    return mark_points_bgr(img_bgr, np.array([[x, y]], dtype=np.int32), labels=[label], radius=radius)


def build_vlm_panel(
    imgs_rgb: Optional[List[np.ndarray]],
    text: Optional[str] = None,
    slot_h: int = 480,
    slot_w: int = 640,
    reserve_text_row: bool = False,
    min_text_height: int = 180,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> np.ndarray:
    """Build a fixed VLM grid; optional text row appended.

    - Always returns width `grid_cols * slot_w`.
    - Height is `grid_rows * slot_h` without text; with text or when `reserve_text_row=True`,
      grows by at least `min_text_height`.
    """
    max_tiles = max(1, int(grid_rows) * int(grid_cols))
    imgs_rgb = [] if imgs_rgb is None else list(imgs_rgb)
    imgs_bgr = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in imgs_rgb[:max_tiles]]
    while len(imgs_bgr) < max_tiles:
        imgs_bgr.append(np.full((slot_h, slot_w, 3), 255, dtype=np.uint8))
    slots = [fit_to_slot(im, slot_h, slot_w) for im in imgs_bgr]
    grid_rows_bgr = []
    for row_idx in range(grid_rows):
        start = row_idx * grid_cols
        grid_rows_bgr.append(np.hstack(slots[start:start + grid_cols]))
    grid = np.vstack(grid_rows_bgr)

    # Decide whether to append a text row
    if not text and not reserve_text_row:
        panel = grid
    else:
        # Build or reserve a text image area
        if text:
            text_img = generate_text_image(grid.shape[1], text)
            if text_img.ndim == 2:
                text_img = cv2.cvtColor(text_img, cv2.COLOR_GRAY2BGR)
        else:
            # Reserve a blank white area for text
            text_img = np.full((max(1, min_text_height), grid.shape[1], 3), 255, dtype=np.uint8)

        # Ensure a minimum text height for readability
        if text_img.shape[0] < min_text_height:
            pad_h = min_text_height - text_img.shape[0]
            pad = np.full((pad_h, text_img.shape[1], 3), 255, dtype=np.uint8)
            text_img = np.vstack([text_img, pad])

        panel = np.vstack([grid, text_img])

    return panel
