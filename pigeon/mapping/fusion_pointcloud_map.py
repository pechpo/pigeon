from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import cupy as cp
import open3d as o3d
import torch

from ..utils.geometry_utils import dbscan, get_point_cloud, transform_points
from ..utils.img_utils import concat_images, mask_far_depth_rgb
from ..utils.stair_vlm import STAIR_DIRECTION_PROMPT, parse_stair_direction_response
from .object_pointcloud_map import ObjectPointCloudMap, get_random_subarray, too_offset
from ..models.vlm import VLMClient

from pigeon.utils.geometry_utils_gpu import (
    get_point_cloud,
    dbscan,
    get_random_subarray,
)
from pigeon.utils.geometry_utils import rho_theta, extract_yaw, within_fov_cone


def get_point_cloud(
    depth_or_pcd: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
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
        v, u = np.indices(depth_or_pcd.shape)
        z = depth_or_pcd
        x = (u - depth_or_pcd.shape[1] // 2) * z / fx
        y = (v - depth_or_pcd.shape[0] // 2) * z / fy
        cloud = np.stack((z, -x, -y), axis=-1)
        return cloud

    v, u = np.where(mask)
    if len(depth_or_pcd.shape) == 3 and depth_or_pcd.shape[-1] == 3:
        return depth_or_pcd[v, u]
    z = depth_or_pcd[v, u]
    x = (u - depth_or_pcd.shape[1] // 2) * z / fx
    y = (v - depth_or_pcd.shape[0] // 2) * z / fy
    cloud = np.stack((z, -x, -y), axis=-1)

    return cloud


_DEFAULT_TEXT_PROMPT: Dict[str, str] = {
    "Detection Confirmation": (
        (
            """
You are an image recognition assistant. You will be presented with an object's image captured from single or multiple views. Your task is to check whether the {goal_object} is present in the given image. Please do not confuse {goal_object} with the following objects: {object_list}. Only respond "True" if you are certain the {goal_object} is clearly present in the image and located indoors. Respond "False" if the target object is clearly not present. Respond "Unconfirmed" if you are unsure whether the target object is present, or if its location cannot be confidently determined. Respond "Outdoor" if it is present but located outdoors.
Please follow the format like this, without bracket --
<thought> [put your step-by-step thinking process here] </thought>
<answer> [put True, False, Unconfirmed or Outdoor here] </answer>
"""
        ).strip()
    ),
}

@dataclass
class LabelTrack:
    """Container describing fused 3D evidences for a single semantic label."""

    volume: int = 0
    confidence: float = 0.0
    pcd: cp.ndarray = field(default_factory=lambda: cp.zeros((0, 3), dtype=cp.float32))

    @property
    def is_empty(self) -> bool:
        return len(self.pcd) == 0


@dataclass
class FusedObject:
    """A fused 3D object which tracks multiple labels that might correspond to the same object."""

    labels: Dict[str, LabelTrack]
    id: int
    best_label: str
    bbox_min: np.ndarray = None
    bbox_max: np.ndarray = None
    confirmed: bool = False
    in_sight: bool = True
    blacklisted: bool = False
    confirmation_count: int = 0
    true_confirmation_count: int = 0
    directional_confirmation_count: int = 0
    last_confirm_step: int = -3
    navigation_fail_count: int = 0
    need_more_view_pos: bool = False
    last_submitted_view_pos: Optional[int] = None
    view_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    view_pos_valid: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    visited_view_pos_count: int = 0
    dense_view_sampling_attempted: bool = False
    center: Optional[np.ndarray] = None
    image_ids: Optional[List[Optional[int]]] = None

    def ensure_label(self, label: str) -> LabelTrack:
        if label not in self.labels:
            self.labels[label] = LabelTrack()
        return self.labels[label]

    def non_empty_tracks(self) -> Iterable[LabelTrack]:
        for track in self.labels.values():
            if not track.is_empty:
                yield track

    def non_empty_labels_tracks(self) -> Iterable[LabelTrack]:
        for label, track in self.labels.items():
            if not track.is_empty:
                yield label, track


class FusionPointCloudMap(ObjectPointCloudMap):
    """Multi-view point-cloud fusion for object hypotheses.

    The class is a refactor of the original VLFM ``DetectionFusionMap`` mixin.  It keeps
    the capabilities that are required by the planners (object management, reliability
    scoring, multi-view confirmation hooks) while presenting a clearer API based on the
    ``ObjectPointCloudMap`` interface.
    """

    use_dbscan: bool = True
    filter_out_of_range_points: bool = True

    def __init__(
        self,
        erosion_size: float,
        reliable_threshold: float = 0.7,  # Confidence threshold to consider an object a reliable target
        suspected_threshold: float = 0.25,  # Confidence threshold to consider an object a target which needs multi-view confirmation
        merge_radius: float = 0.2,
        max_detection_points: int = 3000,
        pre_dbscan_points: bool = True,  # TODO: turn it to False when use 3090 / 4090
        scene_cloud_size: int = 10000,
        explored_area: Optional[np.ndarray] = None,
        steppable_explored_area: Optional[np.ndarray] = None,
        episode_pixel_origin: Optional[Tuple[int, int]] = None,
        pixels_per_meter: Optional[float] = None,
        voxel_size: float = 0.05,
        obstacle_map: Optional[Any] = None,
    ) -> None:
        super().__init__(erosion_size)
        self.reliable_threshold = reliable_threshold
        self.suspected_threshold = suspected_threshold
        self.detection_count_threshold = 0
        self.merge_radius = merge_radius
        self.max_detection_points = max_detection_points
        self.pre_dbscan_points = pre_dbscan_points
        self.scene_cloud_size = scene_cloud_size
        self.explored_area = explored_area
        self.steppable_explored_area = steppable_explored_area
        self._episode_pixel_origin = episode_pixel_origin
        self.pixels_per_meter = pixels_per_meter
        self.voxel_size = voxel_size
        self._min_intersection_points = 20
        self._scene_visibility_threshold = 20
        self._navigation_blacklist_threshold = 400
        self.device = "cuda:0"
        self.o3d_device = o3d.core.Device(self.device)
        self._vlm_client: Optional[Any] = None
        # Keep the most recent scene cloud for estimating the local floor when
        # choosing a navigation point.  The fused object clouds themselves are
        # left untouched; only the temporary navigation cloud is filtered.
        self._latest_scene_cloud = cp.zeros((0, 3), dtype=cp.float32)
        self._last_navigation_filter_records: List[Dict[str, Any]] = []
        self._vlm_viz_callback: Optional[Any] = None
        self._approve_downstair_callback: Optional[Callable[[], bool]] = None
        self._stair_direction_confirmation_enabled: bool = True
        self._confirmation_prompt_template = _DEFAULT_TEXT_PROMPT["Detection Confirmation"]
        self._confirmation_prompt = self._confirmation_prompt_template
        self._stair_vlm_far_threshold_m: float = 4.75
        self.dummy = False
        self.dummy_goal = None
        self.singleview_confirmation: bool = True

        self.multiview_confirmation: bool = True
        self._candidate_object_names: List[str] = []
        self._objects: List[FusedObject] = []
        self._staircases: List[FusedObject] = []
        self._suspected_objects: List[FusedObject] = []
        self._reliable_objects: List[FusedObject] = []
        self._candidate_view_positions = np.zeros((0, 2), dtype=np.float32)
        self.image_history: Dict[int, Dict[str, np.ndarray]] = {}
        self.last_nav_object_id: Optional[int] = None
        self.last_selected_viewpoint_object_id: Optional[int] = None
        self.current_image_id: Optional[int] = None
        self._image_num: int = 0
        self._step: int = 0
        self._num_objects: int = 0
        self._target_class: Optional[str] = None
        self.multifloor = False
        self.dense_view_sampling_enabled = False
        self._obstacle_map = obstacle_map
        self.last_detection_records = []
        self.last_update_diagnostics: Dict[str, Any] = {}
        self.last_confirmation_records: List[Dict[str, Any]] = []
        self._last_confirmation_raw_response: Optional[str] = None

    def make_dummy(self, goal=None):
        self.dummy = True
        self.singleview_confirmation: bool = False
        self.multiview_confirmation: bool = False
        self.dummy_goal = goal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        self._step = 0
        self._image_num = 0
        self._num_objects = 0
        self._objects.clear()
        self._staircases.clear()
        self._suspected_objects.clear()
        self._reliable_objects.clear()
        self._candidate_view_positions = np.zeros((0, 2), dtype=np.float32)
        self.image_history.clear()
        self.last_target_coord = None
        self.last_nav_object_id = None
        self.last_selected_viewpoint_object_id = None
        self.current_image_id = None
        self._confirmation_prompt = self._confirmation_prompt_template
        self.dummy_goal = None
        self.last_detection_records = []
        self.last_update_diagnostics = {}
        self.last_confirmation_records = []
        self._last_confirmation_raw_response = None
        self._latest_scene_cloud = cp.zeros((0, 3), dtype=cp.float32)
        self._last_navigation_filter_records = []

    def remove_object(self, object_id: Optional[int]) -> bool:
        """Remove one fused object candidate without blacklisting its label."""
        if object_id is None:
            return False
        object_id = int(object_id)
        original_count = len(self._objects)
        self._objects[:] = [obj for obj in self._objects if int(obj.id) != object_id]
        self._suspected_objects[:] = [obj for obj in self._suspected_objects if int(obj.id) != object_id]
        self._reliable_objects[:] = [obj for obj in self._reliable_objects if int(obj.id) != object_id]
        self._staircases[:] = [obj for obj in self._staircases if int(obj.id) != object_id]
        if self.last_nav_object_id == object_id:
            self.last_nav_object_id = None
            self.last_target_coord = None
        if self.last_selected_viewpoint_object_id == object_id:
            self.last_selected_viewpoint_object_id = None
        return len(self._objects) < original_count

    def setup_camera_properties(self, hfov: float, min_depth: float, max_depth: float) -> None:
        self.hfov = hfov
        self.min_depth = min_depth
        self.max_depth = max_depth

    def set_candidate_objects(self, candidate_names: Sequence[str], target_name: Optional[str] = None) -> None:
        """Sets (or overwrites) the list of semantic labels to track."""

        # Preserve order but remove duplicates.
        deduped = []
        for name in candidate_names:
            if name not in deduped:
                deduped.append(name)
        self._candidate_object_names = deduped
        if deduped:
            self._target_class = deduped[0]
        tmpl = getattr(self, "_confirmation_prompt_template", _DEFAULT_TEXT_PROMPT["Detection Confirmation"])
        goal_name = (target_name or self._target_class or "").replace("|", " or ")
        self._confirmation_prompt = tmpl.format(
            goal_object=goal_name,
            object_list="|".join(name for name in (self._candidate_object_names[1:])),
        )

    def set_confirmation_client(self, client: Optional[Any]) -> None:
        """Registers a VLM client used for textual confirmation prompts."""

        self._vlm_client = client

    def set_vlm_viz_callback(self, callback: Optional[Any]) -> None:
        self._vlm_viz_callback = callback

    def set_downstair_approval_callbacks(
        self,
        approve_callback: Optional[Callable[[], bool]],
    ) -> None:
        self._approve_downstair_callback = approve_callback

    def set_stair_direction_confirmation_enabled(self, enabled: bool) -> None:
        self._stair_direction_confirmation_enabled = bool(enabled)

    def _push_vlm_viz(self, images: Any, text: str, response: Optional[str] = None) -> None:
        if self._vlm_viz_callback is not None:
            self._vlm_viz_callback(images, text, response)

    def _maybe_promote_pending_downstair(self, staircase_response: Optional[str]) -> None:
        if staircase_response is None:
            return
        if (
            parse_stair_direction_response(staircase_response) == "downstair"
            and self._approve_downstair_callback is not None
        ):
            self._approve_downstair_callback()

    def register_image(
        self,
        rgb: np.ndarray,
        agent2world: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ) -> int:
        """Caches an observation for later multi-view confirmation."""

        self._image_num += 1
        self.image_history[self._image_num] = {
            "image": np.array(rgb, copy=True),
            "agent2world": np.array(agent2world, copy=True),
            "depth": np.array(depth, copy=True) if depth is not None else None,
        }
        self.current_image_id = self._image_num
        return self._image_num

    def get_image(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        record = self.image_history[image_id]
        return record["image"].copy(), record["agent2world"].copy()

    def get_depth(self, image_id: int) -> Optional[np.ndarray]:
        depth = self.image_history[image_id].get("depth")
        return depth.copy() if isinstance(depth, np.ndarray) else None

    def _merge_objs(self, src: FusedObject, dest: FusedObject):
        for label in src.labels.keys():
            self._merge_detection(src.ensure_label(label), dest.ensure_label(label))
        if src.id == self.last_nav_object_id:
            self.last_nav_object_id = dest.id
        self._get_best_matching_label(dest)
        self._update_bbox(dest)
        dest.confirmed = src.confirmed or dest.confirmed
        dest.in_sight = src.in_sight or dest.in_sight
        dest.blacklisted = src.blacklisted and dest.blacklisted
        dest.last_confirm_step = max(src.last_confirm_step, dest.last_confirm_step)
        dest.navigation_fail_count = max(src.navigation_fail_count, dest.navigation_fail_count)
        dest.true_confirmation_count = max(src.true_confirmation_count, dest.true_confirmation_count)
        dest.dense_view_sampling_attempted = (
            src.dense_view_sampling_attempted or dest.dense_view_sampling_attempted
        )
        self._generate_view_positions(dest)
        self._estimate_center(dest.ensure_label(self._target_class))
        image_ids = []
        if src.image_ids:
            image_ids.extend(src.image_ids)
        if dest.image_ids:
            image_ids.extend(dest.image_ids)
        if len(image_ids):
            dest.image_ids = image_ids

    def update_map(
        self,
        object_names: Sequence[str],
        object_masks: Sequence[np.ndarray],
        tf_camera_to_episodic: np.ndarray,
        detection_confidences: Optional[Sequence[float]] = None,
        rgb: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        point_cloud_episodic_frame: Optional[np.ndarray] = None,
        out_of_range_points_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Fuse a batch of detections into the persistent object set."""

        self._step += 1
        self._candidate_view_positions = np.zeros((0, 2), dtype=np.float32)
        self.last_update_diagnostics = {
            "map_step": int(self._step),
            "target_class": self._target_class,
            "object_count_before": int(len(self._objects)),
            "input_detection_count": int(len(object_names)),
            "updates": [],
        }
        self.last_confirmation_records = []
        curr_position = tf_camera_to_episodic[:3, 3].astype(np.float32)
        # Build per-detection point clouds and a sparsified scene cloud for overlap tests.
        detections, scene_cloud = self._extract_object_cloud(
            point_cloud_episodic_frame, object_names, object_masks, detection_confidences, out_of_range_points_mask
        )
        self._latest_scene_cloud = cp.asarray(scene_cloud)

        for name, mask, track in detections:  # detection merging
            # staircases are managed separately
            if name == "staircases":
                l, m = track.pcd[:, -1].min(), track.pcd[:, -1].mean()
                if l > 0.6:
                    continue

                intersection_volumes = []
                for obj in self._staircases:
                    volume = self._intersection_volume(self._get_object_pcd(obj), track.pcd, self.merge_radius)
                    if volume > self._min_intersection_points:
                        obj.in_sight = True
                    intersection_volumes.append(volume)

                intersection_volumes = np.array(intersection_volumes)
                merge_indices = np.where(intersection_volumes > self._min_intersection_points)[0]
                if len(merge_indices):
                    target_obj = self._staircases[merge_indices[0]]
                    merge_indices = merge_indices[-1:1:-1]
                    self._merge_detection(track, target_obj.ensure_label(name))
                else:
                    self._staircases.append(self._create_object(name, track))
                for idx in merge_indices:
                    self._merge_detection(self._staircases[idx].ensure_label(name), target_obj.ensure_label(name))
                    target_obj.confirmed = target_obj.confirmed or self._staircases[idx].confirmed
                    self._staircases.pop(idx)

                continue

            is_valid_detection = self._is_valid_detection(track, mask, curr_position)

            was_detected = False
            intersection_volumes = []
            for obj in self._objects:
                # Use voxel intersection volume to decide whether to merge into an existing object.
                volume = self._intersection_volume(self._get_object_pcd(obj), track.pcd, self.merge_radius)
                if volume > self._min_intersection_points:
                    # If the object has remarkable intersection with a detection, it must be in sight.
                    obj.in_sight = True
                intersection_volumes.append(volume)
            if len(self._objects) > 0:
                best_obj_idx = np.argmax(intersection_volumes)
                was_detected = intersection_volumes[best_obj_idx] > self._min_intersection_points

            max_intersection = (
                int(max(intersection_volumes)) if intersection_volumes else 0
            )
            closest_distance = None
            if not track.is_empty:
                closest_idx = self._get_closest_point_index(track.pcd, curr_position)
                if closest_idx is not None:
                    closest_point = track.pcd[closest_idx]
                    if hasattr(closest_point, "get"):
                        closest_point = closest_point.get()
                    else:
                        closest_point = np.asarray(closest_point)
                    closest_distance = float(
                        np.linalg.norm(closest_point[:2] - curr_position[:2])
                    )
            if track.is_empty:
                validity_reason = "empty_projected_cloud"
            elif track.confidence <= self.suspected_threshold:
                validity_reason = "below_suspected_threshold"
            elif closest_distance is not None and closest_distance <= 1.0:
                validity_reason = "too_close_to_agent"
            else:
                validity_reason = "accepted"
            update_record: Dict[str, Any] = {
                "label": str(name),
                "confidence": float(track.confidence),
                "point_count": int(len(track.pcd)),
                "closest_distance_m": closest_distance,
                "valid": bool(is_valid_detection),
                "validity_reason": validity_reason,
                "max_intersection_points": max_intersection,
                "matched_object_id": None,
                "created_object_id": None,
                "object_count_before": int(len(self._objects)),
            }

            # if the detection confidence is high enough to be suspected, do multi-view confirmation
            if is_valid_detection or name == "staircases":
                if was_detected:
                    best_obj = self._objects[best_obj_idx]
                    before_track = best_obj.ensure_label(name)
                    update_record["matched_object_id"] = int(best_obj.id)
                    update_record["matched_object_confidence_before"] = float(
                        before_track.confidence
                    )
                    # merge the detection track with the best-matching object
                    self._merge_detection(track, before_track)
                    best_obj.in_sight = True
                    best_obj.center = self._estimate_center(best_obj.labels[name])
                    if best_obj.need_more_view_pos and best_obj.best_label != "staircases":
                        self._generate_view_positions(best_obj)
                    update_record["object_best_label_after"] = best_obj.best_label
                    update_record["matched_object_confidence_after"] = float(
                        best_obj.labels[name].confidence
                    )
                else:
                    created = self._create_object(name, track)
                    self._objects.append(created)
                    update_record["created_object_id"] = int(created.id)
                    update_record["object_best_label_after"] = created.best_label
                    update_record["matched_object_confidence_after"] = float(
                        created.labels[name].confidence
                    )
            else:
                update_record["rejection_reason"] = "invalid_detection"
            update_record["object_count_after"] = int(len(self._objects))
            self.last_update_diagnostics["updates"].append(update_record)


        stair_rgb = rgb
        if rgb is not None and depth_image is not None:
            stair_rgb = mask_far_depth_rgb(
                rgb,
                depth_image,
                self.min_depth,
                self.max_depth,
                far_threshold_m=self._stair_vlm_far_threshold_m,
            )

        self._refresh_stair_lists(scene_cloud, stair_rgb)
        self._refresh_object_lists(scene_cloud, curr_position, tf_camera_to_episodic, rgb, self._target_class)

    def has_object(self, target_class: Optional[str] = None) -> bool:
        if self.dummy:
            return self.dummy_goal is not None
        return any(self._is_reliable(obj, self._target_class) for obj in self._reliable_objects)

    def get_best_object(
        self,
        target_class: str,
        curr_position: np.ndarray,
        ignore_threshold: bool = False,
    ) -> Optional[np.ndarray]:
        """Return the closest reliable point for the requested label."""
        if self.dummy:
            return self.dummy_goal

        filtered_objs = self._filter_objects(target_class, ignore_threshold)
        target_cloud = self._get_navigation_label_pcd(filtered_objs, target_class)
        if len(target_cloud) == 0:
            self.last_nav_object_id = None
            return None
        closest_point_index = self._get_closest_point_index(target_cloud, curr_position)
        closest_point_2d = target_cloud[closest_point_index][:2].get()
        if self.last_target_coord is None:
            self.last_target_coord = closest_point_2d
            self.last_nav_object_id = int(target_cloud[closest_point_index, -1])
        else:
            delta_dist = np.linalg.norm(closest_point_2d - self.last_target_coord)
            curr_dist = np.linalg.norm(curr_position[:2] - closest_point_2d)
            if not (delta_dist < 0.1 or (delta_dist < 0.5 and curr_dist > 2.0)):
                self.last_target_coord = closest_point_2d
                self.last_nav_object_id = int(target_cloud[closest_point_index, -1])

        return self.last_target_coord

    def _get_navigation_label_pcd(
        self, objs: List[FusedObject], target_class: str
    ) -> cp.ndarray:
        """Build a target cloud while removing points on the local floor.

        Detection masks can include a strip of floor below an object.  Those
        points are useful for fusion and confirmation, but are poor navigation
        targets after the stop distance was reduced.  Filtering is conservative:
        it is applied only when the estimated floor is close to the object's
        lowest points and enough object points remain; otherwise the original
        cloud is retained.
        """
        self._last_navigation_filter_records = []
        clouds: List[cp.ndarray] = []
        for obj in objs:
            track = obj.labels.get(target_class)
            if track is None or track.is_empty:
                continue
            object_id = int(getattr(obj, "id", -1))
            cloud = cp.asarray(track.pcd)
            object_points = int(len(cloud))
            record: Dict[str, Any] = {
                "object_id": object_id,
                "target_class": str(target_class),
                "original_points": object_points,
                "retained_points": object_points,
                "ground_filter_applied": False,
                "fallback": False,
                "reason": "not_attempted",
            }
            if object_points == 0:
                record["reason"] = "empty_object_cloud"
                self._last_navigation_filter_records.append(record)
                continue

            finite = cp.all(cp.isfinite(cloud[:, :3]), axis=1)
            valid_cloud = cloud[finite]
            if len(valid_cloud) == 0:
                record["reason"] = "no_finite_points"
                self._last_navigation_filter_records.append(record)
                continue

            object_xy = cp.mean(valid_cloud[:, :2], axis=0)
            object_min_z = float(cp.min(valid_cloud[:, 2]).get())
            floor_z: Optional[float] = None
            scene = cp.asarray(self._latest_scene_cloud)
            if scene.ndim == 2 and scene.shape[1] >= 3 and len(scene):
                scene_finite = cp.all(cp.isfinite(scene[:, :3]), axis=1)
                scene_xy = scene[:, :2] - object_xy[None, :]
                near = cp.sum(scene_xy * scene_xy, axis=1) <= 1.0**2
                local = scene[scene_finite & near]
                if len(local) >= 20:
                    floor_z = float(cp.percentile(local[:, 2], 10).get())

            # If the local scene does not expose the floor, the lower edge of
            # the object mask is the safest fallback estimate.  Do not filter
            # elevated objects when the observed floor is substantially below.
            if floor_z is None:
                floor_z = object_min_z
                record["floor_source"] = "object_lower_edge"
            else:
                record["floor_source"] = "local_scene_percentile10"
            record["floor_height_m"] = floor_z
            record["object_min_height_m"] = object_min_z
            floor_gap = object_min_z - floor_z
            record["object_floor_gap_m"] = floor_gap
            if floor_gap > 0.35 or floor_gap < -0.15:
                record["reason"] = "object_not_near_floor"
                self._last_navigation_filter_records.append(record)
                clouds.append(cp.hstack((valid_cloud[:, :3], cp.full((len(valid_cloud), 1), object_id, dtype=cp.float32))))
                continue

            threshold = floor_z + 0.08
            keep = valid_cloud[:, 2] > threshold
            retained = int(cp.count_nonzero(keep).get())
            record["ground_threshold_m"] = threshold
            record["retained_points"] = retained
            # Tiny masks often contain fewer than 20 voxels.  Requiring 20
            # survivors would silently disable the filter for exactly the
            # small targets where a floor strip is most damaging.
            min_retained = max(3, min(20, int(np.ceil(0.10 * object_points))))
            if retained < min_retained:
                record["fallback"] = True
                record["reason"] = "too_few_points_after_ground_filter"
                keep = cp.ones((len(valid_cloud),), dtype=cp.bool_)
            else:
                record["ground_filter_applied"] = True
                record["reason"] = "removed_floor_points"
            filtered = valid_cloud[keep, :3]
            clouds.append(
                cp.hstack((filtered, cp.full((len(filtered), 1), object_id, dtype=cp.float32)))
            )
            self._last_navigation_filter_records.append(record)

        if not clouds:
            return cp.zeros((0, 4), dtype=cp.float32)
        return cp.concatenate(clouds, axis=0)

    def get_target_cloud(
        self,
        target_class: str,
        ignore_threshold: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        filtered_objs = self._filter_objects(target_class, ignore_threshold)
        return self._get_label_pcd(filtered_objs, target_class).get()

    def get_stairs(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        filtered = []
        for obj in self._staircases:
            track = obj.labels.get("staircases")
            if track is None or track.is_empty or obj.blacklisted:
                continue
            # FIXME: this is kinda tricky, maybe we should get rid of it
            if track.confidence >= 0.2 and obj.confirmed:
                filtered.append(obj)
        return self._get_label_pcd(filtered, "staircases").get()

    def get_all_vp(self):
        """Return only the selected object's next viewpoint for map rendering."""
        selected_id = self.last_selected_viewpoint_object_id
        if selected_id is None:
            return np.array([])
        objects = list(self._suspected_objects) + list(self._reliable_objects)
        objects = [obj for obj in objects if int(getattr(obj, "id", -1)) == int(selected_id)]
        view_positions = []
        for obj in objects[:1]:
            if obj.confirmation_count >= 5 or obj.directional_confirmation_count >= 3:
                continue
            labels = getattr(obj, "labels", {})
            track = labels.get(self._target_class) if isinstance(labels, dict) else None
            if track is None or getattr(track, "is_empty", False):
                continue
            vp = getattr(obj, "view_positions", None)
            vv = getattr(obj, "view_pos_valid", None)
            if vp is None or vv is None:
                continue
            vp_np = np.asarray(vp, dtype=np.float32)
            vv_np = np.asarray(vv, dtype=bool)
            if vp_np.ndim != 2 or vv_np.shape[0] != vp_np.shape[0]:
                continue
            valid_indices = np.flatnonzero(vv_np)
            if valid_indices.size == 0:
                continue
            selected = getattr(obj, "last_submitted_view_pos", None)
            if selected is None or selected not in valid_indices:
                selected = int(valid_indices[0])
            view_positions.append(vp_np[int(selected)].reshape(1, 2))

        return np.concatenate(view_positions, axis=0) if view_positions else np.array([])

    def set_selected_viewpoint(self, point: Optional[np.ndarray]) -> None:
        """Track the object owning the selected view, for visualization only."""
        self.last_selected_viewpoint_object_id = None
        if point is None:
            return
        point_arr = np.asarray(point, dtype=np.float32).reshape(-1)
        if point_arr.size < 2:
            return
        objects = list(self._suspected_objects) + list(self._reliable_objects)
        best = None
        best_distance = float("inf")
        for obj in objects:
            vp = np.asarray(getattr(obj, "view_positions", []), dtype=np.float32)
            valid = np.asarray(getattr(obj, "view_pos_valid", []), dtype=bool)
            if vp.ndim != 2 or vp.shape[1] < 2 or valid.shape[0] != vp.shape[0]:
                continue
            indices = np.flatnonzero(valid)
            if indices.size == 0:
                continue
            distances = np.linalg.norm(vp[indices, :2] - point_arr[:2], axis=1)
            local = int(np.argmin(distances))
            if float(distances[local]) < best_distance:
                best_distance = float(distances[local])
                best = obj
        if best is not None and best_distance < 0.35:
            self.last_selected_viewpoint_object_id = int(best.id)

    def get_active_viewpoint_ray(self) -> np.ndarray:
        """Return the selected object's center-to-viewpoint line for rendering."""
        selected_id = self.last_selected_viewpoint_object_id
        if selected_id is None:
            return np.zeros((0, 2), dtype=np.float32)
        objects = list(self._suspected_objects) + list(self._reliable_objects)
        for obj in objects:
            if int(getattr(obj, "id", -1)) != int(selected_id):
                continue
            vp = np.asarray(getattr(obj, "view_positions", []), dtype=np.float32)
            valid = np.asarray(getattr(obj, "view_pos_valid", []), dtype=bool)
            index = getattr(obj, "last_submitted_view_pos", None)
            if (
                obj.center is None
                or vp.ndim != 2
                or vp.shape[1] < 2
                or valid.shape[0] != vp.shape[0]
            ):
                continue
            if index is None or index < 0 or index >= len(vp) or not valid[index]:
                valid_indices = np.flatnonzero(valid)
                if valid_indices.size == 0:
                    continue
                index = int(valid_indices[0])
            return np.asarray([obj.center[:2], vp[int(index), :2]], dtype=np.float32)
        return np.zeros((0, 2), dtype=np.float32)

    def get_candidate_vp(self) -> np.ndarray:
        return self._candidate_view_positions.copy()

    def get_suspected_view_positions(
        self, target_class: Optional[str], curr_position: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect unvisited view positions from suspected objects of the target class."""
        view_pos, centers = [], []
        for obj in self._suspected_objects:
            if target_class and target_class not in obj.labels:
                continue
            if obj.confirmation_count >= 5 or obj.directional_confirmation_count >= 3:
                continue
            unvisited = np.where(obj.view_pos_valid)[0]
            if (
                unvisited.size == 0
                and obj.visited_view_pos_count < 3
                and self.dense_view_sampling_enabled
                and not obj.dense_view_sampling_attempted
            ):
                obj.dense_view_sampling_attempted = True
                self._generate_view_positions(obj)
                unvisited = np.where(obj.view_pos_valid)[0]
            if unvisited.size == 0:
                continue
            if obj.last_submitted_view_pos is None or not obj.view_pos_valid[obj.last_submitted_view_pos]:
                obj.last_submitted_view_pos = int(np.random.choice(unvisited))
            view_idx = obj.last_submitted_view_pos
            view_pos.append(obj.view_positions[view_idx])
            centers.append(obj.center[:2] if obj.center is not None else None)
        if not view_pos:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        centers_np = np.array([[np.nan, np.nan] if c is None else c for c in centers], dtype=np.float32)
        return np.vstack(view_pos), centers_np

    def _filter_objects(
        self, target_class: Optional[str] = None, ignore_threshold: bool = False
    ) -> Iterable[FusedObject]:
        filtered = []
        thresh = 0.9

        candidate_list = self._reliable_objects
        if ignore_threshold and len(candidate_list) == 0:
            candidate_list = self._objects
        if len(candidate_list) == 0:
            return []

        best_conf = 0.0
        best_obj: Optional[FusedObject] = None
        for obj in candidate_list:
            track = obj.labels.get(target_class)
            if track is None or track.is_empty or obj.blacklisted:
                continue
            # FIXME: this is kinda tricky, maybe we should get rid of it
            if candidate_list is self._reliable_objects or track.confidence >= thresh:
                filtered.append(obj)
            elif track.confidence > best_conf:
                best_conf = track.confidence
                best_obj = obj
        if len(filtered) == 0 and best_obj is not None and best_conf > (0.5 if self.multifloor else 0.0):
            filtered.append(best_obj)

        return filtered

    def _get_object_pcd(self, obj: FusedObject) -> cp.ndarray:
        pcd = cp.vstack([d.pcd for d in obj.non_empty_tracks()])
        return cp.asarray(torch.unique(torch.as_tensor(pcd, device=self.device), dim=0))

    def _get_label_pcd(self, objs: List[FusedObject], target_class: str, include_object_id: bool = False) -> cp.ndarray:
        pcds: List[cp.ndarray] = []
        for obj in objs:
            track = obj.labels.get(target_class)
            if include_object_id:
                obj_id_array = cp.full((len(track.pcd), 1), obj.id, dtype=cp.float32)
                pcds.append(cp.hstack((track.pcd, obj_id_array)))
            else:
                pcds.append(track.pcd)
        return cp.concatenate(pcds, axis=0) if pcds else cp.zeros((0, 3), dtype=cp.float32)

    def increase_navigation_fail_count(self, num: int = 1) -> None:
        if self.last_nav_object_id is None:
            return
        for obj in self._objects:
            if obj.id == self.last_nav_object_id:
                obj.navigation_fail_count += num
                if obj.navigation_fail_count > self._navigation_blacklist_threshold:
                    obj.blacklisted = True
                break

    # ------------------------------------------------------------------
    # Confirmation helpers
    # ------------------------------------------------------------------
    def confirm_detection(
        self,
        rgb: np.ndarray,
        text: str,
        viz_images: Optional[Any] = None,
    ) -> Optional[str]:
        if self._vlm_client is None:
            return None
        display_images = rgb if viz_images is None else viz_images
        self._push_vlm_viz(display_images, text)
        response = self._vlm_client._get_response(rgb=rgb, text=text)
        self._push_vlm_viz(display_images, text, str(response))
        self._last_confirmation_raw_response = str(response)

        try:
            return response.split("<answer>")[1].split("</answer>")[0].strip().lower()
        except Exception:
            return None

    def multiview_confirm(self, obj: FusedObject, target_class: str) -> None:
        if not obj.image_ids:
            return
        ids = list(dict.fromkeys(image_id for image_id in obj.image_ids if image_id is not None))
        if not ids:
            return
        images = [self.get_image(image_id)[0] for image_id in ids if image_id in self.image_history]
        if not images:
            return
        mosaic = concat_images(images)
        multiview_prompt = (
            f"{self._confirmation_prompt}\n\n"
            "Important multi-view rule: the input is a mosaic made from multiple separate views. "
            "Evaluate every view independently. Respond True only if the same target object is clearly visible "
            "in every view. Respond False if the target object is clearly absent from one or more views. "
            "Respond Unconfirmed if any view is ambiguous, occluded, or does not provide enough evidence."
        )
        response = self.confirm_detection(mosaic, text=multiview_prompt, viz_images=images)
        obj.directional_confirmation_count += 1

        self._update_confidence_by_response(obj, target_class, response, context="multiview")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_object_cloud(
        self,
        pointcloud_episodic_frame: Optional[np.ndarray],
        object_names: Sequence[str],
        object_masks: Sequence[np.ndarray],
        confidences: Sequence[float],
        out_of_range_points_mask: Optional[np.ndarray],
    ) -> Tuple[List[LabelTrack], np.ndarray]:
        detections: List[Tuple[str, np.ndarray, LabelTrack]] = []
        self.last_detection_records = []
        # Randomly thin the scene cloud to keep later intersection checks cheap.
        sparsed_pointcloud = get_random_subarray(pointcloud_episodic_frame.reshape(-1, 3), self.scene_cloud_size)

        for name, mask, confidence in zip(object_names, object_masks, confidences):
            raw_mask = np.asarray(mask) > 0
            raw_pixels = np.argwhere(raw_mask)
            raw_bbox = None
            if raw_pixels.size:
                raw_bbox = [
                    int(raw_pixels[:, 1].min()),
                    int(raw_pixels[:, 0].min()),
                    int(raw_pixels[:, 1].max()),
                    int(raw_pixels[:, 0].max()),
                ]
            final_mask = (mask > 0).astype(np.uint8)
            final_mask = cv2.erode(final_mask, None, iterations=int(self._erosion_size))
            if self.filter_out_of_range_points:
                final_mask = self._filter_depth_supported_mask(final_mask, out_of_range_points_mask)
            else:
                final_mask = self._filter_depth_supported_mask(final_mask)
            if final_mask is None:
                continue

            object_cloud = get_point_cloud(pointcloud_episodic_frame, final_mask)
            # First cap the cloud size before clustering to avoid DBSCAN OOM.
            if self.pre_dbscan_points:
                object_cloud = get_random_subarray(object_cloud, self.max_detection_points * 2)
            if self.use_dbscan and len(object_cloud) > 0:
                object_cloud = dbscan(object_cloud)
            vol = len(object_cloud)
            final_pixels = np.argwhere(final_mask > 0)
            final_bbox = None
            if final_pixels.size:
                final_bbox = [
                    int(final_pixels[:, 1].min()),
                    int(final_pixels[:, 0].min()),
                    int(final_pixels[:, 1].max()),
                    int(final_pixels[:, 0].max()),
                ]
            record = {
                "label": str(name),
                "confidence": float(confidence),
                "raw_mask_pixels": int(raw_pixels.shape[0]),
                "raw_mask_bbox_xyxy": raw_bbox,
                "projected_mask_pixels": int(final_pixels.shape[0]),
                "projected_mask_bbox_xyxy": final_bbox,
                "projected_cloud_points": int(vol),
            }
            if vol == 0:
                record["status"] = "empty_projected_cloud"
                self.last_detection_records.append(record)
                continue
            object_cloud = get_random_subarray(object_cloud, self.max_detection_points)
            track = LabelTrack(volume=vol, confidence=confidence, pcd=self._voxel_downsample(object_cloud))
            projected = (
                object_cloud.get()
                if hasattr(object_cloud, "get")
                else np.asarray(object_cloud)
            )
            if projected.ndim == 2 and projected.shape[0]:
                record["projected_centroid_xyz"] = np.mean(projected, axis=0).tolist()
                record["projected_bounds_xyz"] = [
                    np.min(projected, axis=0).tolist(),
                    np.max(projected, axis=0).tolist(),
                ]
            record["status"] = "accepted"
            self.last_detection_records.append(record)
            detections.append((name, mask, track))

        return detections, self._voxel_downsample(sparsed_pointcloud)

    def _is_valid_detection(self, detection: LabelTrack, mask: np.ndarray, curr_position: np.ndarray) -> bool:
        if detection.is_empty:
            return False
        if detection.confidence <= self.suspected_threshold:
            return False
        # Ignore detections that are too close to current position to avoid self-occlusion noise.
        index = self._get_closest_point_index(detection.pcd, curr_position)
        closest = detection.pcd[index].get()
        return np.linalg.norm((closest[:2] - curr_position[:2])) > 1.0

    def _merge_detection(self, src: LabelTrack, dest: LabelTrack) -> None:
        # Merge confidences by volume-weighted average, then re-voxelize combined points.
        new_vol = dest.volume + src.volume
        dest.confidence = (dest.volume * dest.confidence + src.volume * src.confidence) / new_vol
        dest.volume = new_vol
        dest.pcd = self._voxel_downsample(cp.concatenate((src.pcd, dest.pcd), axis=0).get())

    def _voxel_downsample(self, points: cp.ndarray) -> cp.ndarray:
        if points.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return (
            cp.asarray(torch.unique(torch.as_tensor((points / self.voxel_size).astype(int)), dim=0)) * self.voxel_size
        )

    def _update_bbox(self, obj: FusedObject) -> None:
        stack = self._get_object_pcd(obj)
        obj.bbox_min, obj.bbox_max = stack.min(axis=0).get(), stack.max(axis=0).get()

    def _bbox_corners(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
        if bbox_min.size == 0 or bbox_max.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        corners = np.array(
            [
                [bbox_min[0], bbox_min[1], bbox_min[2]],
                [bbox_min[0], bbox_min[1], bbox_max[2]],
                [bbox_min[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_max[1], bbox_max[2]],
                [bbox_max[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_min[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_min[2]],
                [bbox_max[0], bbox_max[1], bbox_max[2]],
            ],
            dtype=np.float32,
        )
        return corners

    def _intersection_volume(self, pcd1: cp.array, pcd2: cp.array, radius: float) -> int:
        if pcd1.size == 0 or pcd2.size == 0:
            return 0
        a = torch.unique(torch.as_tensor(pcd1, device=self.device), dim=0)
        b = torch.unique(torch.as_tensor(pcd2, device=self.device), dim=0)
        merged = torch.concatenate([a, b], dim=0)
        uniq = torch.unique(merged, dim=0)
        return len(merged) - len(uniq)

    def _create_object(self, label: str, detection: LabelTrack) -> FusedObject:
        detections = {name: LabelTrack() for name in self._candidate_object_names}
        detections[label] = detection
        self._num_objects += 1
        obj = FusedObject(
            labels=detections,
            id=self._num_objects,
            best_label=label,
            confirmed=self.dummy,
            center=self._estimate_center(detection),
        )
        self._update_bbox(obj)
        if label != "staircases":
            self._generate_view_positions(obj)
        return obj

    def _generate_view_positions(self, obj: FusedObject) -> None:
        # Generate candidate viewpoints around the object
        pts = self._get_object_pcd(obj).get()
        points_tensor = o3d.core.Tensor.from_numpy(pts.astype(np.float32))
        pcd = o3d.t.geometry.PointCloud(self.o3d_device)
        pcd.point.positions = points_tensor
        # Estimate object size from its point cloud bounding box
        pcd_diameter = np.linalg.norm((pcd.get_max_bound() - pcd.get_min_bound()).numpy())

        pcd_radius = pcd_diameter / 2
        center = obj.center
        # Choose a reasonable viewing distance based on object size
        r = max(1, min(pcd_diameter * 2, 3))
        # Start with 8 directions, then densify once if all viewpoints are exhausted
        # before the object has been actively inspected three times.
        num_view_rays = 32 if obj.dense_view_sampling_attempted else 8
        view_angles = np.arange(num_view_rays) * (2 * np.pi / num_view_rays)
        delta = np.stack([r * np.cos(view_angles), r * np.sin(view_angles), np.ones_like(view_angles) * 0.88], axis=-1)
        view_pos = []
        candidate_view_pos = center[:2] + delta[:, :2]
        obstacle_map = getattr(self, "_obstacle_map", None)
        explored_area = getattr(obstacle_map, "explored_area", None)
        if not isinstance(explored_area, np.ndarray):
            explored_area = self.explored_area
        steppable_area = getattr(obstacle_map, "steppable_explored_area", None)
        if not isinstance(steppable_area, np.ndarray):
            steppable_area = self.steppable_explored_area
        observe_navigable_area = getattr(obstacle_map, "_observe_navigable_map", None)
        if not isinstance(observe_navigable_area, np.ndarray):
            observe_navigable_area = steppable_area
        if not isinstance(steppable_area, np.ndarray):
            return
        # March along each ray to find the last navigable point
        partition = int(r / 0.1)
        start_distance = min(max(0.2, pcd_radius + 0.05), r)
        start_step = max(1, int(np.ceil(start_distance / 0.1)))
        if start_step > partition:
            return
        for point in delta:
            sample_distances = np.arange(start_step, partition + 1, dtype=np.float32) * 0.1
            sample_points = center + (sample_distances / r)[:, None] * point
            sample_points_px = self._xy_to_px(sample_points[:, :2])[:, ::-1]
            in_bounds = np.all(sample_points_px >= 0, axis=-1) & np.all(
                sample_points_px < steppable_area.shape,
                axis=-1,
            )
            last_navigable_point = None
            for in_bound, (sample_point, pts) in zip(in_bounds, zip(sample_points, sample_points_px)):
                if not in_bound:
                    break
                # Unknown space blocks a visibility claim. Non-steppable space
                # alone does not: a table or other obstacle may block the robot
                # while still leaving a valid line of sight to the object.
                if not explored_area[pts[0], pts[1]]:
                    break
                sample_distance = float(np.linalg.norm(sample_point[:2] - center[:2]))
                if (
                    not observe_navigable_area[pts[0], pts[1]]
                    and sample_distance > pcd_radius
                ):
                    break
                if steppable_area[pts[0], pts[1]]:
                    last_navigable_point = sample_point
            if last_navigable_point is None:
                continue
            # sample_points already include the object center; avoid adding it twice
            view_pos.append(last_navigable_point[:2].reshape(1, 2))
        # Save candidate view positions and bookkeeping info
        new_view_pos = np.array(view_pos).reshape(-1, 2)
        if candidate_view_pos.size:
            self._candidate_view_positions = np.vstack(
                [
                    self._candidate_view_positions,
                    candidate_view_pos.astype(np.float32, copy=False),
                ]
            )
        old_view_pos_valid = obj.view_pos_valid
        obj.view_pos_valid = np.ones((len(view_pos)), dtype=bool)
        old_last_sub = obj.last_submitted_view_pos
        obj.last_submitted_view_pos = None
        if len(obj.view_positions) and len(new_view_pos):
            dist = np.linalg.norm(obj.view_positions.reshape(-1, 1, 2) - new_view_pos.reshape(1, -1, 2), axis=-1)
            near_idx = np.where(dist <= 0.1)
            obj.view_pos_valid[near_idx[1]] = old_view_pos_valid[near_idx[0]]
            if obj.dense_view_sampling_attempted:
                old_invalid = obj.view_positions[~old_view_pos_valid]
                if len(old_invalid):
                    dist_to_invalid = np.linalg.norm(
                        new_view_pos.reshape(-1, 1, 2) - old_invalid.reshape(1, -1, 2),
                        axis=-1,
                    )
                    obj.view_pos_valid[np.any(dist_to_invalid < 0.3, axis=1)] = False
            if old_last_sub is not None:
                near_last_sub_idx = np.argmin(dist[old_last_sub])
                if dist[old_last_sub, near_last_sub_idx] < 0.5:
                    obj.last_submitted_view_pos = near_last_sub_idx

        obj.view_positions = new_view_pos
        obj.need_more_view_pos = len(view_pos) < 3
        if self.current_image_id is not None:
            obj.image_ids = [self.current_image_id for _ in range(len(view_pos))]

    @staticmethod
    def _estimate_center(detection: LabelTrack) -> Optional[np.ndarray]:
        if detection.is_empty:
            return None
        return cp.mean(detection.pcd, axis=0).get()

    def _update_confidence_by_response(
        self,
        obj: FusedObject,
        target_class: str,
        response: str,
        context: str = "unknown",
    ) -> None:
        """Adjust object confidence based on VLM response."""
        track = obj.labels.get(target_class)
        if track is None or track.is_empty:
            return

        before = {
            "confidence": float(track.confidence),
            "confirmed": bool(obj.confirmed),
            "confirmation_count": int(obj.confirmation_count),
            "true_confirmation_count": int(obj.true_confirmation_count),
            "best_label": obj.best_label,
        }
        obj.last_confirm_step = self._step
        obj.confirmation_count += 1
        if target_class == "staircases":
            normalized = parse_stair_direction_response(response)
            obj.confirmed = normalized == "upstair"
            if normalized == "upstair":
                track.confidence = min(1.0, track.confidence * 1.2)
            elif normalized == "downstair":
                track.confidence *= 0.7
            self.last_confirmation_records.append(
                {
                    "step": int(self._step),
                    "context": str(context),
                    "object_id": int(obj.id),
                    "target_class": str(target_class),
                    "response": response,
                    "raw_response": (self._last_confirmation_raw_response or "")[-2000:],
                    "before": before,
                    "after": {
                        "confidence": float(track.confidence),
                        "confirmed": bool(obj.confirmed),
                        "confirmation_count": int(obj.confirmation_count),
                        "true_confirmation_count": int(obj.true_confirmation_count),
                        "best_label": obj.best_label,
                    },
                }
            )
            return

        obj.confirmed = response in ["true", "outdoor"]
        if response == "true":
            obj.true_confirmation_count += 1
            track.confidence = min(1.0, track.confidence * 1.2)
        elif response == "false":
            track.confidence *= 0.7
        elif response == "outdoor" and track.confidence > 0:
            track.confidence *= -1.0
            obj.blacklisted = True
        self.last_confirmation_records.append(
            {
                "step": int(self._step),
                "context": str(context),
                "object_id": int(obj.id),
                "target_class": str(target_class),
                "response": response,
                "raw_response": (self._last_confirmation_raw_response or "")[-2000:],
                "before": before,
                "after": {
                    "confidence": float(track.confidence),
                    "confirmed": bool(obj.confirmed),
                    "confirmation_count": int(obj.confirmation_count),
                    "true_confirmation_count": int(obj.true_confirmation_count),
                    "best_label": obj.best_label,
                    "blacklisted": bool(obj.blacklisted),
                },
            }
        )

    def _refresh_stair_lists(
        self,
        scene_cloud: np.ndarray,
        rgb: Optional[np.ndarray],
    ) -> None:
        """Update staircase visibility and refresh staircase list
        using the current scene point cloud."""
        staircase_response = None
        for obj in self._staircases:
            if not obj.in_sight:
                # If we are not looking at the object, decide if it should still be considered visible.
                if (
                    self._intersection_volume(self._get_object_pcd(obj), scene_cloud, self.merge_radius)
                    > self._min_intersection_points
                ):
                    obj.in_sight = True

            if obj.in_sight:
                if (
                    staircase_response is None
                    and self._stair_direction_confirmation_enabled
                    and obj.labels["staircases"].confidence > 0.2
                    and (self._step - obj.last_confirm_step >= 3)
                    and obj.confirmation_count < 2
                ):
                    staircase_response = self.confirm_detection(rgb, text=STAIR_DIRECTION_PROMPT)
                if staircase_response is not None:
                    self._update_confidence_by_response(
                        obj,
                        "staircases",
                        staircase_response,
                        context="stair_direction",
                    )
                    self._maybe_promote_pending_downstair(staircase_response)
            obj.in_sight = False

    def _refresh_object_lists(
        self,
        scene_cloud: np.ndarray,
        curr_position: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        rgb: Optional[np.ndarray],
        target_class: Optional[str],
    ) -> None:
        """Update object visibility and refresh suspected/reliable object lists
        using the current scene point cloud, position, and optional VLM
        confirmation."""
        self._suspected_objects.clear()
        self._reliable_objects.clear()
        has_confirmed_obj_in_sight = any(obj.in_sight and obj.confirmed for obj in self._objects)
        response = None
        for obj in self._objects:
            if not obj.in_sight:
                # If we are not looking at the object, decide if it should still be considered visible.
                if (
                    self._intersection_volume(self._get_object_pcd(obj), scene_cloud, self.merge_radius)
                    > self._min_intersection_points
                ):
                    obj.in_sight = True
                    for track in obj.non_empty_tracks():
                        volume = self._intersection_volume(track.pcd, scene_cloud, radius=self.merge_radius)
                        self._merge_detection(LabelTrack(volume=volume), track)  # update without detection, conf=0
                        # TODO: Try update without detection for objects detected.

            if obj.in_sight:
                if (
                    not has_confirmed_obj_in_sight
                    and self.singleview_confirmation
                    and obj.confirmation_count < 5
                    and (self._step - obj.last_confirm_step >= 3)
                    and self._is_suspected(obj, target_class)
                    and response is None  #  confirmation at most once per frame to limit latency.
                ):
                    #  For a suspected object who hasn't been confirmed too many times, ask LLM for single-view confirmation.
                    response = self.confirm_detection(
                        rgb,
                        text=self._confirmation_prompt,
                    )
                if response is not None and self._is_suspected(obj, target_class):
                    self._update_confidence_by_response(
                        obj,
                        target_class,
                        response,
                        context="single_view",
                    )

                obj.best_label = self._get_best_matching_label(obj)
            obj.in_sight = False
            if self._is_suspected(obj, target_class) and not self._is_reliable(obj, target_class):
                self._suspected_objects.append(obj)
            self._mark_view_visited(obj, curr_position, tf_camera_to_episodic)
        for obj in self._objects:
            if self._is_reliable(obj, target_class):
                self._reliable_objects.append(obj)

    def _mark_view_visited(
        self, obj: FusedObject, curr_position: np.ndarray, tf_camera_to_episodic: np.ndarray
    ) -> None:
        """Check whether the current pose corresponds to the last submitted view of this object.
        If the robot is close enough and roughly facing the object from that view, mark the view as visited.
        Also record the image id for later multi-view confirmation. It also invalidates the unreachable view poses."""
        idx = obj.last_submitted_view_pos
        if idx is None or obj.view_positions.size == 0:
            return
        vp_px = self._xy_to_px(obj.view_positions)
        obj.view_pos_valid &= self.steppable_explored_area[vp_px[:, 1], vp_px[:, 0]]
        if not obj.view_pos_valid[idx]:
            return
        view_pos = obj.view_positions[idx]
        dist = np.linalg.norm(view_pos - curr_position[:2])
        _, theta = rho_theta(curr_position[:2], extract_yaw(tf_camera_to_episodic), obj.center[:2])
        if dist <= 0.6 and abs(theta) <= np.pi / 12:
            obj.visited_view_pos_count += 1
            if obj.visited_view_pos_count == 3:
                obj.view_pos_valid.fill(0)
            obj.view_pos_valid[idx] = False
            obj.image_ids[idx] = self.current_image_id
            if self.multiview_confirmation and obj.confirmation_count < 5:
                self.multiview_confirm(obj, self._target_class)

    def mark_view_inaccessible(self, view_point: np.ndarray):
        for obj in self._suspected_objects:
            if np.linalg.norm(obj.view_positions[obj.last_submitted_view_pos] - view_point) < 0.5:
                obj.view_pos_valid[obj.last_submitted_view_pos] = False

    def _get_best_matching_label(self, obj: FusedObject) -> str:
        best_label = ""
        best_score = 0.0
        for label, track in obj.labels.items():
            if track.is_empty:
                continue
            score = track.volume * track.confidence
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    def _is_reliable(self, obj: FusedObject, target_class: str) -> bool:
        track = obj.labels.get(target_class)
        if track is None or track.is_empty:
            return False
        confidence_is_reliable = track.confidence > self.reliable_threshold
        true_confirmation_override = (
            self.dense_view_sampling_enabled and obj.true_confirmation_count >= 3
        )
        return (
            obj.best_label == target_class
            and obj.confirmed
            and (confidence_is_reliable or true_confirmation_override)
        )

    def _is_suspected(self, obj: FusedObject, target_class: str) -> bool:
        track = obj.labels.get(target_class)
        if track is None or obj.blacklisted:
            return False
        return track.confidence > self.suspected_threshold

    def _get_closest_point_index(
        self, cloud: cp.ndarray, curr_position: np.ndarray
    ) -> Tuple[np.ndarray, Optional[int]]:
        if len(cloud) == 0:
            return None
        ndim = curr_position.shape[0]
        distances = cp.linalg.norm(cloud[:, :ndim] - cp.asarray(curr_position).reshape(1, ndim), axis=1)
        index = cp.argmin(distances)
        return index.get()

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """Converts an array of (x, y) coordinates to pixel coordinates.

        Args:
            points: The array of (x, y) coordinates to convert.

        Returns:
            The array of (x, y) pixel coordinates.
        """
        px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        px[:, 0] = self.steppable_explored_area.shape[0] - px[:, 0]
        return px.astype(int)

    def _px_to_xy(self, px: np.ndarray) -> np.ndarray:
        """Converts an array of pixel coordinates to (x, y) coordinates.

        Args:
            px: The array of pixel coordinates to convert.

        Returns:
            The array of (x, y) coordinates.
        """
        px_copy = px.copy()
        px_copy[:, 0] = self.steppable_explored_area.shape[0] - px_copy[:, 0]
        points = (px_copy - self._episode_pixel_origin) / self.pixels_per_meter
        return points[:, ::-1]
