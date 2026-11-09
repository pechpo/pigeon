from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import os
import random

import numpy as np

from .random_frontier_planner import RandomFrontierPlanner
from ..models.coco_classes import COCO_CLASSES
from ..mapping.object_pointcloud_map import ObjectPointCloudMap
from ..utils.shared_memory import ShmArray, attach_ndarray
from .gpu_client import GPUWorkerClient
from ..utils.geometry_utils import rho_theta, xyz_yaw_pitch_to_tf_matrix

COCO_CLASS_SET = {c.lower() for c in COCO_CLASSES}


def calc_box_area(box):
    sizes = box[2:] - box[:2]
    return sizes[0] * sizes[1]


def xyxy2cs(xyxy):
    return ((xyxy[2:] + xyxy[:2]) / 2).astype(int), ((xyxy[2:] - xyxy[:2]) / 2).astype(int)


class RandomObjectPlanner(RandomFrontierPlanner):
    """Random frontier exploration with object search and point-goal pursuit.

    Phases:
    - init: spin in place for 12 steps.
    - explore: randomly choose frontiers to explore.
    """

    actions = ("move_forward", "turn_left", "turn_right", "stop", "look_up", "look_down")
    models = ["pointnav_rnn", "yolov7", "mobile_sam"]

    # Simple alias so subclasses can do:
    #   self._downstair_detection = False
    # to turn off downstairs protection without touching the map directly.
    @property
    def _downstair_detection(self) -> bool:
        return bool(getattr(self._omap, "_staircase_protection", True))

    @_downstair_detection.setter
    def _downstair_detection(self, enabled: bool) -> None:
        if hasattr(self, "_omap"):
            self._omap._staircase_protection = bool(enabled)

    def __init__(self) -> None:
        super().__init__()
        # Override/extend state for object-aware behavior
        self._phase = "init"
        self._target_class: Optional[str] = None
        self._target_object_name: Optional[str] = None
        self._goal_proximity_thresh: float = 0.90
        self._same_goal_thresh: float = 0.25
        self._passby_thresh: float = 2.0  # meters to pass by frontier goals near previous goal
        self._objmap = ObjectPointCloudMap(erosion_size=5)  # 5 for vlfm
        # Shared-memory blocks for model I/O
        self._rgb_slot = ShmArray()
        # When enabled, include "staircases" in YOLO filter classes and
        # project detected stairs into the obstacle map as obstacles.
        self._upstair_detection = True
        self.mp3d = False
        self._last_detector_output = None
        self._last_detection_filter = None

    def _update_detector_routing(self) -> None:
        """Route each semantic class to YOLO or OWL purely by COCO membership."""
        has_owl = "owl_v2" in self.models
        self._yolo_classes: List[str] = []
        self._owl_classes: List[str] = []
        classes = getattr(self, "_classes", [])
        for cls in classes:
            name = str(cls).strip().lower()
            if not has_owl:
                # No OWL model available; route everything to YOLO.
                self._yolo_classes.append(cls)
            elif name in COCO_CLASS_SET:
                # COCO classes go to YOLO.
                self._yolo_classes.append(cls)
            elif self.mp3d or name == "staircases":
                # Non-COCO classes go to OWLv2.
                self._owl_classes.append(cls)
            else:
                # For HM3D-OVON
                self._owl_classes.append(cls)

    def _target_uses_owl(self) -> bool:
        """Return True iff the current target is routed to OWLv2."""
        target = getattr(self, "_target_class", None)
        if not isinstance(target, str):
            return False
        owl_classes = getattr(self, "_owl_classes", [])
        return target in owl_classes

    def set_yolo_conf_thresh(self) -> None:
        # Use a lower threshold when the target is handled by OWLv2.
        target_conf = 0.10 if self._target_uses_owl() else 0.50
        self._yolo_conf_thresh = [target_conf]  # target object
        if self._upstair_detection:
            self._yolo_conf_thresh.append(0.30)  # stairs

    def _semantic_stair_processing_enabled(self) -> bool:
        return (
            bool(getattr(self, "multifloor", False))
            and bool(getattr(self, "_upstair_detection", False))
            and getattr(self, "_phase", None)
            not in {
                "pointnav_descent",
                "post_stair_settle",
            }
        )

    def _filter_semantic_stair_classes(self, classes: List[str]) -> List[str]:
        filtered = list(classes)
        if self._semantic_stair_processing_enabled():
            return filtered
        return [cls for cls in filtered if str(cls).strip().lower() != "staircases"]

    # ------------------------------ Public API ------------------------------
    def _vlm_target_name(self) -> str:
        target_name = getattr(self, "_target_object_name", None)
        if isinstance(target_name, str) and target_name.strip():
            return target_name.replace("|", " or ")
        target_class = getattr(self, "_target_class", None)
        if isinstance(target_class, str) and target_class.strip():
            return target_class.replace("|", " or ")
        return "the target"

    def set_target_object(self, name: str) -> None:
        self._target_object_name = name
        # for object map without fusion, it only needs two classes: target and stairs
        normalized = {
            "sofa": "couch",
            "plant": "potted plant",
            "tv_monitor": "tv",
            # for HM3D-OVON
            "plant": "potted plant",
            "freezer": "refrigerator",
            "flower vase": "vase",
            "flowerpot": "potted plant",
            "decorative plant": "potted plant",
            "bedframe": "bed",

        }.get(name, name)
        self._target_class = normalized
        self._classes = [self._target_class]
        if self._upstair_detection:
            self._classes.append("staircases")
        # Decide routing for each class purely by COCO membership.
        self._update_detector_routing()
        self._cls_index = {c: i for i, c in enumerate(self._classes)}
        self.set_yolo_conf_thresh()

    def reset(self, reset_stairs=True) -> None:
        super().reset(reset_stairs)
        self._phase = "init"
        self._objmap.reset()
        self._objmap.set_gpu_worker(self._gpu)
        self._last_detector_output = None
        self._last_detection_filter = None

    def _record_target_stop(self, det_seg: Optional[Dict[str, Any]], has_obj: bool) -> None:
        """Record the evidence used by the target-stop shortcut."""
        objmap = getattr(self, "_objmap", None)
        object_id = getattr(objmap, "last_nav_object_id", None)
        selected = None
        if objmap is not None and object_id is not None:
            selected = next(
                (obj for obj in getattr(objmap, "_objects", []) if getattr(obj, "id", None) == object_id),
                None,
            )
        track = getattr(selected, "labels", {}).get(self._target_class) if selected is not None else None
        current_classes = [
            str(name)
            for name in (det_seg or {}).get("classes", [])
            if name is not None
        ]
        self._record_trace_event(
            "target_stop_decision",
            target_class=self._target_class,
            object_id=int(object_id) if object_id is not None else None,
            has_reliable_object=bool(has_obj),
            current_frame_classes=current_classes,
            current_frame_has_target=bool(self._target_class in current_classes),
            distance_to_goal=(
                float(np.linalg.norm(np.asarray(self._curr_xy) - np.asarray(self._goal_xy)))
                if self._curr_xy is not None and self._goal_xy is not None
                else None
            ),
            threshold=float(self._goal_proximity_thresh),
            object_best_label=getattr(selected, "best_label", None),
            object_confirmed=bool(getattr(selected, "confirmed", False)) if selected is not None else None,
            confirmation_count=int(getattr(selected, "confirmation_count", 0)) if selected is not None else None,
            true_confirmation_count=(
                int(getattr(selected, "true_confirmation_count", 0)) if selected is not None else None
            ),
            target_confidence=float(getattr(track, "confidence", 0.0)) if track is not None else None,
        )

    # ------------------------------ Core Logic ------------------------------

    def _run_detector(self, rgb: np.ndarray) -> Dict[str, Any]:
        yolo_classes = self._filter_semantic_stair_classes(getattr(self, "_yolo_classes", self._classes))
        owl_classes = self._filter_semantic_stair_classes(getattr(self, "_owl_classes", []))
        combined: Dict[str, Any] = {"found": False, "bboxes": [], "scores": [], "classes": []}

        # Run YOLO first if requested
        if yolo_classes:
            det_yolo = self._gpu.yolov7(rgb, filter_class=yolo_classes)
            if det_yolo.get("found", False):
                combined["bboxes"].extend(det_yolo.get("bboxes") or [])
                combined["scores"].extend(det_yolo.get("scores") or [])
                combined["classes"].extend(det_yolo.get("classes") or [])

        # Run OWLv2 for non-COCO targets when enabled
        if owl_classes:
            det_owl = self._gpu.owl_v2(rgb, filter_class=owl_classes)
            if det_owl.get("found", False):
                bbox = []
                score = []
                residual = defaultdict(list)
                for b, s, c in zip(det_owl.get("bboxes", []), det_owl.get("scores", []), det_owl.get("classes", [])):
                    if c == "staircases":
                        bbox.append(np.array(b))
                        score.append(s)
                    else:
                        residual["bboxes"].append(b)
                        residual["scores"].append(s)
                        residual["classes"].append(c)
                if len(bbox):
                    areas = [calc_box_area(box) for box in bbox]
                    idx = np.argmax(np.array(areas))
                    b = bbox[idx]
                    c, s = xyxy2cs(b)
                    for _ in range(len(bbox)):
                        for box in bbox:
                            center, sizes = xyxy2cs(box)
                            if np.all(np.abs(c - center) < s + sizes):
                                b[:2] = np.min(np.stack([b[:2], box[:2]]), axis=0)
                                b[2:] = np.max(np.stack([b[2:], box[2:]]), axis=0)
                                c, s = xyxy2cs(b)

                    combined["bboxes"].append(b.tolist())
                    combined["scores"].append(float(np.mean(score)))
                    combined["classes"].append("staircases")

                combined["bboxes"].extend(residual.get("bboxes") or [])
                combined["scores"].extend(residual.get("scores") or [])
                combined["classes"].extend(residual.get("classes") or [])

        combined["found"] = len(combined["bboxes"]) > 0
        return combined

    def _detect_and_update(self, rgb: np.ndarray, depth: np.ndarray) -> Optional[Dict[str, Any]]:
        # Run selected detector each step
        det = self._run_detector(rgb)
        raw_bboxes = list(det.get("bboxes") or [])
        raw_scores = list(det.get("scores") or [])
        raw_classes = list(det.get("classes") or [])
        self._last_detector_output = {
            "found": bool(det.get("found", False)),
            "detections": [
                {
                    "class": raw_classes[i] if i < len(raw_classes) else None,
                    "score": raw_scores[i] if i < len(raw_scores) else None,
                    "bbox_normalized": raw_bboxes[i] if i < len(raw_bboxes) else None,
                }
                for i in range(max(len(raw_bboxes), len(raw_scores), len(raw_classes)))
            ],
        }
        filter_records = []
        if not bool(det.get("found", False)):
            self._last_detection_filter = {
                "reason": "detector_no_detection",
                "detections": filter_records,
            }
            return None

        bboxes_norm = det.get("bboxes") or []
        scores = det.get("scores") or []
        classes = det.get("classes") or []
        if len(bboxes_norm) == 0 or len(scores) == 0:
            self._last_detection_filter = {
                "reason": "detector_empty_output",
                "detections": filter_records,
            }
            return None

        # Per-class confidence filtering using self._yolo_conf_thresh vector
        kept_indices = []
        for i, (score, name) in enumerate(zip(scores, classes)):
            j = self._cls_index.get(name)
            thr = self._yolo_conf_thresh[j]
            record = {
                "index": int(i),
                "class": name,
                "score": float(score),
                "threshold": float(thr),
                "passed_threshold": bool(float(score) >= float(thr)),
                "bbox_normalized": bboxes_norm[i],
                "bbox_px": None,
                "sam_success": False,
            }
            filter_records.append(record)
            if float(score) >= float(thr):
                kept_indices.append(i)
        if not kept_indices:
            self._last_detection_filter = {
                "reason": "confidence_threshold",
                "detections": filter_records,
            }
            return None

        h, w = rgb.shape[:2]
        bboxes_px = []
        kept_scores = []
        kept_classes = []
        masks = []

        for i in kept_indices:
            x1, y1, x2, y2 = bboxes_norm[i]
            bx1 = int(max(0, min(w - 1, round(float(x1) * w))))
            by1 = int(max(0, min(h - 1, round(float(y1) * h))))
            bx2 = int(max(0, min(w - 1, round(float(x2) * w))))
            by2 = int(max(0, min(h - 1, round(float(y2) * h))))
            record = filter_records[i]
            record["bbox_px"] = [bx1, by1, bx2, by2]
            if bx2 <= bx1 or by2 <= by1:
                record["reason"] = "invalid_pixel_bbox"
                continue
            m = self._gpu.mobile_sam(rgb, (bx1, by1, bx2, by2))
            if m is None:
                record["reason"] = "sam_failed"
                continue
            record["sam_success"] = True
            m_use = m if m.ndim == 2 else m[..., 0]
            bboxes_px.append((bx1, by1, bx2, by2))
            kept_scores.append(float(scores[i]))
            kept_classes.append(classes[i])
            masks.append(m_use.astype(np.uint8, copy=False))

        if not bboxes_px:
            self._last_detection_filter = {
                "reason": "all_segmentation_failed",
                "detections": filter_records,
            }
            return None

        # Build visualization record: pure multi-object fields
        det_seg: Dict[str, Any] = {
            "bboxes_px": [list(map(int, b)) for b in bboxes_px],
            "scores": kept_scores,
            "classes": kept_classes,
            "masks": masks,
        }
        self._last_detection_filter = {
            "reason": "accepted",
            "detections": filter_records,
        }

        # Update object and stairs maps with each detection (no branching on target_class)
        tf = self._tf_camera_to_episodic
        d_for_fx = depth[..., 0] if depth.ndim == 3 else depth
        _, w_fx = d_for_fx.shape[:2]
        fx = fy = float(w_fx) / (2.0 * np.tan(float(self._camera_fov_rad) / 2.0))
        d = depth
        if d.ndim == 3:
            d = d[..., 0]
        d_f32 = d.astype(np.float32, copy=False)

        for cls_name, m in zip(kept_classes, masks):
            obj_mask = (m.astype(np.uint8) > 0).astype(np.uint8)
            label = str(cls_name).strip().lower()
            if label == "staircases" and not self._semantic_stair_processing_enabled():
                continue
            self._objmap.update_map(
                label,
                d_f32,
                obj_mask,
                tf,
                float(self._min_depth),
                float(self._max_depth),
                fx,
                fy,
            )

        # Project accumulated stairs cloud onto obstacle map as obstacles
        if self._semantic_stair_processing_enabled() and self._objmap.has_object("staircases", True):
            stairs_cloud = self._objmap.get_target_cloud("staircases", True)
            if stairs_cloud.size > 0:
                if hasattr(self._omap, "mark_semantic_obstacles"):
                    self._omap.mark_semantic_obstacles(stairs_cloud, kind="upstairs")

        # If we now have a target, transition to target phase and set goal
        if self._target_class and self._objmap.has_object(self._target_class):
            tgt_xy = self._objmap.get_best_object(self._target_class, self._curr_xy)
            if tgt_xy is not None and len(tgt_xy) == 2:
                goal_xy = np.asarray(tgt_xy, dtype=np.float32)
                # If the nearest object point lies in a banned region,
                # treat it as unreachable and do not enter target phase.
                if self.is_banned(goal_xy):
                    return det_seg
                if hasattr(self._omap, "set_selected_frontier"):
                    self._omap.set_selected_frontier(goal_xy)
                if self._goal_xy is None:
                    self._goal_xy = goal_xy
                elif self._distance(goal_xy, self._goal_xy) > self._same_goal_thresh:
                    self._should_reset_pointnav = True
                    self._goal_xy = goal_xy
                self._phase = "target"

        return det_seg

    def _choose_goal(self, passby: bool = False) -> None:
        frontiers = self._frontiers.copy()
        if passby and self._goal_xy is not None and frontiers.size >= 2:
            # Remove frontiers too far from previous goal
            dists = np.linalg.norm(frontiers - np.asarray(self._curr_xy, dtype=np.float32), axis=1)
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] < self._passby_thresh:
                self._goal_xy = frontiers[nearest_idx].astype(np.float32)
        if isinstance(frontiers, np.ndarray) and frontiers.size >= 2:
            idx = random.randrange(frontiers.shape[0])
            self._goal_xy = frontiers[idx].astype(np.float32)
        else:
            self._goal_xy = None
        # Update visualization highlight for the selected frontier
        if hasattr(self._omap, "set_selected_frontier"):
            self._omap.set_selected_frontier(self._goal_xy)

    def _maybe_align_view(self, det_seg: Optional[Dict[str, Any]], has_obj: bool) -> Optional[Dict[str, Any]]:
        return None

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._start_layer_step()
        rgb = obs.get("rgb")
        depth = obs.get("depth")
        gps = obs.get("gps")
        yaw = obs.get("yaw")

        # Precompute agent pose and transform once per step.
        self._set_agent_pose(gps, yaw)

        # Update obstacle map/frontiers using the precomputed transform
        self._update_mapping(depth, rgb)

        # Prune object map using explored area each step (only needs transform)
        tf = self._tf_camera_to_episodic
        if tf is None:
            cam_pos = np.array([self._curr_xy[0], self._curr_xy[1], self._camera_height], dtype=np.float32)
            tf = xyz_yaw_pitch_to_tf_matrix(cam_pos, float(self._curr_yaw), float(self._camera_pitch_rad))
        self._objmap.update_explored(tf, float(self._max_depth), float(self._camera_fov_rad))

        # If the current goal becomes an obstacle (e.g., due to staircase
        # protection), drop and ban it so we replan instead of walking into it.
        if self._goal_xy is not None and self._goal_is_occupied():
            self._ban_point(self._goal_xy)
            self._goal_xy = None
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)
            self._should_reset_pointnav = True

        # Always attempt detection each step
        det_seg = self._detect_and_update(rgb, depth)
        self._refresh_detection_viz(det_seg=det_seg)
        has_obj = bool(self._target_class) and self._objmap.has_object(self._target_class)

        # Phase: initial spin (unless we've entered target already)
        if self._phase == "init" and self._init_steps < self._max_init_steps and self._phase != "target":
            self._init_steps += 1
            out = {
                "action": "turn_left",
                "map_viz": self._visualize_map(),
                "det_seg": det_seg,
                "nav_goal": self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None,
                "stop_called": False,
                "target_detected": has_obj,
            }
            if self._init_steps >= self._max_init_steps:
                self._phase = "explore"
            return out

        # Target phase: pursue object nav point if still present
        if self._phase == "target":
            if has_obj:
                # Refresh goal from map and step pointnav
                tgt_xy = self._objmap.get_best_object(self._target_class, self._curr_xy)
                if tgt_xy is not None and len(tgt_xy) == 2:
                    tgt_xy_arr = np.asarray(tgt_xy, dtype=np.float32)
                    # If the target location is banned, fallback to not-has-obj behavior
                    if self.is_banned(tgt_xy_arr):
                        has_obj = False
                    else:
                        self._goal_xy = tgt_xy_arr
                        # If near the object goal, stop
                        if self._near_goal(self._goal_proximity_thresh):
                            self._record_target_stop(det_seg, has_obj)
                            return {
                                "action": "stop",
                                "map_viz": self._visualize_map(),
                                "det_seg": det_seg,
                                "nav_goal": self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None,
                                "stop_called": True,
                                "target_detected": has_obj,
                            }
                        # Otherwise, step pointnav towards the goal
                        return self._pointnav_step(
                            depth,
                            self._goal_xy,
                            extra={
                                "det_seg": det_seg,
                                "target_detected": has_obj,
                            },
                        )
        if not has_obj:  # Lost the target; return to exploration
            self._phase = "explore"
            self._goal_xy = None
            self._omap.set_selected_frontier(None)
            self._should_reset_pointnav = True

        stair_out = self._maybe_handle_stairs_or_escape(
            depth,
            extra={
                "det_seg": det_seg,
                "target_detected": has_obj,
            },
        )
        if stair_out is not None:
            return stair_out

        align_action = self._maybe_align_view(det_seg, has_obj)
        if align_action is not None:
            return align_action

        if self._goal_xy is None:
            # Pick a random frontier goal if available
            self._choose_goal_with_trace()
        else:
            if self._near_goal(self._proximity_threshold):  # change the goal if arrive at the current one
                goal_history = self._goal_xy.copy()
                self._choose_goal_with_trace(passby=True)
                if self._goal_xy is not None:
                    self._should_reset_pointnav = self._distance(goal_history, self._goal_xy) > self._same_goal_thresh

        if self._goal_xy is None:
            return {
                "action": "stop",
                "map_viz": self._visualize_map(),
                "det_seg": det_seg,
                "nav_goal": None,
                "stop_called": True,
                "target_detected": has_obj,
            }

        # Step pointnav towards the exploration frontier goal
        return self._pointnav_step(
            depth,
            self._goal_xy,
            extra={
                "det_seg": det_seg,
                "target_detected": has_obj,
            },
        )
