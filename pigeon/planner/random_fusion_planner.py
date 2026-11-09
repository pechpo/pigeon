import copy
import os
from typing import Any, Dict, Optional, List, Tuple
from collections import defaultdict

import numpy as np

from .random_object_planner import RandomObjectPlanner
from ..mapping.fusion_pointcloud_map import FusionPointCloudMap
from ..models.vlm import VLMClient
from ..utils.geometry_utils import rho_theta
from ..utils.geometry_utils_gpu import dbscan


def calc_box_area(box):
    sizes = box[2:] - box[:2]
    return sizes[0] * sizes[1]


def xyxy2cs(xyxy):
    return ((xyxy[2:] + xyxy[:2]) / 2), ((xyxy[2:] - xyxy[:2]) / 2)


class RandomFusionPlanner(RandomObjectPlanner):
    models = ["pointnav_rnn", "yolov7", "mobile_sam"]
    # Prompt used by the VLM to propose confusing/related classes and a confidence prior
    candidate_prompt_template = [
        {
            "role": "system",
            "content": (
                """You are an intelligent agent that helps the object detector identify detected objects. 
Based on the labels I provide, you'll give the labels that may cause misdetection, potential room objects, and the confidence threshold.
At each step, I will give you a label. You should considerthe following:
(1) For misdetection, provide objects that may be misdetected by detectors such as YOLOv7, considering similarities in shape, color, size, etc. The labels should be in COCO classes, with at least 3 and at most 5 labels.
(2) For the confidence score, the confidence score should be in [0.25, 0.65], as the smallest is an apple in a home (0.25) and the largest is a bed in a home (0.65).
Please follow the format like this, without bracket --
<thought> [put your step-by-step thinking process here] </thought>
<answer> [put a list of objects and a confidence score here] </answer>
"""
            ),
        },
        {"role": "user", "content": "cake"},
        {
            "role": "assistant",
            "content": (
                """
<thought> In the COCO classes, confidence threshold for cake: A cake is typically round or square and can have decorative icing or layers. It is quiet smaller than a bed (0.65), but bigger than an apple (0.25), so the confidence threshold should be low, around 0.30, which is in [0.25, 0.65].
Misdetection possibilities:
donut: A donut, especially a large one or a stack of donuts, can resemble a cake because of its circular shape and icing-like decorations.
pizza: A whole pizza, particularly one with thick crust or layered toppings, might resemble a cake in certain contexts.
sandwich: A stacked sandwich, especially one with visible layers, could look similar to a layered cake.
pie: A pie can be misdetected as a cake due to its circular shape and pastry-like crust, which is similar to a cake's appearance.</thought>
<answer> [donut, pizza, sandwich, pie, 0.30] </answer>
"""
            ),
        },
        {"role": "user", "content": "cabinet"},
        {
            "role": "assistant",
            "content": (
                """
<thought> In the COCO classes, confidence threshold for cabinet: A cabinet is typically rectangular, with flat surfaces, often in the form of a box or with shelves inside. It's smaller than larger furniture pieces like beds or dining tables (0.60) but still sizable compared to smaller household objects. The confidence threshold for a cabinet should be around 0.50, which is in [0.25, 0.65].
Misdetection possibilities:
bookshelf: A bookshelf can be similar in shape and size to a cabinet, with a flat surface and storage space, which may lead to confusion.
dresser: A dresser, especially one with drawers, could resemble a cabinet because of the similar rectangular shape and flat surfaces.
closet: A closet, particularly a small one, can resemble a cabinet in terms of its rectangular form and purpose for storage. </thought>
<answer> [bookshelf, dresser, closet, 0.50] </answer>
"""
            ),
        },
        {"role": "user", "content": "{goal_object}"},
    ]

    def __init__(self) -> None:
        super().__init__()
        self._objmap = FusionPointCloudMap(
            erosion_size=5,
            explored_area=self._omap.explored_area,
            steppable_explored_area=self._omap.steppable_explored_area,
            episode_pixel_origin=self._omap._episode_pixel_origin,
            pixels_per_meter=self._omap.pixels_per_meter,
            obstacle_map=self._omap,
        )
        # Optional VLM augments the target class list with likely confusions
        self._vlm: Optional[VLMClient] = VLMClient(os.environ.get("VLM_MODEL", "gpt-4o"))
        self._objmap.set_confirmation_client(self._vlm)
        self._bind_downstair_approval()
        self.candidate_prompt = None
        self._goal_view_facing: Optional[np.ndarray] = None
        self._last_detector_routing = None
        self._last_category_setup = None

    def _approve_pending_downstair(self) -> bool:
        if not hasattr(self._omap, "approve_pending_downstair"):
            return False
        if not self._omap.approve_pending_downstair():
            return False
        self._refresh_filtered_frontiers()
        return True

    def _bind_downstair_approval(self) -> None:
        objmap = getattr(self, "_objmap", None)
        if objmap is None or not hasattr(objmap, "set_downstair_approval_callbacks"):
            return
        objmap.set_downstair_approval_callbacks(self._approve_pending_downstair)

    def set_target_object(self, name: str) -> None:
        super().set_target_object(name)
        self._setup_object_map()

    def _setup_object_map(self) -> None:
        # setting runtime properties
        self._objmap.setup_camera_properties(self._camera_fov_rad, self._min_depth, self._max_depth)
        target = self._target_class
        vlm_target = self._vlm_target_name()
        target_is_stair = str(target).strip().lower() in {"stair", "stairs", "staircase", "staircases"}
        classes = [] if target_is_stair else [target]
        category_response = ""
        self.candidate_prompt = copy.deepcopy(self.candidate_prompt_template)
        self.candidate_prompt[-1]["content"] = self.candidate_prompt[-1]["content"].format(goal_object=vlm_target)
        if self._vlm is not None and target and not target_is_stair:
            resp = self._vlm._get_response_text(self.candidate_prompt, failed_response="")
            category_response = str(resp)
            extra, reliable_thr = self._parse_category_response(resp)
            for cls in extra:
                if cls and cls not in classes:
                    classes.append(cls)
        else:
            extra, reliable_thr = [], None
        self._last_category_setup = {
            "target_class": target,
            "target_name": vlm_target,
            "candidate_names": list(classes),
            "vlm_extra_names": list(extra),
            "vlm_recommended_reliable_threshold": (
                float(reliable_thr) if reliable_thr is not None else None
            ),
            "applied_reliable_threshold": float(self._objmap.reliable_threshold),
            "target_confidence_threshold": 0.30,
            "stair_confidence_threshold": 0.15 if self._upstair_detection else None,
        }
        self._record_trace_event(
            "category_setup",
            target_class=target,
            candidate_names=list(classes),
            recommended_reliable_threshold=(
                float(reliable_thr) if reliable_thr is not None else None
            ),
            applied_reliable_threshold=float(self._objmap.reliable_threshold),
            response=category_response[-2000:],
        )
        if target_is_stair:
            self._classes = []
        elif self.mp3d:
            self._classes = classes[:1]
        if getattr(self, "_upstair_detection", False):
            self._classes.append("staircases")
        self._objmap.set_candidate_objects(classes or ["staircases"], target_name=vlm_target)
        self._cls_index = {c: i for i, c in enumerate(self._classes)}
        target_conf = 0.30
        self._yolo_conf_thresh = [target_conf] * len(self._classes)  # target object
        if self._upstair_detection:
            self._yolo_conf_thresh[-1] = 0.15  # stairs
        # Route each candidate (and optional stairs) to YOLO or OWL
        # purely by COCO membership and available models.
        self._update_detector_routing()

    def _parse_category_response(self, resp: str) -> Tuple[List[str], Optional[float]]:
        # Extract confusing classes and the suggested reliability threshold from VLM output.
        denormalized = {
            "couch": "sofa",
            "potted plant": "plant",
            "tv": "tv_monitor",
        }.get(self._target_class, "")
        if not resp:
            return [], None
        resp = str(resp).replace("\n", ",").split("<answer>")[-1].split("[")[-1].split("]")[0]
        tokens = [p.strip() for p in resp.split(",") if p.strip()]
        if not tokens:
            return [], None
        try:
            return [t for t in tokens[:-1] if t != denormalized], float(tokens[-1])
        except Exception:
            return tokens[:-1], None

    def _sync_stair_direction_confirmation(self) -> bool:
        enabled = self._semantic_stair_processing_enabled()
        if hasattr(self._objmap, "set_stair_direction_confirmation_enabled"):
            self._objmap.set_stair_direction_confirmation_enabled(enabled)
        return enabled

    def _run_detector(self, rgb: np.ndarray, cls_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # Run YOLO on COCO-like classes and OWLv2 on non-COCO classes;
        # combine results and handle filtering downstream in fusion map.
        yolo_classes = self._filter_semantic_stair_classes(getattr(self, "_yolo_classes", self._classes))
        owl_classes = self._filter_semantic_stair_classes(getattr(self, "_owl_classes", []))
        combined: Dict[str, Any] = {"found": False, "bboxes": [], "scores": [], "classes": []}
        self._last_detector_routing = {
            "yolo_classes": list(yolo_classes),
            "owl_classes": list(owl_classes),
            "sources": [],
        }

        if yolo_classes:
            det_yolo = self._gpu.yolov7(rgb, filter_class=yolo_classes)
            self._last_detector_routing["sources"].append(
                {
                    "name": "yolov7",
                    "requested_classes": list(yolo_classes),
                    "found": bool(det_yolo.get("found", False)),
                    "detections": [
                        {"class": c, "score": float(s), "bbox": b}
                        for b, s, c in zip(
                            det_yolo.get("bboxes", []),
                            det_yolo.get("scores", []),
                            det_yolo.get("classes", []),
                        )
                    ],
                }
            )
            if det_yolo.get("found", False):
                combined["bboxes"].extend(det_yolo.get("bboxes") or [])
                combined["scores"].extend(det_yolo.get("scores") or [])
                combined["classes"].extend(det_yolo.get("classes") or [])

        if owl_classes:
            det_owl = self._gpu.owl_v2(rgb, filter_class=owl_classes)
            self._last_detector_routing["sources"].append(
                {
                    "name": "owl_v2",
                    "requested_classes": list(owl_classes),
                    "found": bool(det_owl.get("found", False)),
                    "detections": [
                        {"class": c, "score": float(s), "bbox": b}
                        for b, s, c in zip(
                            det_owl.get("bboxes", []),
                            det_owl.get("scores", []),
                            det_owl.get("classes", []),
                        )
                    ],
                }
            )
            if det_owl.get("found", False):
                # combined["bboxes"].extend(det_owl.get("bboxes") or [])
                # combined["scores"].extend(det_owl.get("scores") or [])
                # combined["classes"].extend(det_owl.get("classes") or [])
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
                    for i in range(len(bbox)):
                        if i == len(bbox):
                            break
                        c1, s1 = xyxy2cs(bbox[i])
                        merge_cnt = 0
                        while True:
                            merged = False
                            for j in range(len(bbox) - 1, i, -1):
                                c2, s2 = xyxy2cs(bbox[j])
                                if np.all(np.abs(c1 - c2) < (s1 + s2) * 0.5):
                                    bbox[i][:2] = np.min(np.stack([bbox[i][:2], bbox[j][:2]]), axis=0)
                                    bbox[i][2:] = np.max(np.stack([bbox[i][2:], bbox[j][2:]]), axis=0)
                                    c1, s1 = xyxy2cs(bbox[i])
                                    merged = True
                                    merge_cnt += 1
                                    score[i] += score[j]
                                    bbox.pop(j)
                                    score.pop(j)
                            if not merged:
                                break
                        if merge_cnt:
                            score[i] /= merge_cnt

                    combined["bboxes"].extend([b.tolist() for b in bbox])
                    combined["scores"].extend(score)
                    combined["classes"].extend(["staircases"] * len(bbox))

                combined["bboxes"].extend(residual.get("bboxes") or [])
                combined["scores"].extend(residual.get("scores") or [])
                combined["classes"].extend(residual.get("classes") or [])

        combined["found"] = len(combined["bboxes"]) > 0
        self._last_detector_output = {
            "found": bool(combined["found"]),
            "detections": [
                {
                    "class": c,
                    "score": float(s),
                    "bbox_normalized": b,
                }
                for b, s, c in zip(
                    combined["bboxes"],
                    combined["scores"],
                    combined["classes"],
                )
            ],
        }
        if not combined["found"]:
            return None
        indices = sorted(range(len(combined["bboxes"])), key=lambda i: combined["bboxes"][i][1])
        combined["bboxes"] = [combined["bboxes"][i] for i in indices]
        combined["scores"] = [combined["scores"][i] for i in indices]
        combined["classes"] = [combined["classes"][i] for i in indices]
        return combined

    def _prepare_update(self, rgb: np.ndarray, det: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        bboxes_norm = det.get("bboxes") or []
        scores = det.get("scores") or []
        classes = det.get("classes") or []
        num_det = min(len(bboxes_norm), len(scores), len(classes))
        if num_det == 0:
            self._last_detection_filter = {
                "reason": "detector_empty_output",
                "detections": [],
            }
            return None

        filter_records = []
        if self.mp3d:
            kept_indices = []
            for i, (score, name) in enumerate(zip(scores, classes)):
                j = self._cls_index.get(name)
                thr = self._yolo_conf_thresh[j]
                filter_records.append(
                    {
                        "index": int(i),
                        "class": name,
                        "score": float(score),
                        "threshold": float(thr),
                        "passed_threshold": bool(float(score) >= float(thr)),
                        "bbox_normalized": bboxes_norm[i],
                        "bbox_px": None,
                        "sam_success": False,
                    }
                )
                if float(score) >= float(thr):
                    kept_indices.append(i)
            if not kept_indices:
                self._last_detection_filter = {
                    "reason": "confidence_threshold",
                    "detections": filter_records,
                }
                return None
        else:
            kept_indices = list(range(num_det))
            filter_records = [
                {
                    "index": int(i),
                    "class": classes[i],
                    "score": float(scores[i]),
                    "threshold": None,
                    "passed_threshold": True,
                    "bbox_normalized": bboxes_norm[i],
                    "bbox_px": None,
                    "sam_success": False,
                }
                for i in range(num_det)
            ]

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

        det_seg: Dict[str, Any] = {
            "bboxes_px": [list(map(int, b)) for b in bboxes_px],
            "scores": kept_scores,
            "classes": kept_classes,
            "masks": masks,
        }
        self._last_detection_filter = {
            "reason": "accepted" if bboxes_px else "all_segmentation_failed",
            "detections": filter_records,
        }
        return det_seg

    def _process_depth(self, depth: np.ndarray) -> np.ndarray:
        # Process depth to ensure it's in the correct format for mapping
        if depth.ndim == 3:
            return depth[..., 0]
        return depth

    def _detect_and_update(self, rgb: np.ndarray, depth: np.ndarray) -> Optional[Dict[str, Any]]:
        had_reliable_target = bool(
            self._target_class and self._objmap.has_object(self._target_class)
        )
        semantic_stair_enabled = self._sync_stair_direction_confirmation()
        det = self._run_detector(rgb)
        if det is None:
            self._last_detection_filter = {
                "reason": "detector_no_detection",
                "detections": [],
            }
        if det is not None:
            det_seg = self._prepare_update(rgb, det)
        if det is None or det_seg is None:
            det_seg = {"classes": [], "masks": [], "scores": []}

        object_names, object_masks, detection_confidences = det_seg["classes"], det_seg["masks"], det_seg["scores"]

        # Fuse 2D detections + masks into the 3D fusion map
        self._objmap.update_map(
            object_names,
            object_masks,
            self._tf_camera_to_episodic,
            detection_confidences,
            rgb,
            depth,
            self._omap.dense_points_epi,
            self._omap.out_of_range_mask,
        )

        has_reliable_target = bool(
            self._target_class and self._objmap.has_object(self._target_class)
        )
        if has_reliable_target:
            tgt_xy = self._objmap.get_best_object(self._target_class, self._curr_xy)
            if tgt_xy is not None and len(tgt_xy) == 2:
                goal_xy = np.asarray(tgt_xy, dtype=np.float32)
                if self.is_banned(goal_xy):
                    return det_seg
                if hasattr(self._omap, "set_selected_frontier"):
                    self._omap.set_selected_frontier(goal_xy)
                if self._goal_xy is None:
                    self._goal_xy = goal_xy
                elif self._distance(goal_xy, self._goal_xy) > self._same_goal_thresh:
                    self._should_reset_pointnav = True
                    self._goal_xy = goal_xy
                # Target acquired: switch to navigation toward the selected instance
                self._phase = "target"
        if had_reliable_target != has_reliable_target:
            self._record_trace_event(
                "target_map_transition",
                target_class=self._target_class,
                reliable_before=had_reliable_target,
                reliable_after=has_reliable_target,
                suspected_ids=[
                    int(getattr(obj, "id", -1))
                    for obj in getattr(self._objmap, "_suspected_objects", [])
                ],
                reliable_ids=[
                    int(getattr(obj, "id", -1))
                    for obj in getattr(self._objmap, "_reliable_objects", [])
                ],
                selected_object_id=getattr(self._objmap, "last_nav_object_id", None),
                selected_goal=self._trace_xy(self._goal_xy),
            )

        # Project accumulated stairs cloud onto obstacle map as obstacles
        if semantic_stair_enabled and "staircases" in object_names:
            stairs_cloud = self._objmap.get_stairs()
            if stairs_cloud.size > 0:
                if hasattr(self._omap, "mark_semantic_obstacles"):
                    self._omap.mark_semantic_obstacles(dbscan(stairs_cloud, 0.8, 50).get(), kind="upstairs")

        self._omap.set_additional_vis_info(
            self._objmap.get_all_vp(),
            np.array([]),
            self._objmap.get_active_viewpoint_ray(),
        )

        return det_seg

    def _choose_goal(self, passby: bool = False) -> None:
        # Prefer unvisited suspect viewpoints as temporary frontiers.
        self._goal_view_facing = None
        if hasattr(self._objmap, "set_selected_viewpoint"):
            self._objmap.set_selected_viewpoint(None)
        suspects, centers = self._objmap.get_suspected_view_positions(self._target_class, self._curr_xy)
        if isinstance(suspects, np.ndarray) and suspects.size >= 2:
            candidates = [(p, c) for p, c in zip(suspects, centers) if not self.is_banned(p)]
            if candidates:
                idx = int(np.random.randint(len(candidates)))
                pos, center = candidates[idx]
                self._goal_xy = np.asarray(pos, dtype=np.float32)
                if hasattr(self._objmap, "set_selected_viewpoint"):
                    self._objmap.set_selected_viewpoint(self._goal_xy)
                if isinstance(center, np.ndarray) and center.size >= 2 and not np.any(np.isnan(center)):
                    self._goal_view_facing = center.astype(np.float32)
                if hasattr(self._omap, "set_selected_frontier"):
                    self._omap.set_selected_frontier(self._goal_xy)
                self._last_goal_selection_source = "suspected_object_viewpoint"
                return
        self._last_goal_selection_source = "random_frontier_or_fallback"
        super()._choose_goal(passby)

    def _maybe_align_view(
        self, det_seg: Optional[Dict[str, Any]], has_obj: bool, thresh=0.6
    ) -> Optional[Dict[str, Any]]:
        # When standing on a suspect viewpoint, spend a few steps turning to face the object center.
        if self._goal_xy is None:
            self._goal_view_facing = None
            return None
        if self._goal_view_facing is None:
            return None
        if self._distance(self._goal_xy, self._curr_xy) > thresh:
            return None
        _, theta = rho_theta(self._curr_xy, self._curr_yaw, self._goal_view_facing)
        if abs(theta) <= np.pi / 12:
            self._goal_view_facing = None
            return None
        return {
            "action": "turn_left" if theta > 0 else "turn_right",
            "map_viz": self._visualize_map(),
            "det_seg": det_seg,
            "nav_goal": self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None,
            "stop_called": False,
            "target_detected": has_obj,
        }
