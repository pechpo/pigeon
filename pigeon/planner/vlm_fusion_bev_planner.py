from .vlm_fusion_planner import VLMFusionPlanner
from typing import Any, Dict, List, Optional, Tuple
import re
import cv2
import numpy as np

from ..utils.viz import mark_image_bgr

class VLMFusionBEVPlanner(VLMFusionPlanner):
    """Placeholder BEV fusion planner.

    The BEV-specific decision logic has been cleared. This class is kept so the
    existing planner registry and imports remain stable until a dedicated BEV
    implementation is added again.
    """

    models = ["pointnav_rnn", "yolov7", "owl_v2", "mobile_sam"]
    _stair_review_image_count = 6
    stagnant_ban_prompt = (
        "You are helping an indoor robot decide whether to temporarily ban its current navigation goal.\n"
        "Image 1 is the current BEV map.\n"
        "Image 2 is the current color BEV map.\n"
        "Images 3 to 6 are recent first-person views related to the current goal.\n"
        "The robot has been stagnant near this goal.\n"
        "The semantic search target is {goal_object}.\n"
        "When visible, the same current goal is marked in the first-person views by a red circular marker labeled 1.\n"
        #"The current navigation goal point is {nav_goal} in episodic XY meters.\n"
        "Reply 1 if this goal region should be banned, otherwise reply 0.\n"
        "The goal is a red point on the BEV and colored BEV map.\n"
        "Large green circle near the red point indicates that the robot fails to approach the goal point.\n"
        "If the red-point goal seems irrelevant to the semantic target or clearly unreachable, ban the goal.\n"
        "Please follow the format:\n"
        "<thought> [reasoning content] </thought>\n"
        "<answer> [1 or 0] </answer>\n"
    )
    stair_skip_prompt = (
        "You are helping an indoor robot decide whether it should leave the current floor and search via stairs.\n"
        "Images 1 to 6 are first-person views selected to summarize covered observations on the current floor.\n"
        "The semantic search target is {goal_object}.\n"
        "Reply 1 if the robot should go search on another floor now, otherwise reply 0.\n"
        "Observations leading to dark regions or dark hallways which lacks data don't represent the environment undiscovered.\n"
        "Even a short flight of stairs can create a barrier between different levels\n"
        "Please follow the format:\n"
        "<thought> [reasoning content] </thought>\n"
        "<answer> [1 or 0] </answer>\n"
    )
    stair_frontier_selection_prompt_template = (
        "You are helping an indoor robot choose a staircase entry to leave the current floor.\n"
        "You are given first-person observations of up to 4 candidate staircase-entry frontiers.\n"
        "The same numbered marker can appear in multiple images and always refers to the same candidate frontier.\n"
        "Some candidates may have fewer observations than others.\n"
        "The semantic search target is {goal_object}.\n"
        "Select the best numbered staircase-entry frontier to move to next.\n"
        #"You must choose a marker that faces the stairs so that you can go up or down immediately after reaching it.\n"
        "You must choose a marker which is located at the edge of the staircase, so that you can go up or down immediately after reaching it.\n"
        "If none looks reliable, return <answer> 0 </answer>.\n"
        "Please follow the format:\n"
        "<thought> [reasoning content] </thought>\n"
        "<answer> [marker id or 0] </answer>\n"
    )
    target_stair_frontier_selection_prompt_template = (
        "You are helping an indoor robot choose a navigation frontier near a stair.\n"
        "You are given first-person observations of up to 4 candidate frontiers.\n"
        "The same numbered marker can appear in multiple images and always refers to the same candidate frontier.\n"
        "Some candidates may have fewer observations than others.\n"
        "The semantic search target is {goal_object}.\n"
        "Select the best numbered frontier that is near the stair.\n"
        "If none looks reliable, return <answer> 0 </answer>.\n"
        "Please follow the format:\n"
        "<thought> [reasoning content] </thought>\n"
        "<answer> [marker id or 0] </answer>\n"
    )
    frontier_selection_bev_prompt_template = (
        "You are a robot and after exploration, you have the given images.\n"
        "You are given first-person observations of candidate frontiers.\n"
        "The same numbered marker can appear in multiple images and always refers to the same candidate frontier.\n"
        "Some candidates may have fewer observations than others.\n"
        "Each candidate view represents a possible direction to find the {goal_object}.\n"
        "Based on the numerical markers in the images, select one of these numbers to move next.\n"
        "If you're not confident in moving to the marker to find the {goal_object}, you should return <answer> 0 </answer>.\n"
        "Please note all closed doors cannot be opened.\n"
        "Please don't select frontiers leading to dark regions or dark hallways which lacks data.\n"
        "Please select frontiers which is on the ground rather than those floating in air.\n"
        "Please follow the format like this, without bracket --\n"
        "<thought> [put your step-by-step thinking process here] </thought>\n"
        "<answer> [put a single marker id or 0 here] </answer>\n"
    )

    def __init__(self) -> None:
        super().__init__()
        self._enable_color_bev = True
        self._persist_xy_history = np.empty((100, 2), dtype=float)
        self._persist_xy_history_start = 0
        self._goal_obs_pool: List[Dict[str, Any]] = []
        self._last_goal_xy: Optional[np.ndarray] = None
        self._tmp_ban_frontier_radius = 3.0
        self._tmp_ban_obstacle_radius = 0.5
        self._tmp_ban_list = np.zeros((0, 2), dtype=np.float32)
        self._stair_query_cooldown = 30
        self._stair_query_exploration_threshold = 100.0
        self._stair_query_stuck_threshold = 0.010  # 0.015
        self._post_floor_change_degree_stair_cooldown = 150
        self._degree_stair_trigger_block_until_tstep = 0
        self._last_stair_query_tstep = -self._stair_query_cooldown
        self._recent_step_rgbs: List[np.ndarray] = []
        self._recent_step_agent_tfs: List[Optional[np.ndarray]] = []
        self._vlm_labeled_frontiers_xy: Optional[np.ndarray] = None
        self._target_stair_area_threshold_m2 = 0.04
        self._final_check_enabled = False
        self._final_check_cooldown_steps = 10
        self._final_check_cooldown_until_step = -1
        self._final_check_attempted: Dict[int, Tuple[int, ...]] = {}
        self._final_check_owl_cache: Dict[int, Tuple[List[List[float]], List[float]]] = {}

    def set_final_check_enabled(self, enabled: bool) -> None:
        self._final_check_enabled = bool(enabled)

    def _final_check_before_target_stop(
        self, det_seg: Optional[Dict[str, Any]], has_obj: bool
    ) -> Optional[Dict[str, Any]]:
        if not self._final_check_enabled or not has_obj:
            return None
        if self._tstep <= self._final_check_cooldown_until_step:
            self._record_trace_event(
                "final_check", result="passed", reason="cooldown_default_pass",
                step=int(self._tstep), cooldown_until_step=int(self._final_check_cooldown_until_step),
            )
            return None

        objmap = self._objmap
        object_id = getattr(objmap, "last_nav_object_id", None)
        obj = next(
            (item for item in getattr(objmap, "_objects", []) if getattr(item, "id", None) == object_id),
            None,
        )
        track = getattr(obj, "labels", {}).get(self._target_class) if obj is not None else None
        if obj is None or track is None or track.is_empty:
            self._record_trace_event("final_check", result="unconfirmed", reason="missing_target_cloud", object_id=object_id)
            return self._final_check_fallback()

        pcd = track.pcd
        if hasattr(pcd, "get"):
            pcd = pcd.get()
        pcd = np.asarray(pcd, dtype=np.float32).reshape(-1, 3)
        if len(pcd) > 12000:
            pcd = pcd[np.linspace(0, len(pcd) - 1, 12000).astype(np.int32)]

        records = getattr(objmap, "image_history", {})
        center = np.asarray(obj.center, dtype=np.float32) if obj.center is not None else np.mean(pcd, axis=0)
        candidates = []
        scan_count = 0
        for image_id, record in records.items():
            image = record.get("image")
            depth = record.get("depth")
            camera_tf = record.get("agent2world")
            if not isinstance(image, np.ndarray) or not isinstance(depth, np.ndarray) or camera_tf is None:
                continue
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            camera_tf = np.asarray(camera_tf, dtype=np.float32)
            camera_pos = camera_tf[:3, 3]
            distance = float(np.linalg.norm(camera_pos - center))
            if not 1.0 <= distance <= 5.0 or depth.ndim != 2:
                continue

            height, width = depth.shape[:2]
            camera_points = (np.linalg.inv(camera_tf) @ np.column_stack((pcd, np.ones(len(pcd)))).T).T[:, :3]
            z = camera_points[:, 0]
            focal = width / (2.0 * np.tan(float(self._camera_fov_rad) / 2.0))
            u = np.rint(width / 2.0 - camera_points[:, 1] * focal / np.maximum(z, 1e-6)).astype(np.int32)
            v = np.rint(height / 2.0 - camera_points[:, 2] * focal / np.maximum(z, 1e-6)).astype(np.int32)
            visible = (z >= self._min_depth) & (z <= self._max_depth) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if int(np.count_nonzero(visible)) < 12:
                continue
            pixels = np.unique(np.column_stack((u[visible], v[visible])), axis=0)
            if len(pixels) < 8:
                continue
            flat_pixels = v[visible] * width + u[visible]
            nearest_z = np.full(height * width, np.inf, dtype=np.float32)
            np.minimum.at(nearest_z, flat_pixels, z[visible])

            # Geometric prefilter avoids OWLv2 calls for frames where the object cloud is out of view.
            scan_count += 1
            cached_detections = self._final_check_owl_cache.get(int(image_id))
            if cached_detections is None:
                try:
                    detection = self._gpu.owl_v2(image, threshold=0.15, filter_class=[self._target_class])
                except Exception as error:
                    self._record_trace_event("final_check_candidate_error", image_id=int(image_id), error=repr(error))
                    continue
                boxes = detection.get("bboxes", []) if detection.get("found") else []
                scores = detection.get("scores", []) if detection.get("found") else []
                cached_detections = (boxes, scores)
                self._final_check_owl_cache[int(image_id)] = cached_detections
            boxes, scores = cached_detections
            best = None
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = np.asarray(box, dtype=np.float32) * np.array([width, height, width, height])
                inside = (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) & (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
                count = int(np.count_nonzero(inside))
                if count and (best is None or count > best[0]):
                    best = (count, float(score), inside)
            if best is None or best[0] < 8:
                continue

            projected = pixels[best[2]]
            observed_norm = depth[projected[:, 1], projected[:, 0]].astype(np.float32)
            valid_depth = np.isfinite(observed_norm) & (observed_norm > 0.0) & (observed_norm < 0.99)
            projected = projected[valid_depth]
            if len(projected) < 8:
                continue
            observed_m = observed_norm[valid_depth] * (self._max_depth - self._min_depth) + self._min_depth
            expected_m = nearest_z[projected[:, 1] * width + projected[:, 0]]
            matched = np.abs(observed_m - expected_m) <= np.maximum(0.12, expected_m * 0.04)
            ratio = float(np.count_nonzero(matched) / len(projected))
            if ratio < 0.35:
                continue
            matched_pixels = projected[matched]
            direction = camera_pos[[0, 2]] - center[[0, 2]]
            bearing = float(np.arctan2(direction[1], direction[0]))
            candidates.append({
                "image_id": int(image_id), "image": image, "pixels": matched_pixels,
                "depth_shape": (height, width),
                "ratio": ratio, "match_count": int(len(matched_pixels)), "depth_count": int(len(projected)),
                "owl_score": best[1], "distance": distance, "bearing": bearing,
            })

        selected = self._select_final_check_views(candidates)
        signature = tuple(item["image_id"] for item in selected)
        prior_signature = self._final_check_attempted.get(int(object_id))
        if not selected or signature == prior_signature:
            reason = "no_depth_matched_views" if not selected else "same_views_already_checked"
            self._record_trace_event(
                "final_check", result="unconfirmed", reason=reason, object_id=int(object_id),
                geometric_frames_scanned=scan_count, candidate_count=len(candidates), selected_image_ids=list(signature),
            )
            return self._final_check_fallback()

        self._final_check_attempted[int(object_id)] = signature
        outlined = [self._outline_final_check_view(item) for item in selected]
        prompt = (
            f"The green outlines show the projected point-cloud boundary of a candidate 3D region in {len(outlined)} view(s).\n"
            f"Determine whether this region belongs to the indoor target object: {self._vlm_target_name()}.\n"
            "Judge the outlined region itself across the views. Do not decide based on whether the target appears elsewhere. "
            "Perspective or occlusion may make one view ambiguous.\n"
            "Respond True when the combined evidence clearly supports that the outlined region is the target object. "
            "Respond False when the combined evidence clearly shows a different object. Respond Unconfirmed when evidence is insufficient.\n"
            "Please follow the format exactly, without brackets:\n"
            "<thought> [brief reasoning] </thought>\n"
            "<answer> [True, False, or Unconfirmed] </answer>"
        )
        self._handle_vlm_exchange(tuple(outlined), prompt)
        response = str(self._vlm._get_response(rgb=tuple(outlined), text=prompt, failed_response=""))
        self._handle_vlm_exchange(tuple(outlined), prompt, response)
        answer_match = re.search(r"<answer>\s*(true|false|unconfirmed)\s*</answer>", response, flags=re.IGNORECASE)
        answer = answer_match.group(1).lower() if answer_match else "invalid_response"
        object_removed = objmap.remove_object(object_id) if answer == "false" else False
        self._record_trace_event(
            "final_check", result=answer, raw_response=response, object_id=int(object_id),
            object_removed=object_removed,
            geometric_frames_scanned=scan_count, candidate_count=len(candidates), selected_image_ids=list(signature),
            selected_views=[{
                "image_id": item["image_id"], "distance_m": round(item["distance"], 3),
                "depth_match_ratio": round(item["ratio"], 4), "matched_depth_pixels": item["match_count"],
                "valid_projected_depth_pixels": item["depth_count"], "owl_score": round(item["owl_score"], 4),
                "bearing_rad": round(item["bearing"], 4),
            } for item in selected],
        )
        if answer in {"true", "unconfirmed"}:
            return None
        return self._final_check_fallback()

    @staticmethod
    def _select_final_check_views(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        remaining = sorted(candidates, key=lambda item: (item["ratio"], item["owl_score"]), reverse=True)
        selected = [remaining.pop(0)]
        while remaining and len(selected) < 4:
            ranked = []
            for item in remaining:
                angular = [abs((item["bearing"] - chosen["bearing"] + np.pi) % (2 * np.pi) - np.pi) for chosen in selected]
                ranked.append((min(angular), item["ratio"], item["owl_score"], item))
            ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
            chosen = ranked[0][3]
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    @staticmethod
    def _outline_final_check_view(item: Dict[str, Any]) -> np.ndarray:
        image = np.asarray(item["image"]).copy()
        pixels = np.asarray(item["pixels"], dtype=np.float32)
        depth_height, depth_width = item["depth_shape"]
        image_height, image_width = image.shape[:2]
        pixels[:, 0] *= image_width / max(depth_width, 1)
        pixels[:, 1] *= image_height / max(depth_height, 1)
        pixels = np.rint(pixels).astype(np.int32)
        if len(pixels) >= 3:
            hull = cv2.convexHull(pixels.reshape(-1, 1, 2))
            cv2.polylines(image, [hull], True, (0, 255, 0), 3, cv2.LINE_AA)
        elif len(pixels):
            for x, y in pixels:
                cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
        return image

    def _final_check_fallback(self) -> Dict[str, Any]:
        self._final_check_cooldown_until_step = self._tstep + self._final_check_cooldown_steps
        self._phase = "explore"
        self._goal_xy = None
        self._goal_view_facing = None
        self._should_reset_pointnav = True
        return {
            "action": "turn_left",
            "map_viz": self._visualize_map(),
            "right_panel_viz": self._vlm_grid_bgr,
            "right_panel_text": self._get_bottom_text(),
            "det_seg": None,
            "nav_goal": None,
            "stop_called": False,
            "target_detected": False,
        }

    def reset(self, reset_stairs: bool = True) -> None:
        super().reset(reset_stairs)
        self._persist_xy_history = np.empty((100, 2), dtype=float)
        self._persist_xy_history_start = 0
        self._goal_obs_pool = []
        self._last_goal_xy = None
        self._tmp_ban_list = np.zeros((0, 2), dtype=np.float32)
        self._degree_stair_trigger_block_until_tstep = 0
        self._last_stair_query_tstep = -self._stair_query_cooldown
        self._recent_step_rgbs = []
        self._recent_step_agent_tfs = []
        self._vlm_labeled_frontiers_xy = None
        self._final_check_attempted = {}
        self._final_check_owl_cache = {}
        self._final_check_cooldown_until_step = -1

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        result = super().act(obs)
        if not self._is_last_eval_step(obs):
            return result

        pre_forced_stop = {
            "action": result.get("action", "stop"),
            "nav_goal": result.get("nav_goal", None),
            "target_detected": bool(result.get("target_detected", False)),
            "stop_called": bool(result.get("stop_called", result.get("action") == "stop")),
        }
        result = dict(result)
        result["action"] = "stop"
        result["stop_called"] = True
        result["forced_stop"] = True
        result["forced_stop_reason"] = "last_step"
        result["pre_forced_stop"] = pre_forced_stop
        return result

    def _is_last_eval_step(self, obs: Dict[str, Any]) -> bool:
        try:
            step = int(obs.get("eval_step", -1))
            max_steps = int(obs.get("max_steps", 0))
        except (TypeError, ValueError):
            return False
        return max_steps > 0 and step >= max_steps - 1

    def _finish_stair_transition(self, goal_sign: int) -> None:
        super()._finish_stair_transition(goal_sign)
        self._degree_stair_trigger_block_until_tstep = int(self._post_floor_change_degree_stair_cooldown)

    def get_status_text(self) -> str:
        base_status = super().get_status_text()
        hist = self._get_persist_xy_history(1)
        frustum_mask = getattr(self._omap, "last_update_frustum_mask", None)
        if self._goal_xy is None:
            goal_in_frustum = "n/a"
        elif isinstance(frustum_mask, np.ndarray) and frustum_mask.size > 0:
            goal_in_frustum = "true" if self._goal_in_frustum_mask(frustum_mask) else "false"
        else:
            goal_in_frustum = "unknown"
        return (
            f"{base_status} | bev_goal_imgs: {len(self._goal_obs_pool)}"
            f" | tstep: {self._tstep}"
            f" | hist_len: {len(hist)}"
            f" | goal_in_frustum: {goal_in_frustum}"
        )

    def _active_banned_points(self) -> np.ndarray:
        active = self._banned_points if isinstance(self._banned_points, np.ndarray) else np.zeros((0, 2), dtype=np.float32)
        if self._tmp_ban_list.size == 0:
            return active
        if active.size == 0:
            return self._tmp_ban_list
        return np.vstack([active, self._tmp_ban_list])

    def _tmp_ban_mask(self, xy: np.ndarray) -> np.ndarray:
        # Return a boolean mask for points that fall within any temporary ban center's radius.
        if self._tmp_ban_list.size == 0:
            return np.zeros(len(xy), dtype=bool)
        dists = np.linalg.norm(xy[:, None, :] - self._tmp_ban_list[None, :, :], axis=-1)
        return np.any(dists < self._tmp_ban_frontier_radius, axis=1)

    def _sync_tmp_ban_frontier_exclusion(self) -> None:
        if hasattr(self._omap, "set_exposed_boundary_exclusion"):
            self._omap.set_exposed_boundary_exclusion(self._tmp_ban_list, self._tmp_ban_frontier_radius)

    def _active_ban_mask(self, xy: np.ndarray) -> np.ndarray:
        if not isinstance(xy, np.ndarray):
            xy = np.asarray(xy, dtype=np.float32)
        if xy.ndim == 1:
            xy = xy.reshape(1, 2)
        perm_mask = np.zeros(xy.shape[0], dtype=bool)
        if self._banned_points.size > 0:
            dists = np.linalg.norm(xy[:, None, :] - self._banned_points[None, :, :], axis=-1)
            perm_mask = np.any(dists < float(self._ban_threshold), axis=1)
        return perm_mask | self._tmp_ban_mask(xy)

    def is_banned(self, point_xy: Optional[np.ndarray]) -> bool:
        if point_xy is None:
            return False
        if not isinstance(point_xy, np.ndarray):
            point_xy = np.asarray(point_xy, dtype=np.float32)
        if point_xy.size != 2:
            return False
        return bool(self._active_ban_mask(point_xy.reshape(1, 2))[0])

    def _invalidate_banned_frontier_memory(self) -> None:
        for mem in getattr(self, "_frontier_mem", []):
            xy = mem.get("xy", None)
            if xy is not None and self.is_banned(xy):
                mem["valid"] = False

    def _sync_tmp_ban_obstacles(self) -> None:
        if hasattr(self._omap, "set_temporary_obstacles"):
            self._omap.set_temporary_obstacles(self._tmp_ban_list, self._tmp_ban_obstacle_radius)

    def _update_mapping(self, depth: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
        self._sync_tmp_ban_frontier_exclusion()
        self._sync_tmp_ban_obstacles()
        super()._update_mapping(depth, rgb)
        self._maintain_tmp_ban_point(sync_obstacles=False, sync_frontier_exclusion=False)

    def get_color_bev_viz(self) -> Optional[np.ndarray]:
        if self._color_bev is None:
            return None
        goal_xy = self._goal_xy if isinstance(self._goal_xy, np.ndarray) and self._goal_xy.size == 2 else None
        banned = self._active_banned_points()
        frontiers = self._frontiers if isinstance(self._frontiers, np.ndarray) else None
        temporary_obstacles = self._tmp_ban_list if self._tmp_ban_list.size > 0 else None
        return self._color_bev.visualize(
            frontiers_xy=frontiers,
            selected_frontier_xy=goal_xy,
            banned_xy=banned if banned.size > 0 else None,
            temporary_obstacle_xy=temporary_obstacles,
            temporary_obstacle_radius_m=self._tmp_ban_obstacle_radius,
            labeled_frontiers_xy=self._vlm_labeled_frontiers_xy,
        )

    def get_height_bev_viz(self) -> Optional[np.ndarray]:
        if self._height_bev is None:
            return None
        goal_xy = self._goal_xy if isinstance(self._goal_xy, np.ndarray) and self._goal_xy.size == 2 else None
        banned = self._active_banned_points()
        frontiers = self._frontiers if isinstance(self._frontiers, np.ndarray) else None
        temporary_obstacles = self._tmp_ban_list if self._tmp_ban_list.size > 0 else None
        downstair_overlay_mask, downstair_overlay_height = self._get_downstair_height_overlay()
        return self._height_bev.visualize(
            frontiers_xy=frontiers,
            selected_frontier_xy=goal_xy,
            banned_xy=banned if banned.size > 0 else None,
            temporary_obstacle_xy=temporary_obstacles,
            temporary_obstacle_radius_m=self._tmp_ban_obstacle_radius,
            labeled_frontiers_xy=self._vlm_labeled_frontiers_xy,
            height_overlay_mask=downstair_overlay_mask,
            height_overlay_values=downstair_overlay_height,
        )

    def _show_bev_exchange(self, images, text: str, has_bev: bool = True) -> None:
        panel_images = images
        if has_bev and isinstance(images, (list, tuple)):
            panel_images = list(images)[2:]
        self._show_vlm_exchange(panel_images, text)

    def _handle_bev_exchange(
        self,
        images,
        prompt: str,
        response: Optional[str] = None,
        has_bev: bool = True,
    ) -> None:
        text = f"Prompt: {prompt}" if response is None else f"VLM: {response}"
        self._show_bev_exchange(images, text, has_bev=has_bev)

    def _refresh_vlm_viz(self) -> None:
        self._update_viz(
            planner_viz=self._visualize_map(labeled_frontiers_xy=self._vlm_labeled_frontiers_xy),
            aux_map_viz=self.get_color_bev_viz(),
            right_panel_viz=self._vlm_grid_bgr,
            bottom_text=self._get_bottom_text(),
            refresh=True,
        )

    def _select_covering_observations(
        self,
        observations: List[Dict[str, Any]],
        max_views: int = 2,
        coverage_mask: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if not observations:
            return []
        selected: List[Dict[str, Any]] = []
        covered = None
        remaining = list(range(len(observations)))
        while remaining and len(selected) < max_views:
            best_idx = remaining[0]
            best_gain = -1
            for idx in remaining:
                mask = observations[idx].get("frustum_mask")
                if isinstance(mask, np.ndarray) and mask.size > 0:
                    visible = mask.astype(bool, copy=False)
                    if isinstance(coverage_mask, np.ndarray) and coverage_mask.shape == visible.shape:
                        visible = visible & coverage_mask
                    gain = int(np.count_nonzero(visible if covered is None else (visible & ~covered)))
                else:
                    gain = 0
                if gain > best_gain:
                    best_idx = idx
                    best_gain = gain
            selected.append(observations[best_idx])
            mask = observations[best_idx].get("frustum_mask")
            if isinstance(mask, np.ndarray) and mask.size > 0:
                visible = mask.astype(bool, copy=False)
                if isinstance(coverage_mask, np.ndarray) and coverage_mask.shape == visible.shape:
                    visible = visible & coverage_mask
                covered = visible.copy() if covered is None else (covered | visible)
            remaining.remove(best_idx)
        return selected

    def _get_candidate_views(
        self,
        candidate: Dict[str, Any],
        max_views: int = 2,
    ) -> List[Dict[str, Any]]:
        preset_views = candidate.get("views")
        if isinstance(preset_views, list) and preset_views:
            return preset_views[:max_views]

        xy = np.asarray(candidate["xy"], dtype=np.float32).reshape(2)
        observations: List[Dict[str, Any]] = []
        for rec in self._iter_current_layer_mapping_history():
            rgb, agent_tf = self._get_mapping_observation_data(rec)
            if not isinstance(rgb, np.ndarray) or not isinstance(agent_tf, np.ndarray):
                continue
            uv = self._project_to_uv(xy, rgb, tf_camera_to_episodic=agent_tf)
            if uv is None:
                continue
            frustum_mask = self._refresh_mapping_record_frustum(rec)
            observations.append(
                {
                    "rgb": rgb.copy(),
                    "pixel": uv,
                    "frustum_mask": (
                        frustum_mask.astype(bool, copy=True)
                        if isinstance(frustum_mask, np.ndarray)
                        else None
                    ),
                }
            )

        fallback_rgb = candidate.get("rgb")
        fallback_pixel = candidate.get("pixel")
        if isinstance(fallback_rgb, np.ndarray) and fallback_pixel is not None:
            observations.append(
                {
                    "rgb": fallback_rgb.copy(),
                    "pixel": tuple(fallback_pixel),
                    "frustum_mask": None,
                }
            )

        return self._select_covering_observations(
            observations,
            max_views=max_views,
            coverage_mask=candidate.get("coverage_mask"),
        )

    def _get_mapping_observation_data(self, rec: Dict[str, Any]):
        image_id = rec.get("image_id")
        if image_id is not None:
            image_history = getattr(self._objmap, "image_history", {})
            image_record = image_history.get(image_id)
            if not isinstance(image_record, dict):
                return None, None
            return image_record.get("image"), image_record.get("agent2world")
        return rec.get("rgb"), rec.get("agent_tf")

    def _build_candidate_view_images(
        self,
        candidates: List[Dict[str, Any]],
        max_views_per_candidate: int = 2,
    ) -> List[np.ndarray]:
        imgs_rgb: List[np.ndarray] = []
        for label, candidate in enumerate(candidates, start=1):
            for view in self._get_candidate_views(candidate, max_views=max_views_per_candidate):
                bgr = cv2.cvtColor(view["rgb"], cv2.COLOR_RGB2BGR)
                u, v = int(view["pixel"][0]), int(view["pixel"][1])
                bgr = mark_image_bgr(bgr, label, u, v, radius=9)
                imgs_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return imgs_rgb

    def _vlm_select_with_prompt(self, candidates, prompt: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if len(candidates) == 0:
            return None, None
        candidate_imgs_rgb = self._build_candidate_view_images(candidates)
        if len(candidate_imgs_rgb) == 0:
            return None, None
        self._last_vlm_decision = {
            "type": "bev_candidate_selection",
            "step": int(self._tstep),
            "prompt": prompt,
            "candidates": [
                {
                    "index": int(idx),
                    "xy": self._trace_candidate_xy(candidate),
                    "source_index": candidate.get("idx"),
                }
                for idx, candidate in enumerate(candidates, start=1)
            ],
            "selected_index": None,
            "selected_xy": None,
            "response": None,
        }
        self._vlm_labeled_frontiers_xy = np.stack([np.asarray(c["xy"], dtype=np.float32) for c in candidates], axis=0)
        sel = 0
        try:
            imgs_rgb = candidate_imgs_rgb
            self._handle_bev_exchange(imgs_rgb, prompt, has_bev=False)
            resp = self._vlm._get_response(tuple(imgs_rgb), prompt, failed_response="<answer> 0 </answer>")
            reply_text = "" if resp is None else str(resp)
            self._handle_bev_exchange(imgs_rgb, prompt, reply_text, has_bev=False)
            try:
                sel = int(reply_text.split("<answer>")[1].split("</answer>")[0].strip())
            except Exception:
                sel = 0
            self._last_vlm_decision["selected_index"] = int(sel)
            self._last_vlm_decision["response"] = reply_text
            self._record_trace_event(
                "vlm_decision",
                decision_type="bev_candidate_selection",
                selected_index=int(sel),
                candidate_count=int(len(candidates)),
                candidate_xy=[
                    self._trace_candidate_xy(candidate) for candidate in candidates
                ],
                response=reply_text[-1000:],
            )
        finally:
            self._vlm_labeled_frontiers_xy = None
        if sel <= 0 or sel > len(candidates):
            return None, self._vlm_grid_bgr
        chosen = candidates[sel - 1]
        self._last_vlm_decision["selected_xy"] = self._trace_candidate_xy(chosen)
        return chosen["xy"].astype(np.float32), self._vlm_grid_bgr

    def _vlm_select(self, candidates) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        prompt = self.frontier_selection_bev_prompt_template.format(goal_object=self._vlm_target_name())
        return self._vlm_select_with_prompt(candidates, prompt)
    
    def _update_persist_xy_history(self):
        if self._goal_xy is None:
            self._last_goal_xy = None
        else:
            goal_xy = np.asarray(self._goal_xy, dtype=np.float32).reshape(2)
            if self._last_goal_xy is not None and np.linalg.norm(goal_xy - self._last_goal_xy) > 1e-6:
                self._persist_xy_history_start = self._tstep
            self._last_goal_xy = goal_xy.copy()
        self._persist_xy_history[self._tstep % len(self._persist_xy_history)] = self._curr_xy
        if (self._tstep + 1 - self._persist_xy_history_start) % len(self._persist_xy_history) == 0:
            self._persist_xy_history_start += 1

    def _get_persist_xy_history(self, gap: int = 50):
        l = len(self._persist_xy_history)
        gap = min(self._tstep - self._persist_xy_history_start, gap)
        start, end = self._persist_xy_history_start % l, (self._tstep - gap + l) % l
        if start <= end:
            return self._persist_xy_history[start:end]
        return np.concatenate([self._persist_xy_history[start:], self._persist_xy_history[:end]], axis=0)

    def _xy_in_frustum_mask(self, xy: Optional[np.ndarray], frustum_mask: np.ndarray) -> bool:
        if xy is None:
            return False
        goal_px = self._omap._xy_to_px(np.asarray(xy, dtype=np.float32).reshape(1, 2))[0]
        row, col = int(goal_px[1]), int(goal_px[0])
        if row < 0 or col < 0 or row >= frustum_mask.shape[0] or col >= frustum_mask.shape[1]:
            return False
        return bool(frustum_mask[row, col])

    def _refresh_mapping_record_frustum(self, rec: Dict[str, Any]) -> Optional[np.ndarray]:
        frustum_mask = rec.get("frustum_mask")
        omap = getattr(self, "_omap", None)
        _, agent_tf = self._get_mapping_observation_data(rec)
        current_revision = int(getattr(omap, "view_mask_revision", -1))
        if (
            isinstance(frustum_mask, np.ndarray)
            and frustum_mask.size > 0
            and int(rec.get("frustum_revision", current_revision)) == current_revision
        ):
            return frustum_mask
        if not isinstance(agent_tf, np.ndarray) or not hasattr(omap, "recompute_view_frustum_mask"):
            return frustum_mask if isinstance(frustum_mask, np.ndarray) and frustum_mask.size > 0 else None
        refreshed = omap.recompute_view_frustum_mask(agent_tf, float(self._camera_fov_rad), float(self._max_depth))
        rec["frustum_mask"] = refreshed.astype(bool, copy=True)
        rec["frustum_revision"] = current_revision
        return rec["frustum_mask"]

    def _goal_in_frustum_mask(self, frustum_mask: np.ndarray) -> bool:
        return self._xy_in_frustum_mask(self._goal_xy, frustum_mask)

    def _get_observation_pool_for_xy(self, goal_xy: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        if goal_xy is None:
            return []
        layer_idx = int(getattr(self, "stair_idx", 0))
        matched_records: List[Dict[str, Any]] = []
        for rec in self.mapping_obs_history:
            rgb, agent_tf = self._get_mapping_observation_data(rec)
            if not isinstance(rgb, np.ndarray) or not isinstance(agent_tf, np.ndarray):
                continue
            if self._project_to_uv(goal_xy, rgb, tf_camera_to_episodic=agent_tf) is None:
                continue
            if int(rec.get("layer_idx", layer_idx)) != layer_idx:
                continue
            frustum_mask = self._refresh_mapping_record_frustum(rec)
            if not isinstance(frustum_mask, np.ndarray) or frustum_mask.size == 0:
                continue
            if not self._xy_in_frustum_mask(goal_xy, frustum_mask):
                continue
            matched = {
                "frustum_mask": frustum_mask.astype(bool, copy=True),
                "agent_tf": agent_tf.copy(),
            }
            if rec.get("image_id") is not None:
                matched["image_id"] = rec["image_id"]
            else:
                matched["rgb"] = rgb.copy()
            matched_records.append(matched)
        return matched_records

    def _iter_current_layer_mapping_history(self):
        layer_idx = int(getattr(self, "stair_idx", 0))
        for rec in reversed(self.mapping_obs_history):
            if int(rec.get("layer_idx", layer_idx)) != layer_idx:
                continue
            yield rec

    def _refresh_goal_observation_pool(self) -> None:
        self._goal_obs_pool = self._get_observation_pool_for_xy(self._goal_xy)

    def _fallback_stair_frontier_goal_idx(
        self,
        stair_frontiers: np.ndarray,
        stair_goal_vec: np.ndarray,
    ) -> Optional[int]:
        if not isinstance(stair_frontiers, np.ndarray) or stair_frontiers.size < 2:
            return None
        dists = np.linalg.norm(stair_frontiers - self._curr_xy.reshape(1, 2), axis=1)
        goal_signs = np.sign(np.asarray(stair_goal_vec, dtype=np.float32).reshape(-1))
        for sign in (-1.0, 1.0):
            mask = goal_signs == sign
            if np.any(mask):
                indices = np.flatnonzero(mask)
                return int(indices[int(np.argmin(dists[indices]))])
        return None

    def _select_stair_frontier_goal_idx(
        self,
        stair_frontiers: np.ndarray,
        stair_frontiers_view_facing: np.ndarray,
        stair_goal_vec: np.ndarray,
    ) -> Optional[int]:
        if not isinstance(stair_frontiers, np.ndarray) or stair_frontiers.size < 2:
            return None
        dists = np.linalg.norm(stair_frontiers - self._curr_xy.reshape(1, 2), axis=1)
        order = np.argsort(dists)
        candidates = []
        for idx in order:
            kind = "down" if stair_goal_vec[idx][0] < 0 else "up"
            rec_fn = getattr(self._omap, "_stair_frontier_record", None)
            rec = rec_fn(stair_frontiers[idx], kind=kind) if callable(rec_fn) else None
            component_mask = rec.get("component_mask") if isinstance(rec, dict) else None
            views = self._get_candidate_views(
                {
                    "xy": stair_frontiers[idx].astype(np.float32),
                    "coverage_mask": component_mask,
                },
                max_views=2,
            )
            if not views:
                continue
            candidates.append(
                {
                    "xy": stair_frontiers[idx].astype(np.float32),
                    "views": views,
                    "idx": int(idx),
                }
            )
            if len(candidates) >= 4:
                break
        if not candidates:
            return self._fallback_stair_frontier_goal_idx(stair_frontiers, stair_goal_vec)
        template = (
            self.target_stair_frontier_selection_prompt_template
            if self._target_is_stair()
            else self.stair_frontier_selection_prompt_template
        )
        prompt = template.format(goal_object=self._vlm_target_name())
        chosen_xy, _ = self._vlm_select_with_prompt(candidates, prompt)
        if chosen_xy is None:
            if not self._has_frontier_option():
                return self._fallback_stair_frontier_goal_idx(stair_frontiers, stair_goal_vec)
            return None
        for candidate in candidates:
            if np.allclose(candidate["xy"], chosen_xy):
                return int(candidate["idx"])
        return None

    def _mark_goal_on_rgb(
        self,
        rgb: np.ndarray,
        agent_tf: Optional[np.ndarray],
        target_xy: Optional[np.ndarray] = None,
        label: int = 1,
    ) -> np.ndarray:
        marked_rgb = rgb.copy()
        goal_xy = self._goal_xy if target_xy is None else target_xy
        if goal_xy is None or not isinstance(agent_tf, np.ndarray):
            return marked_rgb
        uv = self._project_to_uv(
            np.asarray(goal_xy, dtype=np.float32).reshape(2),
            marked_rgb,
            tf_camera_to_episodic=agent_tf,
        )
        if uv is None:
            return marked_rgb
        bgr = cv2.cvtColor(marked_rgb, cv2.COLOR_RGB2BGR)
        bgr = mark_image_bgr(bgr, label, int(uv[0]), int(uv[1]), radius=9)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _recently_stagnant(self) -> bool:
        # The extra BEV review is only meaningful after the episode has matured enough
        # to accumulate map/history context, and when recent GPS motion is very small.
        if self._tstep <= 40:
            return False
        hist = self._get_persist_xy_history(1)
        if len(hist) < 40:
            return False
        stagnant = np.linalg.norm(self._curr_xy - np.mean(hist[-40:], axis=0)) < (0.6 if self._goal_view_facing is None else 0.1)
        if stagnant:
            self._persist_xy_history_start = self._tstep
        return stagnant

    def _tmp_ban_current_goal(self) -> bool:
        if self._goal_xy is None:
            return False
        goal_before = np.asarray(self._goal_xy, dtype=np.float32).reshape(-1)[:2].tolist()
        goal_xy = np.asarray(self._goal_xy, dtype=np.float32).reshape(1, 2)
        if not np.any(self._tmp_ban_mask(goal_xy)):
            self._tmp_ban_list = np.vstack([self._tmp_ban_list, goal_xy])
            self._maintain_tmp_ban_point()
            self._record_trace_event(
                "temporary_goal_banned",
                point_xy=goal_before,
                reason="stagnation",
                radius=float(self._tmp_ban_frontier_radius),
                temporary_ban_count=int(len(self._tmp_ban_list)),
            )
        self._goal_xy = None
        self._goal_view_facing = None
        self._goal_obs_pool = []
        self._should_reset_pointnav = True
        if hasattr(self._omap, "set_selected_frontier"):
            self._omap.set_selected_frontier(None)
        return True
    
    def _stagnant_ban_vlm(self, use_color_bev: Optional[bool] = None):
        if self._goal_xy is None:
            return False
        goal_obs_pool = self._goal_obs_pool
        if len(goal_obs_pool) < 2:
            return self._tmp_ban_current_goal()
        normal_bev_bgr = self._visualize_map()
        color_bev_enabled = self._color_bev is not None if use_color_bev is None else bool(use_color_bev)
        color_bev_bgr = self.get_color_bev_viz() if color_bev_enabled else None
        if normal_bev_bgr is None and color_bev_bgr is None:
            return False
        if normal_bev_bgr is None:
            normal_bev_bgr = color_bev_bgr
        if color_bev_bgr is None:
            color_bev_bgr = normal_bev_bgr

        imgs_rgb = [
            cv2.cvtColor(normal_bev_bgr, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(color_bev_bgr, cv2.COLOR_BGR2RGB),
        ]
        goal_img_count = min(len(goal_obs_pool), 4)
        goal_img_ids = np.random.choice(len(goal_obs_pool), goal_img_count, replace=False)
        for idx in goal_img_ids:
            rgb, agent_tf = self._get_mapping_observation_data(goal_obs_pool[idx])
            if isinstance(rgb, np.ndarray):
                imgs_rgb.append(self._mark_goal_on_rgb(rgb, agent_tf))
        missing_goal_views = max(0, 4 - goal_img_count)
        if missing_goal_views > 0:
            recent_rgbs = [
                self._mark_goal_on_rgb(rgb, agent_tf)
                for rgb, agent_tf in zip(
                    reversed(self._recent_step_rgbs[-2:]),
                    reversed(self._recent_step_agent_tfs[-2:]),
                )
            ]
            imgs_rgb.extend(recent_rgbs[:missing_goal_views])
        goal_xy = np.asarray(self._goal_xy, dtype=np.float32).reshape(-1)
        prompt = self.stagnant_ban_prompt.format(
            goal_object=self._vlm_target_name(),
        )
        self._handle_bev_exchange(imgs_rgb, prompt)
        vlm_input = tuple(imgs_rgb)
        resp = str(self._vlm._get_response(vlm_input, prompt, failed_response="<answer> 0 </answer>")).lower()
        ans = resp.split("<answer>")[-1].split("</answer>")[0].strip() if "<answer>" in resp else resp.strip()
        should_ban = ans.startswith(("1", "t", "y"))
        self._record_trace_event(
            "vlm_decision",
            decision_type="stagnant_goal_ban",
            decision=bool(should_ban),
            response=resp[-1000:],
            goal_xy=goal_xy.tolist(),
            goal_observation_count=int(len(goal_obs_pool)),
        )
        if should_ban:
            self._tmp_ban_current_goal()

        self._handle_bev_exchange(imgs_rgb, prompt, resp)
        if not should_ban:
            return False
        return True
    
    def _maintain_tmp_ban_point(self, sync_obstacles: bool = True, sync_frontier_exclusion: bool = True):
        if sync_frontier_exclusion:
            self._sync_tmp_ban_frontier_exclusion()
        if sync_obstacles:
            self._sync_tmp_ban_obstacles()
        raw_frontiers = getattr(self._omap, "frontiers", None)
        if isinstance(raw_frontiers, np.ndarray) and raw_frontiers.size >= 2:
            self._frontiers = raw_frontiers[~self._active_ban_mask(raw_frontiers)]
        else:
            self._frontiers = np.array([])
        self._sync_frontier_visualization()
        self._invalidate_banned_frontier_memory()
        self._omap.set_banned_frontiers(self._active_banned_points())

    def _clear_tmp_bans(self) -> None:
        self._tmp_ban_list = np.zeros((0, 2), dtype=np.float32)
        self._maintain_tmp_ban_point()
    
    def _handle_stair_strategy(self):
        if not getattr(self, "multifloor", False):
            return
        if self._target_is_stair():
            if (
                self._phase not in {"nav_to_stair", "pointnav_descent", "stair_nav", "post_stair_settle"}
                and self._target_stair_area_ready()
                and self._tstep - self._last_stair_query_tstep >= self._stair_query_cooldown
            ):
                self._last_stair_query_tstep = self._tstep
                self._goal_xy = None
                self._goal_view_facing = None
                self.stair_frontier_goal_idx = None
                self._should_reset_pointnav = True
                self._phase = "nav_to_stair"
            return

        has_stair_option = self._has_stair_option()
        has_frontier_option = self._has_frontier_option()
        same_floor_goal_exhausted = not has_frontier_option
        degree_stair_trigger_ready = self._tstep >= int(self._degree_stair_trigger_block_until_tstep)
        should_change_floor = (
            self._tstep > 50
            and has_stair_option
            and degree_stair_trigger_ready
            and (
                getattr(self._omap, "exploration_degree", 0.0)
                > float(getattr(self, "_stair_query_exploration_threshold", 100.0))
                or self._get_stuck_degree() > self._stair_query_stuck_threshold
            )
        )
        in_stair_transition = self._phase in {"nav_to_stair", "pointnav_descent", "stair_nav", "post_stair_settle"}
        if in_stair_transition:
            return
        should_query = same_floor_goal_exhausted or should_change_floor
        if not should_query:
            return
        if self._tstep - self._last_stair_query_tstep < self._stair_query_cooldown:
            return
        if not has_stair_option:
            if not has_frontier_option:
                self._clear_tmp_bans()
            return
        self._last_stair_query_tstep = self._tstep
        if self._stair_skip_vlm():
            self._goal_xy = None
            self._goal_view_facing = None
            self._goal_obs_pool = []
            self.stair_frontier_goal_idx = None
            self._should_reset_pointnav = True
            self._phase = "nav_to_stair"
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)
            return

    def _same_floor_goal_exhausted(self) -> bool:
        return not self._has_frontier_option()

    def _has_frontier_option(self) -> bool:
        valid_mem = [m for m in getattr(self, "_frontier_mem", []) if m.get("valid", False)]
        frontiers = getattr(self, "_frontiers", None)
        return bool(valid_mem or (isinstance(frontiers, np.ndarray) and len(frontiers) > 0))

    def _has_stair_option(self) -> bool:
        has_down = len(self._omap.downstair_frontier) and (len(self._omap.upstair_frontier) == 0 or not self.ascended)
        has_up = len(self._omap.upstair_frontier) and (len(self._omap.downstair_frontier) == 0 or not self.descended)
        return bool(has_down or has_up)

    def _target_is_stair(self) -> bool:
        return str(getattr(self, "_target_class", "")).strip().lower() in {"stair", "stairs", "staircase", "staircases"}

    def _stair_area_m2(self, mask_name: str) -> float:
        mask = getattr(self._omap, mask_name, None)
        if not isinstance(mask, np.ndarray):
            return 0.0
        ppm = float(getattr(self._omap, "pixels_per_meter", 1.0))
        return float(np.count_nonzero(mask)) / max(ppm * ppm, 1.0)

    def _target_stair_area_ready(self) -> bool:
        thresh = float(self._target_stair_area_threshold_m2)
        if self._stair_area_m2("_pending_downstair_map") >= thresh and hasattr(self, "_approve_pending_downstair"):
            self._approve_pending_downstair()
        return (
            self._stair_area_m2("_downstair_map") >= thresh and len(self._omap.downstair_frontier) > 0
        ) or (
            self._stair_area_m2("_upstair_map") >= thresh and len(self._omap.upstair_frontier) > 0
        )

    def _select_stair_review_records(self, max_images: int = 6):
        history = []
        for rec in self._iter_current_layer_mapping_history():
            rgb, _ = self._get_mapping_observation_data(rec)
            if rec.get("layer_idx") is None or not isinstance(rgb, np.ndarray):
                continue
            frustum_mask = self._refresh_mapping_record_frustum(rec)
            if not isinstance(frustum_mask, np.ndarray) or frustum_mask.size == 0:
                continue
            history.append(
                {
                    "rgb": rgb,
                    "frustum_mask": frustum_mask,
                }
            )
        if not history:
            return [{"rgb": rgb.copy(), "frustum_mask": None} for rgb in reversed(self._recent_step_rgbs[-max_images:])]

        selected = []
        covered = np.zeros_like(history[0]["frustum_mask"], dtype=bool)
        remaining = list(range(len(history)))
        while remaining and len(selected) < max_images:
            best_idx = remaining[0]
            best_gain = -1
            for idx in remaining:
                gain = int(np.count_nonzero(history[idx]["frustum_mask"] & ~covered))
                if gain > best_gain:
                    best_idx = idx
                    best_gain = gain
            selected.append(
                {
                    "rgb": history[best_idx]["rgb"].copy(),
                    "frustum_mask": history[best_idx]["frustum_mask"].astype(bool, copy=True),
                }
            )
            covered |= history[best_idx]["frustum_mask"]
            remaining.remove(best_idx)
        return selected


    def _stair_skip_vlm(self) -> bool:
        imgs_rgb: List[np.ndarray] = []
        review_records = self._select_stair_review_records(self._stair_review_image_count)
        fallback_rgbs = [rgb.copy() for rgb in reversed(self._recent_step_rgbs)]
        while len(review_records) < self._stair_review_image_count and fallback_rgbs:
            review_records.append({"rgb": fallback_rgbs.pop(0), "frustum_mask": None})
        imgs_rgb.extend(rec["rgb"] for rec in review_records if isinstance(rec.get("rgb"), np.ndarray))
        if not imgs_rgb:
            return False
        prompt = self.stair_skip_prompt.format(goal_object=self._vlm_target_name())
        self._handle_bev_exchange(imgs_rgb, prompt, has_bev=False)
        resp = str(self._vlm._get_response(tuple(imgs_rgb), prompt, failed_response="<answer> 0 </answer>")).lower()
        ans = resp.split("<answer>")[-1].split("</answer>")[0].strip() if "<answer>" in resp else resp.strip()
        self._handle_bev_exchange(imgs_rgb, prompt, resp, has_bev=False)
        return ans.startswith(("1", "t", "y"))
    
    def _skip_strategy(self, rgb: Optional[np.ndarray] = None, depth: Optional[np.ndarray] = None):
        self._update_persist_xy_history()
        if rgb is not None:
            self._recent_step_rgbs.append(rgb.copy())
            if isinstance(self._tf_camera_to_episodic, np.ndarray):
                self._recent_step_agent_tfs.append(self._tf_camera_to_episodic.copy())
            else:
                self._recent_step_agent_tfs.append(None)
            if len(self._recent_step_rgbs) > self._stair_review_image_count:
                self._recent_step_rgbs = self._recent_step_rgbs[-self._stair_review_image_count :]
                self._recent_step_agent_tfs = self._recent_step_agent_tfs[-self._stair_review_image_count :]

        self._maintain_tmp_ban_point()
        self._refresh_goal_observation_pool()
        if self._recently_stagnant():
            self._stagnant_ban_vlm()
        self._handle_stair_strategy()
