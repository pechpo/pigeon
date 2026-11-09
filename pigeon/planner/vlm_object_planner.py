from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import os

from .random_object_planner import RandomObjectPlanner
from ..models.vlm import VLMClient
from ..utils.geometry_utils import transform_points
from ..utils.viz import build_vlm_panel, mark_image_bgr
from ..utils.img_utils import generate_text_image


class VLMObjectPlanner(RandomObjectPlanner):
    """Frontier exploration guided by a VLM selection, inspired by DoubleZeroVLMPolicyV2.

    - Inherits detection + target pursuit from RandomObjectPlanner
    - While walking, maintains a memory for each frontier: RGB when seen, its pixel
      location in that RGB, and whether it still aligns with the current frontier set.
    - If a very nearby frontier exists, keep passby behavior. Otherwise select the
      closest `num_option` frontiers (to the agent) and query VLM to decide the next waypoint.
    - Frontiers are marked with red numeric circular markers in their images.
    - Candidate images with markers are shown on the right visualization panel.
    """

    models = ["pointnav_rnn", "owl_v2", "mobile_sam"]

    _frontier_selection_prompt_template = (
        """You are a robot and after exploration, you have the given images.\n"""
        """Every image represents a possible direction to find the {goal_object}.\n"""
        """Based on the numerical markers in the image, select one of these numbers to move next.\n"""
        """If you're not confident in moving to the marker to find the {goal_object}, you should return <answer> 0 </answer>.\n"""
        """Please note all closed doors cannot be opened.\n"""
        """Please follow the format like this, without bracket --\n"""
        """<thought> [put your step-by-step thinking process here] </thought>\n"""
        """<answer> [put a single marker id or 0 here] </answer>\n"""
    )

    _goal_rethink_prompt_template = (
        """You are an AI assistant guiding a robot navigating an indoor environment.\n"""
        """Your task is to help the robot find the {goal_object}.\n"""
        """You are given four sequential images from the robot's perspective:\n"""
        """- Image 1: Observation from 30 steps ago\n"""
        """- Image 2: Observation from 20 steps ago\n"""
        """- Image 3: Observation from 10 steps ago\n"""
        """- Image 4: Current observation (0 steps ago)\n"""
        """Based on this image sequence, analyze the robot's progress and decide on the next action.\n"""
        """Respond with a single digit (1, 2, or 3) representing your decision:\n"""
        """1 - Continue. It is not necessary to select a new navigation goal. No new, strong semantic clues related to the {goal_object} have appeared. (e.g., the robot is just moving down a hallway).\n"""
        """2 - Re-plan. It is better to select a new navigation goal. A strong semantic clue has appeared (e.g., sighting a related contextual object like a 'sofa' when looking for a 'remote'), suggesting a more promising area to explore.\n"""
        """3 - Unstuck. It is highly possible the robot is stuck (e.g., the images are barely changing despite recent steps of movement, or indicate that the robot is at a fixed location) or oscillating. The current navigation target should be banned, even despite the strong semantic clues for finding the target have shown in the images.\n"""
        """Please note all closed doors cannot be opened.\n"""
        """Please follow the format like this, without bracket --\n"""
        """<thought> [put your step-by-step thinking process here] </thought>\n"""
        """<answer> [put a single digit id or 0 here] </answer>\n"""
    )

    def __init__(self) -> None:
        super().__init__()
        # VLM client (model can be overridden by env var VLM_MODEL)
        self._vlm = VLMClient(os.environ.get("VLM_MODEL", "gpt-4o"))
        # Frontier memory entries: {xy (2,), rgb (HWC RGB), pixel(u,v), valid(bool), ts(int)}
        self._frontier_mem: List[Dict[str, Any]] = []
        self._mem_match_thresh: float = 0.5  # meters to consider same frontier
        self._num_option: int = 4
        self._use_grid: bool = False  # deprecated: always feed raw multi-image input to the VLM
        self._vlm_grid_bgr: Optional[np.ndarray] = None  # for right-panel viz
        self._vlm_bottom_text: Optional[str] = None  # for full-width bottom text row
        self._rethink_goal_type = "geometric"  # None / vlm / geometric
        self._stuck_threshold: float = 3.0  # meters over rethink interval to consider stuck (geometric)
        self._xy_history = np.empty((100, 2), dtype=float)
        self._xy_history_start = 0
        self._goal_rethink_interval: int = 30
        self._goal_history_rgbs: List[np.ndarray] = []
        self._goal_history_steps: int = 0
        self._last_goal_xy_for_rethink: Optional[np.ndarray] = None
        self._goal_path_since_last_rethink: float = 0.0
        self._goal_path_total: float = 0.0
        self._last_agent_xy_for_rethink: Optional[np.ndarray] = None
        self._tstep: int = 0
        self._downstair_detection = True
        self._upstair_detection = True
        self.multifloor = False

    def set_yolo_conf_thresh(self) -> None:
        self._yolo_conf_thresh = [0.80]  # target object
        if self._upstair_detection:
            self._yolo_conf_thresh.append(0.35)  # stairs

    def reset(self, reset_stairs=True) -> None:
        super().reset(reset_stairs)
        # Reset per-episode VLM call counter
        if hasattr(self, "_vlm") and hasattr(self._vlm, "reset_call_count"):
            self._vlm.reset_call_count()
        self._frontier_mem.clear()
        # Prepare right-panel space from the start (no text row; bottom text is global).
        try:
            self._vlm_grid_bgr = build_vlm_panel([], text=None)
        except Exception:
            self._vlm_grid_bgr = None
        self._vlm_bottom_text = ""
        self._goal_history_rgbs = []
        self._goal_history_steps = 0
        self._last_goal_xy_for_rethink = None
        self._goal_path_since_last_rethink = 0.0
        self._xy_history_start = 0
        self._goal_path_total = 0.0
        self._last_agent_xy_for_rethink = None
        self._tstep = 0
        self._last_vlm_decision = None

    def _reset_for_reexploration(self) -> None:
        candidate_names = list(getattr(self._objmap, "_candidate_object_names", []))
        target_name = self._vlm_target_name() if hasattr(self, "_vlm_target_name") else self._target_class
        self.reset()
        if candidate_names:
            self._objmap.set_candidate_objects(candidate_names, target_name=target_name)

    def _restart_single_floor_exploration(self, det_seg: Any) -> Dict[str, Any]:
        self._reset_for_reexploration()
        # The returned turn is the first step of the new init scan.
        self._init_steps = 1
        self._tstep += 1
        return self._build_action_result(
            "turn_left",
            extra={
                "det_seg": det_seg,
                "right_panel_viz": self._vlm_grid_bgr,
                "right_panel_text": self._get_bottom_text(),
            },
        )

    def _normalize_vlm_images(self, images: Any) -> List[np.ndarray]:
        if images is None:
            return []
        if isinstance(images, np.ndarray):
            return [images]
        out: List[np.ndarray] = []
        if isinstance(images, (list, tuple)):
            for item in images:
                out.extend(self._normalize_vlm_images(item))
                if len(out) >= 9:
                    break
        return out[:9]

    def _show_vlm_exchange(self, images: Any, text: str) -> None:
        imgs_rgb = self._normalize_vlm_images(images)
        self._vlm_grid_bgr = build_vlm_panel(imgs_rgb, text=None)
        self._vlm_bottom_text = text
        self._refresh_vlm_viz()

    def _handle_vlm_exchange(self, images: Any, prompt: str, response: Optional[str] = None) -> None:
        text = f"Prompt: {prompt}" if response is None else f"VLM: {response}"
        self._show_vlm_exchange(images, text)

    def _bind_confirmation_viz(self) -> None:
        objmap = getattr(self, "_objmap", None)
        if objmap is not None and hasattr(objmap, "set_vlm_viz_callback"):
            objmap.set_vlm_viz_callback(self._handle_vlm_exchange)

    def _final_check_before_target_stop(
        self, det_seg: Optional[Dict[str, Any]], has_obj: bool
    ) -> Optional[Dict[str, Any]]:
        """Subclasses may reject a target STOP and return a fallback action."""
        return None

    def _refresh_vlm_viz(self) -> None:
        self._update_viz(
            planner_viz=self._visualize_map(),
            right_panel_viz=self._vlm_grid_bgr,
            bottom_text=self._get_bottom_text(),
            refresh=True,
        )

    def _get_stair_nav_panel(self) -> Optional[np.ndarray]:
        try:
            stair_tile_bgr = generate_text_image(640, "stairnav")
            stair_tile_rgb = cv2.cvtColor(stair_tile_bgr, cv2.COLOR_BGR2RGB)
            return build_vlm_panel([stair_tile_rgb], text=None)
        except Exception:
            return self._vlm_grid_bgr

    def _get_active_right_panel_viz(self) -> Optional[np.ndarray]:
        if self._phase in {"stair_nav", "pointnav_descent", "post_stair_settle"}:
            return self._get_stair_nav_panel()
        return self._vlm_grid_bgr

    def _get_active_right_panel_text(self) -> str:
        if self._phase == "stair_nav":
            path_m = float(self._goal_path_since_last_rethink) if hasattr(self, "_goal_path_since_last_rethink") else 0.0
            return f"path: {path_m:.1f} m | stairnav"
        if self._phase == "pointnav_descent":
            return "pointnav descent"
        if self._phase == "post_stair_settle":
            return "post stair settle"
        return self._get_bottom_text()

    # ------------------------------ Frontier Memory ------------------------------
    def _project_to_uv(
        self,
        frontier_xy: np.ndarray,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        tf_camera_to_episodic: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int]]:
        """Project a single world-XY frontier into current RGB pixel (u, v).

        Returns None if projection is ill-conditioned, out of image bounds, or
        occluded when depth is provided.
        """
        try:
            tf_inv = np.linalg.inv(
                self._tf_camera_to_episodic if tf_camera_to_episodic is None else tf_camera_to_episodic
            )
        except Exception:
            return None
        floor_h = 0.0 #0.1  # 0.3
        pt3 = np.array([[frontier_xy[0], frontier_xy[1], floor_h]], dtype=np.float32)  # world coords
        cam_pt = transform_points(tf_inv, pt3)[0]  # local coords
        if cam_pt.shape[0] != 3:
            return None
        z, x, y = cam_pt.tolist()
        # Only annotate frontiers that lie in front of the camera. Back-facing points
        # can otherwise mirror into the image and create incorrect markers.
        if z <= 1e-3:
            return None
        h, w = rgb.shape[:2]
        fx = float(w) / (2.0 * np.tan(float(self._camera_fov_rad) / 2.0))
        fy = float(h) / (2.0 * np.tan(float(self._camera_fov_rad) / 2.0))
        cx, cy = w // 2, h // 2
        u = int(round(-x * fx / z + cx))  # pixel coords
        v = int(round(-y * fy / z + cy))
        radius = 9
        if u < radius or u >= (w - radius) or v < radius or v >= (h - radius):  # prevent corner label
            return None
        if depth is not None:  # occlusion check for later pixel supply
            d = depth[..., 0] if depth.ndim == 3 else depth
            if v >= d.shape[0] or u >= d.shape[1]:
                return None
            # Look at a small neighborhood to tolerate slight misalignments.
            r = 10
            v0, v1 = max(0, v - r), min(d.shape[0], v + r + 1)
            u0, u1 = max(0, u - r), min(d.shape[1], u + r + 1)
            patch = d[v0:v1, u0:u1]
            patch = patch * (self._max_depth - self._min_depth) + self._min_depth
            d_obs = float(np.max(patch))
            if d_obs <= self._min_depth:
                return None
            expected_range = float(np.linalg.norm(cam_pt))
            # Skip if the frontier is well beyond sensing range or clearly occluded.
            if expected_range > self._max_depth + 0.5:
                return None
            if expected_range > d_obs + 0.5:
                return None
        return (u, v)

    def _remember_frontiers(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None) -> None:
        """Record newly seen frontiers with their RGB and pixel position (one image per frontier).

        A new memory record is created only when no existing frontier within
        `self._mem_match_thresh` meters is present. Validity is refreshed by comparing
        to the current frontier set.
        """
        fr = self._frontiers
        if isinstance(fr, np.ndarray) and fr.size >= 2:
            ts = int(self._tstep)
            if len(self._frontier_mem) == 0:
                new_indices = range(fr.shape[0])
            else:
                mem_xy = np.stack([m["xy"] for m in self._frontier_mem], axis=0)
                d = np.linalg.norm(fr.reshape(-1, 1, 2) - mem_xy.reshape(1, -1, 2), axis=2)
                min_d = np.min(d, axis=1)
                new_indices = np.where(min_d >= float(self._mem_match_thresh))[0]
            for i in new_indices:
                xy = fr[i].astype(np.float32)
                uv = self._project_to_uv(xy, rgb, depth)
                self._frontier_mem.append(
                    {
                        "xy": xy.copy(),
                        "rgb": rgb.copy(),
                        "pixel": uv if uv is not None else None,
                        "valid": True,
                        "ts": ts,
                    }
                )

        # Refresh validity against current frontier set
        fr_now = self._frontiers if isinstance(self._frontiers, np.ndarray) else np.zeros((0, 2), dtype=np.float32)
        for m in self._frontier_mem:
            if fr_now.size == 0:
                m["valid"] = False
            else:
                m["valid"] = bool(
                    np.min(np.linalg.norm(fr_now - m["xy"].reshape(1, 2), axis=1)) < self._mem_match_thresh
                )
            # If we saw this frontier again and it previously had no pixel, try to fill it now.
            if m.get("valid", False) and m.get("pixel", None) is None:
                uv = self._project_to_uv(m["xy"], rgb, depth)
                if uv is not None:
                    m["pixel"] = uv
                    m["rgb"] = rgb.copy()

        # Cap memory size
        if len(self._frontier_mem) > 256:
            self._frontier_mem = self._frontier_mem[-256:]

    # ------------------------------ Marker + VLM ------------------------------
    def _build_marked_candidates(self, mem_list: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], np.ndarray]:
        imgs_rgb: List[np.ndarray] = []
        uv_arr = []
        for i, m in enumerate(mem_list, start=1):
            if m.get("pixel", None) is None:
                # Skip entries without valid pixel for marking
                continue
            bgr = cv2.cvtColor(m["rgb"], cv2.COLOR_RGB2BGR)
            u, v = int(m["pixel"][0]), int(m["pixel"][1])
            bgr = mark_image_bgr(bgr, i, u, v, radius=9)
            imgs_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            uv_arr.append([u, v])
        return imgs_rgb, np.array(uv_arr, dtype=np.int32)

    def _vlm_select(self, candidates: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if len(candidates) == 0:
            return None, None
        imgs_rgb, _ = self._build_marked_candidates(candidates)
        prompt = self._frontier_selection_prompt_template.format(goal_object=self._vlm_target_name())
        self._last_vlm_decision = {
            "type": "frontier_selection",
            "step": int(self._tstep),
            "prompt": prompt,
            "candidates": [
                {
                    "xy": self._trace_candidate_xy(candidate),
                    "pixel": candidate.get("pixel"),
                    "timestamp": candidate.get("ts"),
                    "valid": candidate.get("valid"),
                }
                for candidate in candidates
            ],
            "selected_index": None,
            "selected_xy": None,
            "response": None,
        }
        self._handle_vlm_exchange(imgs_rgb, prompt)
        resp = self._vlm._get_response(tuple(imgs_rgb), prompt, failed_response="<answer> 0 </answer>")
        # After receiving response, update the panel to show reply text instead of prompt
        try:
            reply_text = str(resp)
        except Exception:
            reply_text = ""
        self._handle_vlm_exchange(imgs_rgb, prompt, reply_text)
        try:
            sel = int(resp.split("<answer>")[1].split("</answer>")[0].strip())
        except Exception:
            sel = 0
        self._last_vlm_decision["selected_index"] = int(sel)
        self._last_vlm_decision["response"] = reply_text
        self._record_trace_event(
            "vlm_decision",
            decision_type="frontier_selection",
            selected_index=int(sel),
            candidate_count=int(len(candidates)),
            candidate_xy=[
                self._trace_candidate_xy(candidate) for candidate in candidates
            ],
            response=reply_text[-1000:],
        )
        if sel <= 0 or sel > len(candidates):
            return None, self._vlm_grid_bgr
        chosen = candidates[sel - 1]
        self._last_vlm_decision["selected_xy"] = self._trace_candidate_xy(chosen)
        return chosen["xy"].astype(np.float32), self._vlm_grid_bgr

    @staticmethod
    def _trace_candidate_xy(candidate: Dict[str, Any]) -> Optional[List[float]]:
        value = candidate.get("xy") if isinstance(candidate, dict) else None
        try:
            return np.asarray(value, dtype=np.float32).reshape(-1)[:2].tolist()
        except Exception:
            return None

    # ------------------------------ Goal Rethink ------------------------------
    def _update_goal_history(self, rgb: Optional[np.ndarray]) -> None:
        self._xy_history[self._tstep % len(self._xy_history)] = self._curr_xy
        if (self._tstep + 1 - self._xy_history_start) % len(self._xy_history) == 0:
            self._xy_history_start += 1
        if rgb is None or self._goal_xy is None:
            return

        goal_xy = self._goal_xy
        if not isinstance(goal_xy, np.ndarray):
            goal_xy = np.asarray(goal_xy, dtype=np.float32)

        is_new_goal = False
        if self._last_goal_xy_for_rethink is None:
            is_new_goal = True
        else:
            try:
                is_new_goal = bool(self._distance(self._last_goal_xy_for_rethink, goal_xy) > self._same_goal_thresh + 0.25)
            except Exception:
                is_new_goal = True

        if is_new_goal:
            self._last_goal_xy_for_rethink = goal_xy.copy()
            self._goal_history_rgbs = []
            self._goal_history_steps = 0
            self._goal_path_since_last_rethink = 0.0
            self._xy_history_start = self._tstep
            self._goal_path_total = 0.0
            self._last_agent_xy_for_rethink = self._curr_xy.copy()
            # Treat the current frame as step 0 for this goal.
            self._goal_history_rgbs.append(rgb.copy())
            return

        # Accumulate geometric path length for the current goal.
        if self._last_agent_xy_for_rethink is None:
            self._last_agent_xy_for_rethink = self._curr_xy.copy()
        else:
            try:
                step_dist = float(np.linalg.norm(self._curr_xy - self._last_agent_xy_for_rethink))
                self._goal_path_since_last_rethink += step_dist
                self._goal_path_total += step_dist
            except Exception:
                pass
            self._last_agent_xy_for_rethink = self._curr_xy.copy()

        self._goal_history_steps += 1
        if self._goal_history_steps % 10 == 0:
            self._goal_history_rgbs.append(rgb.copy())
            if len(self._goal_history_rgbs) > 4:
                self._goal_history_rgbs = self._goal_history_rgbs[-4:]

    def _vlm_goal_rethink(self, imgs_rgb: List[np.ndarray]) -> int:
        prompt = self._goal_rethink_prompt_template.format(goal_object=self._vlm_target_name())
        self._last_vlm_decision = {
            "type": "goal_rethink",
            "step": int(self._tstep),
            "prompt": prompt,
            "response": None,
            "decision": None,
        }
        self._handle_vlm_exchange(imgs_rgb, prompt)
        try:
            resp = self._vlm._get_response(tuple(imgs_rgb), prompt, failed_response="<answer> 1 </answer>")
        except Exception:
            self._last_vlm_decision["response"] = "exception"
            return 1

        try:
            reply_text = str(resp)
        except Exception:
            reply_text = ""
        self._handle_vlm_exchange(imgs_rgb, prompt, reply_text)

        try:
            val = int(reply_text.split("<answer>")[1].split("</answer>")[0].strip())
        except Exception:
            val = 1
        if val not in (1, 2, 3):
            return 1
        self._last_vlm_decision["response"] = reply_text
        self._last_vlm_decision["decision"] = int(val)
        self._record_trace_event(
            "vlm_decision",
            decision_type="goal_rethink",
            decision=int(val),
            response=reply_text[-1000:],
        )
        return val

    def _get_bottom_text(self) -> str:
        path_m = float(self._goal_path_since_last_rethink) if hasattr(self, "_goal_path_since_last_rethink") else 0.0
        path_str = f"path: {path_m:.1f} m"
        if not self._vlm_bottom_text:
            return path_str
        return f"{path_str} | {self._vlm_bottom_text}"

    def _get_xy_history(self, gap: int = 50):
        l = len(self._xy_history)
        gap = min(self._tstep - self._xy_history_start, gap)
        start, end = self._xy_history_start % l, (self._tstep - gap + l) % l
        if start <= end:
            return self._xy_history[start:end]
        return np.concatenate([self._xy_history[start:], self._xy_history[:end]], axis=0)

    def _maybe_rethink_current_goal(self) -> None:
        if self._rethink_goal_type not in ("vlm", "geometric"):
            return
        if self._goal_xy is None:
            return
        if self._goal_history_steps <= 0:
            return
        if self._goal_history_steps % self._goal_rethink_interval != 0:
            return

        if self._rethink_goal_type == "geometric":
            xy_hist = self._get_xy_history()
            if len(xy_hist) and np.linalg.norm(xy_hist - self._curr_xy, axis=-1).min() < 0.2:
                self._ban_point(self._goal_xy, reason="geometric_stagnation")
                self._goal_xy = None
                if hasattr(self._omap, "set_selected_frontier"):
                    self._omap.set_selected_frontier(None)
                self._should_reset_pointnav = True
                self._phase = "explore"
                self._goal_history_rgbs = []
                self._goal_history_steps = 0
                self._last_goal_xy_for_rethink = None
                self._xy_history_start = self._tstep
                self._xy_history[self._tstep % len(self._xy_history)] = self._curr_xy
            # Reset path accumulator for the next interval (stuck or not).
            self._goal_path_since_last_rethink = 0.0
            self._last_agent_xy_for_rethink = self._curr_xy.copy()
            return

        if len(self._goal_history_rgbs) < 4:
            return

        imgs_rgb = self._goal_history_rgbs[-4:]
        decision = self._vlm_goal_rethink(imgs_rgb)
        if decision == 1:
            return

        if decision == 2:
            if self._phase == "explore":
                prev_goal = self._goal_xy.copy()
                self._choose_goal_vlm_with_trace()
                if self._goal_xy is not None:
                    try:
                        if self._distance(prev_goal, self._goal_xy) > self._same_goal_thresh:
                            self._should_reset_pointnav = True
                    except Exception:
                        self._should_reset_pointnav = True
            return

        if decision == 3:
            self._ban_point(self._goal_xy, reason="vlm_goal_rethink")
            self._goal_xy = None
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)
            self._should_reset_pointnav = True
            self._phase = "explore"
            self._goal_history_rgbs = []
            self._goal_history_steps = 0
            self._last_goal_xy_for_rethink = None
            return

    def _choose_goal_vlm(self, allow_spin=True) -> None:
        """Passby-first, then VLM selection with fallbacks."""
        fr = self._frontiers
        if isinstance(fr, np.ndarray) and fr.size >= 2:
            dists = np.linalg.norm(fr - self._curr_xy.reshape(1, 2), axis=1)
            near_idx = int(np.argmin(dists))
            if float(dists[near_idx]) <= float(self._passby_thresh):
                self._last_goal_selection_source = "nearby_frontier_passby"
                self._goal_xy = fr[near_idx].astype(np.float32)
                self._omap.set_selected_frontier(self._goal_xy)
                return

        # First selection: prioritize candidates by distance to the agent,
        # instead of sorting by step (timestamp).
        valid_mem = [m for m in self._frontier_mem if m.get("valid", False)]
        valid_mem.sort(
            key=lambda m: float(np.linalg.norm(np.asarray(m["xy"], dtype=np.float32) - self._curr_xy)),
        )
        # Pick the nearest memories that have usable pixels; keep grabbing until we fill num_option.
        pixel_mem = [m for m in valid_mem if m.get("pixel") is not None]
        if len(pixel_mem) == 1:
            self._last_goal_selection_source = "single_frontier_memory"
            goal_xy = pixel_mem[0]["xy"].astype(np.float32)
        else:
            self._last_goal_selection_source = "vlm_frontier_selection"
            goal_xy, _ = self._vlm_select(pixel_mem[: self._num_option])
        if allow_spin and goal_xy is None:
            self._last_goal_selection_source = "spin_after_no_vlm_choice"
            self._phase = "spin"
            self._steps_to_spin = 12
            return
        if goal_xy is None and len(pixel_mem) > self._num_option:
            self._last_goal_selection_source = "vlm_frontier_resample"
            idxs = np.random.choice(len(pixel_mem), size=min(self._num_option, len(pixel_mem)), replace=False)
            goal_xy, _ = self._vlm_select([pixel_mem[i] for i in idxs])
        if goal_xy is None and isinstance(fr, np.ndarray) and fr.size >= 2:
            self._last_goal_selection_source = "nearest_frontier_fallback"
            dists = np.linalg.norm(fr - self._curr_xy.reshape(1, 2), axis=1)
            goal_xy = fr[int(np.argmin(dists))].astype(np.float32)
        self._goal_xy = goal_xy
        if hasattr(self._omap, "set_selected_frontier"):
            self._omap.set_selected_frontier(self._goal_xy)
    
    def _skip_strategy(self, rgb: Optional[np.ndarray] = None, depth: Optional[np.ndarray] = None):
        # reserved for some strategy who skip frontiers
        pass

    # ------------------------------ Act ------------------------------
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._start_layer_step()
        rgb = obs.get("rgb")
        depth = obs.get("depth")
        gps = obs.get("gps")
        yaw = obs.get("yaw")

        # Precompute agent pose and transform once per step.
        self._set_agent_pose(gps, yaw)

        # Register this frame before mapping records its frustum in history.
        # Both histories then refer to the same image, depth, and camera pose.
        if self._phase != "stair_nav":
            self._objmap.register_image(rgb, self._tf_camera_to_episodic, depth)

        # Update obstacle/frontier map and object maps

        if self._phase != "stair_nav":
            self._update_mapping(depth, rgb)
            self._objmap.update_explored(self._tf_camera_to_episodic, float(self._max_depth), float(self._camera_fov_rad))
            if not self.multifloor and hasattr(self._omap, "approve_pending_downstair"):
                self._omap.approve_pending_downstair()
            # Maintain frontier memory during walking
            self._remember_frontiers(rgb, depth)
        else:
            self._update_stair_nav_height_map(depth)

        # If the current navigation goal has been banned, immediately clear it
        # so that a new (valid) frontier will be selected this step.
        if self._goal_xy is not None and self._should_enforce_goal_ban() and self.is_banned(self._goal_xy):
            self._goal_xy = None
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)
            # Reset pointnav state when switching away from a banned goal
            self._should_reset_pointnav = True

        det_seg = None
        if self._phase != "stair_nav":
            # Attempt detection each step
            det_seg = self._detect_and_update(rgb, depth)
            self._refresh_detection_viz(
                det_seg=det_seg,
                right_panel_viz=self._vlm_grid_bgr,
                bottom_text=self._get_bottom_text(),
            )
            self._skip_strategy(rgb, depth)
        has_obj = bool(self._target_class) and self._objmap.has_object(self._target_class)

        if self.multifloor and self._phase != "target":
            frontier_count = len(self._frontiers) + len([m for m in self._frontier_mem if m.get("valid", False)])
            out = self.handle_stairs(
                # Once stair handling is explicitly selected, do not let same-floor
                # frontier availability suppress the stair action.
                update_only=(
                    self._phase not in ["nav_to_stair", "pointnav_descent", "stair_nav", "post_stair_settle"]
                    and frontier_count > 0
                ),
                depth=depth,
                extra={
                    "det_seg": det_seg,
                    "has_obj": has_obj,
                    "target_detected": has_obj,
                    "right_panel_viz": self._get_active_right_panel_viz(),
                    "right_panel_text": self._get_active_right_panel_text(),
                },
            )
            if out is not None:
                self._tstep += 1
                return out

        # Initial spin
        if self._phase == "init" and self._init_steps < self._max_init_steps and self._phase != "target":
            self._init_steps += 1
            out = {
                "action": "turn_left",
                "map_viz": self._visualize_map(),
                "right_panel_viz": self._vlm_grid_bgr,  # dedicated VLM panel
                "right_panel_text": self._get_bottom_text(),
                "det_seg": det_seg,
                "nav_goal": self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None,
                "stop_called": False,
                "target_detected": has_obj,
            }
            if self._init_steps >= self._max_init_steps:
                self._phase = "explore"
            self._tstep += 1
            return out

        # Target phase: pursue object if present
        if self._phase == "target":
            num_frontiers = len(self._frontiers) + len([m for m in self._frontier_mem if m.get("valid", False)])
            if has_obj or num_frontiers == 0:
                tgt_xy = self._objmap.get_best_object(
                    self._target_class,
                    self._curr_xy,
                    ignore_threshold=not has_obj,
                )
                if tgt_xy is not None and len(tgt_xy) == 2:
                    tgt_xy_arr = np.asarray(tgt_xy, dtype=np.float32)
                    if not self.is_banned(tgt_xy_arr):
                        self._goal_xy = tgt_xy_arr
                        self._update_goal_history(rgb)
                        self._maybe_rethink_current_goal()
                        if self._goal_xy is None or self._phase != "target":
                            has_obj = False
                        else:
                            if self._near_goal(self._goal_proximity_thresh):
                                final_check_action = self._final_check_before_target_stop(det_seg, has_obj)
                                if final_check_action is not None:
                                    self._tstep += 1
                                    return final_check_action
                                self._record_target_stop(det_seg, has_obj)
                                self._tstep += 1
                                return {
                                    "action": "stop",
                                    "map_viz": self._visualize_map(),
                                    "right_panel_viz": self._vlm_grid_bgr,
                                    "right_panel_text": self._get_bottom_text(),
                                    "det_seg": det_seg,
                                    "nav_goal": (
                                        self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None
                                    ),
                                    "stop_called": True,
                                    "target_detected": has_obj,
                                }
                            out = self._pointnav_step(
                                depth,
                                self._goal_xy,
                                extra={
                                    "det_seg": det_seg,
                                    "target_detected": has_obj,
                                    "right_panel_viz": self._vlm_grid_bgr,
                                    "right_panel_text": self._vlm_bottom_text,
                                },
                            )
                            self._tstep += 1
                            return out
            # Lost target -> explore
            self._phase = "explore"
            self._goal_xy = None
            self._omap.set_selected_frontier(None)
            self._should_reset_pointnav = True

        if not self.multifloor and self._phase in {"explore", "spin", "escape"}:
            has_frontier = (
                self._has_frontier_option()
                if hasattr(self, "_has_frontier_option")
                else len(self._frontiers) > 0
            )
            if not has_frontier:
                candidate_xy = (
                    self._objmap.get_best_object(self._target_class, self._curr_xy, ignore_threshold=True)
                    if self._target_class
                    else None
                )
                if (
                    isinstance(candidate_xy, np.ndarray)
                    and candidate_xy.size == 2
                    and not self.is_banned(candidate_xy)
                ):
                    self._goal_xy = np.asarray(candidate_xy, dtype=np.float32)
                    self._phase = "target"
                    self._should_reset_pointnav = True
                else:
                    return self._restart_single_floor_exploration(det_seg)

        stair_out = self._maybe_handle_stairs_or_escape(
            depth,
            extra={
                "det_seg": det_seg,
                "has_obj": has_obj,
                "target_detected": has_obj,
                "right_panel_viz": self._get_active_right_panel_viz(),
                "right_panel_text": self._get_active_right_panel_text(),
            },
        )
        if stair_out is not None:
            self._tstep += 1
            return stair_out

        act = self._maybe_spin(extra={"det_seg": det_seg, "target_detected": has_obj})
        if act is not None:
            self._tstep += 1
            return act

        # If the current navigation goal has become an obstacle in the
        # obstacle map (e.g., because staircase protection just marked
        # it as unsafe), drop and ban it so we replan instead of walking
        # into an occupied cell.
        if self._goal_xy is not None and self._goal_is_occupied():
            self._ban_point(self._goal_xy, reason="occupied_goal")
            self._goal_xy = None
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)
            self._should_reset_pointnav = True

        # Exploration: simplified - passby + VLM handled inside chooser
        if self._goal_xy is None:
            self._choose_goal_vlm_with_trace()
        else:
            if self._near_goal(self._proximity_threshold):
                prev_goal = self._goal_xy.copy()
                self._choose_goal_vlm_with_trace()
                if self._goal_xy is not None:
                    self._should_reset_pointnav = self._distance(prev_goal, self._goal_xy) > self._same_goal_thresh

        if self._goal_xy is not None:
            self._update_goal_history(rgb)
            self._maybe_rethink_current_goal()
            # If rethink decides to ban the current goal (decision=3),
            # try to immediately pick a new frontier goal instead of stopping.
            if self._goal_xy is None and self._phase == "explore":
                self._choose_goal_vlm_with_trace()

        act = self._maybe_spin(extra={"det_seg": det_seg, "target_detected": has_obj})
        if act is not None:
            self._tstep += 1
            return act

        if self._goal_xy is None:
            if not self.multifloor:
                return self._restart_single_floor_exploration(det_seg)
            self._tstep += 1
            return {
                "action": "stop",
                "map_viz": self._visualize_map(),
                "right_panel_viz": self._vlm_grid_bgr,
                "right_panel_text": self._get_bottom_text(),
                "det_seg": det_seg,
                "nav_goal": None,
                "stop_called": True,
                "target_detected": has_obj,
            }

        # PointNav towards selected goal
        out = self._pointnav_step(
            depth,
            self._goal_xy,
            extra={
                "det_seg": det_seg,
                "target_detected": has_obj,
                "right_panel_viz": self._vlm_grid_bgr,
                "right_panel_text": self._get_bottom_text(),
            },
        )
        self._tstep += 1
        return out
