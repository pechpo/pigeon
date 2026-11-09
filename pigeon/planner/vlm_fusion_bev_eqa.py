import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .vlm_object_planner import VLMObjectPlanner
from .vlm_fusion_bev_planner import VLMFusionBEVPlanner


class VLMFusionBEVEQAPlanner(VLMFusionBEVPlanner):
    """BEV fusion planner variant for embodied question answering.

    The planner still uses an extracted core object to drive detection and
    viewpoint search, but all VLM-facing navigation prompts are phrased around
    the episode question. When a VLM call finds enough visual evidence to answer
    the question, it emits an EQA answer and the planner stops the episode.
    """

    is_eqa_planner = True

    question_object_extraction_prompt_template = (
        "You help an indoor robot answer embodied QA questions.\n"
        "Given the episode question, extract the single core physical object or place "
        "that the robot should search for first. Use a short detector-friendly noun phrase, "
        "for example: ceiling fan, bed, painting, table, sink, refrigerator, door, clock.\n"
        "If the question is about a room-level property, return the most relevant room or object.\n"
        "Please follow the format:\n"
        "<thought> [brief reason] </thought>\n"
        "<answer> [core object noun phrase] </answer>\n"
        "Episode question: {question}\n"
    )

    eqa_confirmation_prompt_template = (
        "You are an image recognition assistant helping an indoor robot answer an embodied QA question.\n"
        "The target question is: {goal_object}\n"
        "You will be presented with an object's image captured from single or multiple views.\n"
        "First, decide whether the image(s) contain enough visual evidence to answer the target question. "
        "If they do, include a concise answer in the exact form [EQA]answer[/EQA]. "
        "Always include an [EQA]...[/EQA] line; leave it empty as [EQA][/EQA] if the image(s) do not answer the question.\n"
        "Also check whether the visual evidence is relevant to the target question. "
        "Please do not confuse the needed evidence with these related detector classes: {object_list}.\n"
        "Respond <answer> True </answer> if the needed evidence is clearly present indoors, "
        "<answer> False </answer> if it is clearly not present, "
        "<answer> Unconfirmed </answer> if unsure, or <answer> Outdoor </answer> if it is outdoors.\n"
        "Please follow the format:\n"
        "<thought> [brief visual reasoning] </thought>\n"
        "<answer> [True, False, Unconfirmed or Outdoor] </answer>\n"
        "[EQA][concise QA answer, or empty if not answerable][/EQA]\n"
    )

    stagnant_ban_prompt = (
        "You are helping an indoor robot decide whether to temporarily ban its current navigation goal.\n"
        "Image 1 is the current BEV map.\n"
        "Image 2 is the current color BEV map.\n"
        "Images 3 to 6 are recent first-person views related to the current goal.\n"
        "The robot has been stagnant near this goal.\n"
        "The target question is: {goal_object}\n"
        "If any first-person image contains enough visual evidence to answer the target question, "
        "include the concise answer as [EQA]answer[/EQA].\n"
        "When visible, the same current goal is marked in the first-person views by a red circular marker labeled 1.\n"
        "Reply 1 if this goal region should be banned, otherwise reply 0.\n"
        "The goal is a red point on the BEV and colored BEV map.\n"
        "Large green circle near the red point indicates that the robot fails to approach the goal point.\n"
        "If the red-point goal seems irrelevant to the target question or clearly unreachable, ban the goal.\n"
        "Please follow the format:\n"
        "<thought> [reasoning content] </thought>\n"
        "<answer> [1 or 0] </answer>\n"
        "[EQA] [only if answerable, concise QA answer] [/EQA]\n"
    )
    stair_skip_prompt = (
        "You are helping an indoor robot decide whether it should leave the current floor and search via stairs.\n"
        "Images 1 to 6 are first-person views selected to summarize covered observations on the current floor.\n"
        "The target question is: {goal_object}\n"
        "If any image contains enough visual evidence to answer the target question, "
        "include the concise answer as [EQA]answer[/EQA].\n"
        "Reply 1 if the robot should go search on another floor now, otherwise reply 0.\n"
        "Observations leading to dark regions or dark hallways which lacks data don't represent the environment undiscovered.\n"
        "Even a short flight of stairs can create a barrier between different levels\n"
        "Please follow the format:\n"
        "<thought> [reasoning content] </thought>\n"
        "<answer> [1 or 0] </answer>\n"
        "[EQA] [only if answerable, concise QA answer] [/EQA]\n"
    )
    stair_frontier_selection_prompt_template = (
        "You are helping an indoor robot choose a staircase entry to leave the current floor.\n"
        "You are given first-person observations of up to 4 candidate staircase-entry frontiers.\n"
        "The same numbered marker can appear in multiple images and always refers to the same candidate frontier.\n"
        "Some candidates may have fewer observations than others.\n"
        "The target question is: {goal_object}\n"
        "If any image contains enough visual evidence to answer the target question, "
        "include the concise answer as [EQA]answer[/EQA].\n"
        "Select the best numbered staircase-entry frontier to move to next.\n"
        "You must choose a marker which is located at the edge of the staircase, so that you can go up or down immediately after reaching it.\n"
        "If none looks reliable, return <answer> 0 </answer>.\n"
        "Please follow the format:\n"
        "<thought> [reasoning content] </thought>\n"
        "<answer> [marker id or 0] </answer>\n"
        "[EQA] [only if answerable, concise QA answer] [/EQA]\n"
    )
    target_stair_frontier_selection_prompt_template = (
        "You are helping an indoor robot choose a navigation frontier near a stair.\n"
        "You are given first-person observations of up to 4 candidate frontiers.\n"
        "The same numbered marker can appear in multiple images and always refers to the same candidate frontier.\n"
        "Some candidates may have fewer observations than others.\n"
        "The target question is: {goal_object}\n"
        "If any image contains enough visual evidence to answer the target question, "
        "include the concise answer as [EQA]answer[/EQA].\n"
        "Select the best numbered frontier that is near the stair.\n"
        "If none looks reliable, return <answer> 0 </answer>.\n"
        "Please follow the format:\n"
        "<thought> [reasoning content] </thought>\n"
        "<answer> [marker id or 0] </answer>\n"
        "[EQA] [only if answerable, concise QA answer] [/EQA]\n"
    )
    frontier_selection_bev_prompt_template = (
        "You are a robot and after exploration, you have the given images.\n"
        "You are given first-person observations of candidate frontiers.\n"
        "The same numbered marker can appear in multiple images and always refers to the same candidate frontier.\n"
        "Some candidates may have fewer observations than others.\n"
        "The target question is: {goal_object}\n"
        "If any candidate image contains enough visual evidence to answer the target question, "
        "include the concise answer as [EQA]answer[/EQA].\n"
        "Each candidate view represents a possible direction to find evidence for answering the target question.\n"
        "Based on the numerical markers in the images, select one of these numbers to move next.\n"
        "If you're not confident in moving to the marker to answer the target question, you should return <answer> 0 </answer>.\n"
        "Please note all closed doors cannot be opened.\n"
        "Please don't select frontiers leading to dark regions or dark hallways which lacks data.\n"
        "Please select frontiers which is on the ground rather than those floating in air.\n"
        "Please follow the format like this, without bracket --\n"
        "<thought> [put your step-by-step thinking process here] </thought>\n"
        "<answer> [put a single marker id or 0 here] </answer>\n"
        "[EQA] [only if answerable, concise QA answer] [/EQA]\n"
    )

    def __init__(self) -> None:
        super().__init__()
        self._episode_question: Optional[str] = None
        self._episode_question_id: Optional[str] = None
        self._core_target_object: Optional[str] = None
        self._eqa_answer: Optional[str] = None
        self._eqa_answer_source: Optional[str] = None
        self._setup_target_name_override: Optional[str] = None
        self._post_stair_settle_max_steps = 30
        self._install_eqa_confirmation_prompt()

    def reset(self, reset_stairs: bool = True) -> None:
        super().reset(reset_stairs)
        self._eqa_answer = None
        self._eqa_answer_source = None
        self._install_eqa_confirmation_prompt()

    def set_episode_metadata(self, question: Optional[str] = None, question_id: Optional[str] = None) -> None:
        if isinstance(question, str) and question.strip():
            self._episode_question = question.strip()
        if question_id is not None:
            self._episode_question_id = str(question_id)

    def set_target_object(self, name: str) -> None:
        question = str(name or "").strip()
        self._episode_question = question
        core_object = self._extract_core_object(question)
        self._core_target_object = core_object
        self._setup_target_name_override = core_object
        try:
            super().set_target_object(core_object)
        finally:
            self._setup_target_name_override = None
        self._install_eqa_confirmation_prompt()
        self._refresh_eqa_confirmation_prompt()

    def _vlm_target_name(self) -> str:
        if self._setup_target_name_override:
            return self._setup_target_name_override.replace("|", " or ")
        if isinstance(self._episode_question, str) and self._episode_question.strip():
            return self._episode_question.replace("|", " or ")
        return super()._vlm_target_name()

    def _install_eqa_confirmation_prompt(self) -> None:
        objmap = getattr(self, "_objmap", None)
        if objmap is not None and hasattr(objmap, "_confirmation_prompt_template"):
            objmap._confirmation_prompt_template = self.eqa_confirmation_prompt_template

    def _refresh_eqa_confirmation_prompt(self) -> None:
        objmap = getattr(self, "_objmap", None)
        candidate_names = getattr(objmap, "_candidate_object_names", None)
        if objmap is None or not hasattr(objmap, "set_candidate_objects") or not candidate_names:
            return
        objmap.set_candidate_objects(candidate_names, target_name=self._vlm_target_name())

    def _extract_core_object(self, question: str) -> str:
        fallback = question or "object"
        if not question or self._vlm is None:
            return fallback
        prompt = self.question_object_extraction_prompt_template.format(question=question)
        resp = self._vlm._get_response_text(prompt, failed_response=f"<answer> {fallback} </answer>")
        text = str(resp or "")
        try:
            answer = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        except Exception:
            answer = text.strip()
        answer = re.sub(r"^[\"'`\[]+|[\"'`\].]+$", "", answer).strip()
        return answer or fallback

    def _parse_eqa_answer(self, text: Any, source: str) -> Optional[str]:
        if self._eqa_answer:
            return self._eqa_answer
        match = re.search(r"\[EQA\](.*?)\[/EQA\]", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            return None
        answer = re.sub(r"\s+", " ", match.group(1)).strip()
        answer = answer.strip("[] ")
        if not answer or answer.lower().startswith("only if answerable"):
            return None
        self._eqa_answer = answer
        self._eqa_answer_source = source
        return answer

    def _parse_latest_vlm_eqa_answer(self, source: str) -> Optional[str]:
        vlm = getattr(self, "_vlm", None)
        return self._parse_eqa_answer(getattr(vlm, "last_response", None), source)

    def _fallback_eqa_answer(self) -> str:
        target = str(self._core_target_object or "").strip()
        question = str(self._episode_question or "").strip()
        if target and target.lower() != question.lower() and len(target) <= 80:
            return f"I could not find the {target} needed to answer the question."
        return "I could not find enough visual evidence to answer the question."

    def _eqa_stop_result(self, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = dict(result or {})
        out["action"] = "stop"
        out["stop_called"] = True
        out["target_detected"] = False
        out["eqa_answer"] = self._eqa_answer or ""
        out["eqa_answer_source"] = self._eqa_answer_source
        out["question"] = self._episode_question
        out["question_id"] = self._episode_question_id
        if "map_viz" not in out:
            out["map_viz"] = self._visualize_map()
        if "right_panel_viz" not in out:
            out["right_panel_viz"] = self._vlm_grid_bgr
        if "right_panel_text" not in out:
            out["right_panel_text"] = self._get_bottom_text()
        return out

    def _detect_and_update(self, rgb: np.ndarray, depth: np.ndarray) -> Optional[Dict[str, Any]]:
        prev_phase = getattr(self, "_phase", None)
        prev_goal_xy = self._goal_xy.copy() if isinstance(getattr(self, "_goal_xy", None), np.ndarray) else None
        prev_goal_view_facing = (
            self._goal_view_facing.copy()
            if isinstance(getattr(self, "_goal_view_facing", None), np.ndarray)
            else None
        )
        prev_should_reset_pointnav = bool(getattr(self, "_should_reset_pointnav", False))
        det_seg = super()._detect_and_update(rgb, depth)
        self._parse_latest_vlm_eqa_answer("object_confirmation")

        # EQA stops only when the question has been answered. Reliable object
        # confidence should not switch this planner into ObjectNav target pursuit.
        # Preserve the existing exploration goal; otherwise every confirmed
        # detection clears the goal and forces frontier reselection each step.
        if getattr(self, "_phase", None) == "target":
            self._phase = prev_phase if prev_phase != "target" else "explore"
            self._goal_xy = prev_goal_xy
            self._goal_view_facing = prev_goal_view_facing
            self._should_reset_pointnav = prev_should_reset_pointnav
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(self._goal_xy)
        return det_seg

    def _choose_goal_vlm(self, allow_spin: bool = True) -> None:
        suspects, centers = self._objmap.get_suspected_view_positions(self._target_class, self._curr_xy)
        if isinstance(suspects, np.ndarray) and suspects.size >= 2:
            candidates = [(p, c) for p, c in zip(suspects, centers) if not self.is_banned(p)]
            if candidates:
                if self._goal_xy is not None and not self.is_banned(self._goal_xy) and self._goal_view_facing is not None:
                    return
                self._goal_view_facing = None
                pos, center = min(candidates, key=lambda pc: self._distance(pc[0], self._curr_xy))
                self._goal_xy = np.asarray(pos, dtype=np.float32)
                if isinstance(center, np.ndarray) and center.size >= 2 and not np.any(np.isnan(center)):
                    self._goal_view_facing = center.astype(np.float32)
                if hasattr(self._omap, "set_selected_frontier"):
                    self._omap.set_selected_frontier(self._goal_xy)
                return

        self._goal_view_facing = None
        num_frontiers = len(self._frontiers) + len([m for m in self._frontier_mem if m.get("valid", False)])
        if num_frontiers > 0:
            VLMObjectPlanner._choose_goal_vlm(self, allow_spin=False)
        elif allow_spin:
            self._phase = "spin"
            self._steps_to_spin = 12

        self._parse_latest_vlm_eqa_answer("goal_selection")
        if getattr(self, "_phase", None) == "target":
            self._phase = "explore"
            self._goal_xy = None
            self._goal_view_facing = None
            self._should_reset_pointnav = True
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)

    def _vlm_select_with_prompt(self, candidates, prompt: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        result = super()._vlm_select_with_prompt(candidates, prompt)
        self._parse_latest_vlm_eqa_answer("frontier_selection")
        return result

    def _stair_skip_vlm(self) -> bool:
        result = super()._stair_skip_vlm()
        self._parse_latest_vlm_eqa_answer("stair_skip")
        return result

    def _stagnant_ban_vlm(self, use_color_bev: Optional[bool] = None):
        result = super()._stagnant_ban_vlm(use_color_bev=use_color_bev)
        self._parse_latest_vlm_eqa_answer("stagnant_ban")
        return result

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self._eqa_answer:
            return self._eqa_stop_result()
        result = super().act(obs)
        self._parse_latest_vlm_eqa_answer("act")
        if self._eqa_answer:
            return self._eqa_stop_result(result)
        result = dict(result)
        result["eqa_answer"] = ""
        result["question"] = self._episode_question
        result["question_id"] = self._episode_question_id
        return result

    def get_status_text(self) -> str:
        status = super().get_status_text()
        target = re.sub(r"\s+", " ", str(self._core_target_object or "")).replace("|", " or ").strip()
        if target:
            status = f"{status} | proposed_target: {target}"
        return status

    def get_eqa_result(self) -> Dict[str, str]:
        return {
            "question_id": "" if self._episode_question_id is None else str(self._episode_question_id),
            "question": "" if self._episode_question is None else str(self._episode_question),
            "answer": self._fallback_eqa_answer() if self._eqa_answer is None else str(self._eqa_answer),
            "proposed_target": "" if self._core_target_object is None else str(self._core_target_object),
        }
