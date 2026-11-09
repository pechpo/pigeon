import fcntl
import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

from habitat.config import read_write
from habitat.tasks.nav.nav import NavigationEpisode


_EQA_OBJECTNAV_MEASURES_TO_DROP = ("spl", "soft_spl", "distance_to_goal_reward")


def _load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Refusing to overwrite unreadable EQA answer file: {path}") from exc
    if not isinstance(data, list):
        raise RuntimeError(f"Refusing to overwrite non-list EQA answer file: {path}")
    return data


def existing_eqa_answer_ids(out_dir: Path) -> set:
    ids = set()
    for item in _load_json_list(out_dir / "gpt_answer.json"):
        if isinstance(item, dict):
            question_id = item.get("question_id")
            if question_id is not None and str(question_id).strip():
                ids.add(str(question_id))
    return ids


def patch_eqa_habitat_config(hcfg: Any) -> None:
    """Disable ObjectNav path-efficiency metrics for OpenEQA episodes.

    AEQA records contain a question pose, not a real navigation goal. The
    OpenEQA dataset wrapper supplies synthetic goals only so Habitat's ObjectNav
    task can reset. SPL/SoftSPL divide by the start-goal geodesic distance and
    can hit 0/0 when that synthetic goal projects onto the start point.
    """
    dataset_type = str(getattr(hcfg.habitat.dataset, "type", ""))
    if dataset_type.lower() != "openeqa-v1":
        return

    with read_write(hcfg):
        measurements = hcfg.habitat.task.measurements
        for measure_name in _EQA_OBJECTNAV_MEASURES_TO_DROP:
            if measure_name in measurements:
                measurements.pop(measure_name)


def episode_field(ep: NavigationEpisode, names: Tuple[str, ...], fallback: str = "") -> str:
    for name in names:
        value = getattr(ep, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    info = getattr(ep, "info", None)
    if isinstance(info, dict):
        for name in names:
            value = info.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return fallback


def append_eqa_answer_json(
    out_dir: Path,
    question_id: str,
    question: str,
    answer: str,
    proposed_target: str = "",
    ground_truth_answer: str = "",
    path_length: Optional[float] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "gpt_answer.json"
    path_length_path = out_dir / "path_length_list.pkl"
    lock_path = out_dir / "gpt_answer.json.lock"
    tmp_path = out_dir / f"gpt_answer.{os.getpid()}.tmp"
    tmp_path_length_path = out_dir / f"path_length_list.{os.getpid()}.tmp"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            data = _load_json_list(result_path)

            item = {
                "question_id": str(question_id),
                "question": str(question or ""),
                "answer": str(answer or ""),
                "proposed_target": str(proposed_target or ""),
                "ground_truth_answer": str(ground_truth_answer or ""),
            }
            replaced = False
            for idx, existing in enumerate(data):
                if isinstance(existing, dict) and str(existing.get("question_id", "")) == item["question_id"]:
                    data[idx] = item
                    replaced = True
                    break
            if not replaced:
                data.append(item)
            with tmp_path.open("w") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(result_path)

            if path_length is not None:
                try:
                    if path_length_path.exists():
                        with path_length_path.open("rb") as f:
                            path_lengths = pickle.load(f)
                        if not isinstance(path_lengths, dict):
                            path_lengths = {}
                    else:
                        path_lengths = {}
                except Exception:
                    path_lengths = {}

                path_lengths[item["question_id"]] = float(path_length)
                with tmp_path_length_path.open("wb") as f:
                    pickle.dump(path_lengths, f)
                tmp_path_length_path.replace(path_length_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
