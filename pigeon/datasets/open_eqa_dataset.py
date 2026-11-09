import json
import os
import gzip
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import quaternion
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectGoalNavEpisode, ObjectViewLocation
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis, quat_to_coeffs
from habitat.utils.geometry_utils import quaternion_from_coeff


@registry.register_dataset(name="OpenEQA-v1")
class OpenEQADatasetV1(ObjectNavDatasetV1):
    """Loader for OpenEQA/AEQA question files.

    The source file is a list of question records, not a standard Habitat
    ObjectNav episode JSON. Each record is converted into an ObjectGoalNavEpisode
    so the existing ObjectNav simulator/task stack can reset from the recorded
    pose while pigeon's EQA planner reads the question metadata.
    """

    category_to_task_category_id = {"eqa": 0}
    category_to_scene_annotation_category_id = {"eqa": 0}

    def __init__(self, config=None) -> None:
        self.episodes = []
        self.goals_by_category = {}
        self.category_to_task_category_id = {"eqa": 0}
        self.category_to_scene_annotation_category_id = {"eqa": 0}
        if config is None:
            return
        datasetfile_path = config.data_path.format(split=config.split)
        scenes_dir = getattr(config, "scenes_dir", None)
        opener = gzip.open if str(datasetfile_path).endswith(".gz") else open
        with opener(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=scenes_dir)

    @staticmethod
    def _scene_id_from_episode_history(episode_history: str, scenes_dir: Optional[str]) -> str:
        history = str(episode_history).strip()
        scene_key = history.split("-", 1)[1] if "-" in history else history
        scene_rel = os.path.join("hm3d", "val", history, f"{scene_key}.basis.glb")
        if scenes_dir:
            return os.path.join(scenes_dir, scene_rel)
        return os.path.join("data", "scene_datasets", scene_rel)

    @staticmethod
    def _synthetic_view_points(
        position: Sequence[float], rotation: Sequence[float]
    ) -> List[ObjectViewLocation]:
        """Build synthetic ObjectNav viewpoints for an AEQA question pose.

        AEQA does not provide a target object pose. These points are only a
        compatibility shim for Habitat's ObjectNav task/measurements, so keep
        them away from the start pose to avoid zero-length start-goal distances.
        """
        offsets = (
            (1.5, 0.0, 0.0),
            (-1.5, 0.0, 0.0),
            (0.0, 0.0, 1.5),
            (0.0, 0.0, -1.5),
            (2.5, 0.0, 0.0),
            (-2.5, 0.0, 0.0),
            (0.0, 0.0, 2.5),
            (0.0, 0.0, -2.5),
        )
        return [
            OpenEQADatasetV1._synthetic_view_point(position, rotation, offset)
            for offset in offsets
        ]

    @staticmethod
    def _synthetic_view_point(
        position: Sequence[float],
        rotation: Sequence[float],
        offset: Sequence[float],
    ) -> ObjectViewLocation:
        view_position = list(position)
        if len(view_position) >= 3:
            view_position = [
                float(view_position[0]) + float(offset[0]),
                float(view_position[1]) + float(offset[1]),
                float(view_position[2]) + float(offset[2]),
            ]
        try:
            view_rotation = quaternion_from_coeff(
                OpenEQADatasetV1._aeqa_rotation_to_habitat_coeff(list(rotation))
            )
        except Exception:
            view_rotation = None
        return ObjectViewLocation(
            agent_state=AgentState(position=view_position, rotation=view_rotation),
            iou=1.0,
        )

    @staticmethod
    def _aeqa_rotation_to_habitat_coeff(rotation: List[float]) -> List[float]:
        """Convert AEQA [w, x, y, z] quaternion to Habitat coeffs [x, y, z, w].

        3D-Mem loads AEQA poses with ``quaternion.quaternion(*rotation)`` and
        keeps only yaw before setting the Habitat-Sim agent rotation. Mirror
        that here because Habitat-Lab start_rotation expects backend coeffs.
        """
        q = quaternion.quaternion(*rotation)
        angle, axis = quat_to_angle_axis(q)
        if abs(float(axis[1])) < 1e-6:
            yaw = 0.0
        else:
            yaw = float(angle) * float(axis[1]) / abs(float(axis[1]))
        return quat_to_coeffs(quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))).tolist()

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        records = json.loads(json_str)
        if isinstance(records, dict):
            records = records.get("episodes", records.get("questions", []))
        if not isinstance(records, list):
            raise ValueError("OpenEQA-v1 expects a list of question records")

        self.episodes = []
        self.goals_by_category = {}

        for idx, item in enumerate(records):
            if not isinstance(item, dict):
                continue
            episode_history = item.get("episode_history")
            position = item.get("position")
            rotation = item.get("rotation")
            question = item.get("question")
            question_id = item.get("question_id", idx)
            if not episode_history or position is None or rotation is None or not question:
                continue

            question_id_str = str(question_id)
            info: Dict[str, Any] = {
                "question": str(question),
                "question_id": question_id_str,
                "answer": item.get("answer"),
                "category": item.get("category"),
                "episode_history": str(episode_history),
                "object_id": item.get("object_id"),
                "class": item.get("class"),
            }
            view_points = self._synthetic_view_points(list(position), list(rotation))
            goal = ObjectGoal(
                position=list(view_points[0].agent_state.position),
                radius=1.0,
                object_id="eqa",
                object_category="eqa",
                object_name="eqa",
                object_name_id=0,
                view_points=view_points,
            )
            episode = ObjectGoalNavEpisode(
                episode_id=question_id_str,
                scene_id=self._scene_id_from_episode_history(str(episode_history), scenes_dir),
                start_position=list(position),
                start_rotation=self._aeqa_rotation_to_habitat_coeff(list(rotation)),
                goals=[goal],
                object_category="eqa",
                info=info,
                shortest_paths=None,
            )
            episode.question = str(question)
            episode.question_id = question_id_str
            episode.answer = item.get("answer")
            episode.category = item.get("category")
            self.episodes.append(episode)
