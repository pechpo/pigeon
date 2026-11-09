import json
import os
from typing import Any, Dict, List, Optional, Sequence

from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)


@registry.register_dataset(name="OVONObjectNav-v1")
class OVONObjectNavDatasetV1(ObjectNavDatasetV1):
    """ObjectNav loader variant for HM3D OVON episode files.

    The HM3D OVON release stores the split JSON as an empty index and keeps
    actual episodes in per-scene content files. Those files omit Habitat's
    category id maps and contain a few extra episode keys. This loader keeps
    the ObjectNav episode/goal objects used by Habitat metrics while accepting
    that OVON layout.
    """

    @staticmethod
    def _get_scenes_from_folder(content_scenes_path: str, dataset_dir: str) -> List[str]:
        scenes = ObjectNavDatasetV1._get_scenes_from_folder(content_scenes_path, dataset_dir)
        return [scene for scene in scenes if not os.path.basename(scene).startswith("._")]

    @staticmethod
    def _deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)
        for vidx, view in enumerate(g.view_points or []):
            view_location = ObjectViewLocation(**view)  # type: ignore[arg-type]
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore[arg-type]
            g.view_points[vidx] = view_location
        return g

    @staticmethod
    def _allowed_episode_keys() -> set:
        return {
            "episode_id",
            "scene_id",
            "scene_dataset_config",
            "additional_obj_config_paths",
            "start_position",
            "start_rotation",
            "info",
            "_shortest_path_cache",
            "goals",
            "start_room",
            "shortest_paths",
            "object_category",
        }

    def _ensure_category_maps(
        self,
        episodes: Sequence[Dict[str, Any]],
        goals_by_category: Dict[str, Sequence[Dict[str, Any]]],
    ) -> None:
        if not hasattr(self, "category_to_task_category_id"):
            self.category_to_task_category_id = {}
        if not hasattr(self, "category_to_scene_annotation_category_id"):
            self.category_to_scene_annotation_category_id = {}

        categories = set()
        for ep in episodes:
            category = ep.get("object_category")
            if category:
                categories.add(str(category))
        for goals in goals_by_category.values():
            for goal in goals:
                category = goal.get("object_category")
                if category:
                    categories.add(str(category))

        for category in sorted(categories):
            if category not in self.category_to_task_category_id:
                idx = len(self.category_to_task_category_id)
                self.category_to_task_category_id[category] = idx
            if category not in self.category_to_scene_annotation_category_id:
                self.category_to_scene_annotation_category_id[category] = self.category_to_task_category_id[category]

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if not hasattr(self, "category_to_task_category_id"):
            self.category_to_task_category_id = {}
        if not hasattr(self, "category_to_scene_annotation_category_id"):
            self.category_to_scene_annotation_category_id = {}

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id.update(
                deserialized["category_to_task_category_id"]
            )
        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id.update(
                deserialized["category_to_scene_annotation_category_id"]
            )
        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id.update(
                deserialized["category_to_mp3d_category_id"]
            )

        episodes = deserialized.get("episodes", [])
        if len(episodes) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        goals_by_category = deserialized.get("goals_by_category", {})
        self._ensure_category_maps(episodes, goals_by_category)

        for k, v in goals_by_category.items():
            self.goals_by_category[k] = [self._deserialize_goal(g) for g in v]

        allowed_keys = self._allowed_episode_keys()
        for i, episode_dict in enumerate(episodes):
            episode_data = {k: v for k, v in episode_dict.items() if k in allowed_keys}
            episode = ObjectGoalNavEpisode(**episode_data)
            episode.episode_id = str(i)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = self.goals_by_category[episode.goals_key]

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }
                        path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)  # type: ignore[attr-defined]
