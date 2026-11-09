"""
Count episodes that likely require a floor transition.

Examples:
  HM3D v1
    eval "$(conda shell.bash hook)"
    conda activate pigeon
    python count_required_stairs.py \
      --habitat-cfg pigeon/configs/habitat.objectnav.hm3d.v1.yaml \
      --output-csv outputs/pigeon/required_stairs_hm3dv1.csv

  MP3D
    eval "$(conda shell.bash hook)"
    conda activate pigeon
    python count_required_stairs.py \
      --habitat-cfg pigeon/configs/habitat.objectnav.mp3d.yaml \
      --output-csv outputs/pigeon/required_stairs_mp3d.csv
"""

import argparse
import csv
from tqdm import tqdm
from collections import Counter
from copy import copy
from pathlib import Path
from typing import Iterable, List, Optional

import frontier_exploration  # noqa: F401
from habitat.config import read_write
from habitat.config.default import get_config
from habitat.core.env import Env
from habitat.datasets import make_dataset

from pigeon.measurements import ensure_registered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count episodes that likely require a floor transition."
    )
    parser.add_argument(
        "--habitat-cfg",
        default="pigeon/configs/habitat.objectnav.hm3d.v2.yaml",
        help="Habitat config used to build the dataset and TopDownMap measurement.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Only process the first N episodes after filtering. -1 means all.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Restrict to scenes whose basename matches this value. Can be passed multiple times.",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=None,
        help="Override habitat.simulator.habitat_sim_v0.gpu_device_id.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/pigeon/required_stairs_episodes.csv",
        help="Per-episode output CSV path.",
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Only write episodes whose current measurement marks them as requiring stairs.",
    )
    return parser.parse_args()


def iter_filtered_episodes(episodes: Iterable, scene_filters: List[str]):
    scene_filters = [s.strip() for s in scene_filters if s.strip()]
    for idx, episode in enumerate(episodes):
        scene_name = Path(episode.scene_id).stem
        if scene_filters and scene_name not in scene_filters:
            continue
        yield idx, episode, scene_name


def build_env(habitat_cfg: str, gpu_device_id: Optional[int]):
    ensure_registered()
    hcfg = get_config(config_path=habitat_cfg)
    if gpu_device_id is not None:
        with read_write(hcfg):
            hcfg.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_device_id
    dataset = make_dataset(id_dataset=hcfg.habitat.dataset.type, config=hcfg.habitat.dataset)
    example_dataset = copy(dataset)
    example_dataset.episodes = dataset.episodes[:1]
    env = Env(config=hcfg, dataset=example_dataset)
    env._episode_from_iter_on_reset = False
    env._episode_force_changed = True
    return dataset, env


def main() -> int:
    args = parse_args()
    dataset, env = build_env(args.habitat_cfg, args.gpu_device_id)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    required = 0
    by_scene = Counter()

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_index",
                "episode_id",
                "scene",
                "object_category",
                "start_height",
                "is_feasible_without_stairs",
                "requires_stairs",
            ],
        )
        writer.writeheader()

        for dataset_index, episode, scene_name in tqdm(iter_filtered_episodes(
            dataset.episodes, args.scene
        )):
            if args.limit >= 0 and processed >= args.limit:
                break

            env.current_episode = episode
            env.reset()
            metrics = env.get_metrics()
            top_down = metrics.get("top_down_map", {})
            is_feasible = bool(top_down.get("is_feasible", True))
            requires_stairs = not is_feasible

            processed += 1
            if requires_stairs:
                required += 1
                by_scene[scene_name] += 1

            if args.required_only and not requires_stairs:
                continue

            writer.writerow(
                {
                    "dataset_index": dataset_index,
                    "episode_id": episode.episode_id,
                    "scene": scene_name,
                    "object_category": getattr(episode, "object_category", ""),
                    "start_height": float(episode.start_position[1]),
                    "is_feasible_without_stairs": int(is_feasible),
                    "requires_stairs": int(requires_stairs),
                }
            )

    ratio = (required / processed) if processed else 0.0
    print(f"Processed episodes: {processed}")
    print(f"Require stairs: {required} ({ratio:.2%})")
    if by_scene:
        print("Per-scene required-stairs counts:")
        for scene_name, count in by_scene.most_common():
            print(f"  {scene_name}: {count}")
    print(f"CSV written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
