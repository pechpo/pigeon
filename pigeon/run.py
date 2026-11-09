import argparse
import os
import signal
import sys
import warnings
from dataclasses import dataclass
from pathlib import PurePath
from typing import Optional

from .eval.manager import EvalManager, DataGenManager, RunConfig
import frontier_exploration  # for config registration side-effects

# Ensure child processes (including the dedicated resource_tracker subprocess)
# inherit the warning suppression via PYTHONWARNINGS env var.
_warn_rules = [
    "ignore::UserWarning:multiprocessing.resource_tracker",
    "ignore::UserWarning:multiprocessing.semaphore_tracker",
]
_existing = os.environ.get("PYTHONWARNINGS", "")
for _rule in _warn_rules:
    if _rule not in _existing:
        _existing = f"{_rule},{_existing}" if _existing else _rule
os.environ["PYTHONWARNINGS"] = _existing


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pigeon: parallel evaluation using Habitat Env (no habitat_baselines)",
    )
    parser.add_argument(
        "--habitat_cfg",
        type=str,
        default="pigeon/configs/habitat.objectnav.hm3d.v2.yaml",
        help="Path to Habitat YAML config (defaults to HM3D v2 ObjectNav).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pigeon",
        help="Subdirectory under outputs/ for default outputs.",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="Results CSV path (resumable). Defaults to outputs/<output_dir>/eval_results.csv.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker threads.",
    )
    parser.add_argument(
        "--video_debug",
        action="store_true",
        help="Save an MP4 visualization for each episode.",
    )
    parser.add_argument(
        "--realtime_viz",
        action="store_true",
        help="Show real-time visualization (RGB + topdown map).",
    )
    parser.add_argument(
        "--color_bev",
        action="store_true",
        default=True,
        help="Build and show an additional colored BEV panel (default: enabled).",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory to store episode videos. Defaults to outputs/<output_dir>/videos.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evaluation from existing CSV results (skip finished episodes).",
    )
    parser.add_argument(
        "--only_floor_transition",
        action="store_true",
        help="Only evaluate episodes whose target requires a floor transition.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help=(
            "Only evaluate episodes from one scene. Accepts the scene folder "
            "(e.g. 00839-zt1RVoi7PcG), numeric prefix (00839), or scene id "
            "(zt1RVoi7PcG)."
        ),
    )
    parser.add_argument(
        "--episode_ids",
        type=str,
        default=None,
        help=(
            "Comma- or whitespace-separated episode indices/ids to evaluate. "
            "Applied in addition to --scene when provided."
        ),
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="vlm_fusion_bev",
        help="Planner name (default: vlm_fusion_bev).",
    )
    parser.add_argument(
        "--pointnav_ckpt", type=str, default="data/pointnav_weights.pth", help="Path to PointNav checkpoint"
    )
    parser.add_argument(
        "--gen_rl_data", action="store_true", default=False, help="Whether to generate RL training data"
    )
    parser.add_argument("--rl_data_path", type=str, default="data/rl_data", help="Path to store RL training data")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes to evaluate within one epoch.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs to evaluate.",
    )
    parser.add_argument("--visible_gpus", type=str, default="0", help="Path to store RL training data")
    parser.add_argument(
        "--dense_view_sampling",
        action="store_true",
        default=True,
        help="Enable 8-to-32 ray fallback for exhausted active-object viewpoints (default).",
    )
    parser.add_argument(
        "--final_check",
        action="store_true",
        default=True,
        help="Enable the point-cloud/depth multi-view VLM veto before stopping (default).",
    )
    parser.add_argument(
        "--goal_proximity_thresh",
        type=float,
        default=None,
        help=(
            "Override the planner's stop distance to a detected object goal, in "
            "meters (planner default: 0.90). Lower values make the agent approach "
            "closer before calling STOP."
        ),
    )
    parser.add_argument(
        "--episode_trace",
        dest="episode_trace",
        action="store_true",
        default=False,
        help="Write compact per-episode JSONL traces for failure analysis (default: disabled).",
    )
    parser.add_argument(
        "--no_episode_trace",
        dest="episode_trace",
        action="store_false",
        help="Disable per-episode JSONL traces.",
    )
    args = parser.parse_args(argv)

    if args.goal_proximity_thresh is not None and args.goal_proximity_thresh <= 0:
        parser.error("--goal_proximity_thresh must be positive")

    output_dir = PurePath(args.output_dir)
    if output_dir.is_absolute() or ".." in output_dir.parts:
        parser.error("--output_dir must be a relative subdirectory under outputs/")
    output_root = PurePath("outputs") / output_dir
    if args.results_csv is None:
        args.results_csv = str(output_root / "eval_results.csv")
    if args.video_dir is None:
        args.video_dir = str(output_root / "videos")
    return args


def _parse_episode_ids(value: Optional[str]) -> Optional[set[str]]:
    if value is None or not value.strip():
        return None
    parts = value.replace(",", " ").split()
    return {part.strip() for part in parts if part.strip()}


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ["PIGEON_COLOR_BEV"] = "1" if args.color_bev else "0"

    # Ensure custom Hydra nodes (measurements) are registered before manager parses YAML
    try:
        from .measurements import ensure_registered  # type: ignore

        ensure_registered()
    except Exception:
        pass
    try:
        from .datasets import ensure_registered as ensure_datasets_registered  # type: ignore

        ensure_datasets_registered()
    except Exception:
        pass

    cfg = RunConfig(
        habitat_cfg=args.habitat_cfg,
        num_workers=args.num_workers,
        results_csv=args.results_csv,
        video_dir=args.video_dir,
        video_debug=args.video_debug,
        realtime_viz=args.realtime_viz,
        resume=args.resume,
        scene=args.scene,
        episode_ids=_parse_episode_ids(args.episode_ids),
        only_floor_transition=args.only_floor_transition,
        planner=args.planner,
        pointnav_ckpt=args.pointnav_ckpt,
        rl_data_path=args.rl_data_path,
        num_episodes=args.num_episodes,
        num_epochs=args.num_epochs,
        visible_gpus=[int(g) for g in args.visible_gpus.split(",")],
        dense_view_sampling=args.dense_view_sampling,
        final_check=args.final_check,
        goal_proximity_thresh=args.goal_proximity_thresh,
        episode_trace=args.episode_trace,
    )

    if not args.gen_rl_data:
        mgr = EvalManager(cfg)
    else:
        cfg.planner = "dummy"
        mgr = DataGenManager(cfg)

    # Graceful shutdown on Ctrl-C
    def _signal_handler(sig, frame):
        # Request graceful stop for worker processes, then raise to break out
        mgr.request_stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    mgr.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
