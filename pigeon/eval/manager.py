import os
import time
import multiprocessing as mp
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict, Optional, Any
from copy import copy
from math import ceil

from ..utils.csv_logger import CSVLogger
from ..utils.video_writer import VideoWriterFactory
from .gpu_manager import GPUModelManager
from .worker import EvalWorker, DataGenWorker, DatasetWorker, WorkerContext
from .eqa_utils import episode_field, existing_eqa_answer_ids
from pigeon.datasets import ensure_registered as ensure_datasets_registered

from habitat.config.default import get_config, patch_config
from habitat.config import read_write
from habitat.datasets import make_dataset
from habitat.core.dataset import EpisodeIterator

ensure_datasets_registered()


@dataclass
class RunConfig:
    habitat_cfg: str
    num_workers: int
    results_csv: str
    video_dir: str
    video_debug: bool
    realtime_viz: bool
    resume: bool
    scene: Optional[str]
    episode_ids: Optional[set[str]]
    num_episodes: int
    num_epochs: int
    visible_gpus: List[int]
    only_floor_transition: bool = False
    planner: str = "vlm_fusion_bev"
    pointnav_ckpt: str = "data/pointnav_weights.pth"
    stairnav_ckpt: str = "data/stairnav_3.pth"
    rl_data_path: str = None
    dense_view_sampling: bool = True
    final_check: bool = True
    goal_proximity_thresh: Optional[float] = None
    episode_trace: bool = False


def _episode_scene_keys(episode: Any) -> set[str]:
    scene_id = str(getattr(episode, "scene_id", "") or "")
    scene_path = Path(scene_id)
    scene_folder = scene_path.parent.name
    asset_name = scene_path.name
    asset_stem = scene_path.stem
    if asset_stem.endswith(".basis"):
        asset_stem = asset_stem[: -len(".basis")]

    keys = {scene_id, scene_folder, asset_name, asset_stem}
    if "-" in scene_folder:
        prefix, suffix = scene_folder.split("-", 1)
        keys.update({prefix, suffix})
    return {k.strip().lower() for k in keys if k}


def _episode_matches_scene(episode: Any, scene_filter: Optional[str]) -> bool:
    if scene_filter is None or not scene_filter.strip():
        return True
    return scene_filter.strip().lower() in _episode_scene_keys(episode)


def _episode_matches_id(idx: int, episode: Any, episode_ids: Optional[set[str]]) -> bool:
    if not episode_ids:
        return True
    candidates = {str(idx)}
    for field in ("episode_id", "question_id"):
        value = getattr(episode, field, None)
        if value is not None:
            candidates.add(str(value))
    return bool(candidates & episode_ids)


class GPUWorkerManager:
    def __init__(
        self,
        cfg: RunConfig,
        ctx_kwargs: Dict,
        worker_class: Union[EvalWorker, DataGenWorker],
        worker_kwargs: Dict,
        worker_ids: List[int],
        gpu_id: int = 0,
    ) -> None:
        self.gpu_id = gpu_id
        self.workers: List[Union[EvalWorker, DataGenWorker]] = []

        # Shared model request/response queues
        self.model_request_q: mp.Queue = mp.Queue()
        self.model_reply_qs = {wid: mp.Queue() for wid in worker_ids}

        # Pre-allocate worker processes
        self.workers: List[Union[EvalWorker, DataGenWorker]] = []
        for worker_id in worker_ids:
            ctx = WorkerContext(
                worker_id=worker_id,
                video_writer_factory=VideoWriterFactory(cfg.video_dir, enable_debug=cfg.video_debug),
                planner_name=cfg.planner,
                habitat_cfg=cfg.habitat_cfg,
                realtime_viz=cfg.realtime_viz,
                only_floor_transition=cfg.only_floor_transition,
                dense_view_sampling=cfg.dense_view_sampling,
                final_check=cfg.final_check,
                goal_proximity_thresh=cfg.goal_proximity_thresh,
                model_request_q=self.model_request_q,
                model_reply_q=self.model_reply_qs[worker_id],
                **ctx_kwargs,
            )
            worker_kwargs["ctx"] = ctx
            self.workers.append(worker_class(**worker_kwargs))

        # GPU model manager process (lazy-loads pointnav_rnn)
        self.model_manager = GPUModelManager(
            request_q=self.model_request_q,
            reply_queues=self.model_reply_qs,
            pointnav_ckpt=cfg.pointnav_ckpt,
            stairnav_ckpt=cfg.stairnav_ckpt,
        )

    def start(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        # Start GPU manager
        self.model_manager.start()

        # Start workers
        for w in self.workers:
            w.start()

    def join(self) -> None:
        for w in self.workers:
            w.join(timeout=30)
            if w.is_alive():
                print(f"[EvalManager] Worker {w.pid} did not stop within 30s; terminating it.")
                w.terminate()
                w.join(timeout=5)
                if w.is_alive():
                    w.kill()
                    w.join(timeout=5)
        # Shutdown GPU manager
        self.model_request_q.put({"type": "shutdown"})
        self.model_manager.join(timeout=30)
        if self.model_manager.is_alive():
            print("[EvalManager] GPU model manager did not stop within 30s; terminating it.")
            self.model_manager.terminate()
            self.model_manager.join(timeout=5)
            if self.model_manager.is_alive():
                self.model_manager.kill()
                self.model_manager.join(timeout=5)


class EvalManager:
    """Coordinates multi-process evaluation with resumable CSV logging.

    Responsibilities:
    - Prepare dataset and a queue of episodes.
    - Spawn worker processes, each stepping an episode independently.
    - Ensure results are persisted to CSV and video output is released promptly.
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        Path(cfg.video_dir).mkdir(parents=True, exist_ok=True)
        Path(Path(cfg.results_csv).parent).mkdir(parents=True, exist_ok=True)

        # Planners
        # No shared executor across threads; each worker gets its own.
        # Keep a factory for creating per-worker planners.

        # Dataset (Habitat) and CSV logger
        self.logger = CSVLogger(cfg.results_csv)
        self.stop_event = mp.Event()

        hcfg = get_config(config_path=cfg.habitat_cfg)
        print(hcfg)
        dataset = make_dataset(id_dataset=hcfg.habitat.dataset.type, config=hcfg.habitat.dataset)
        example_dataset = copy(dataset)
        example_dataset.episodes = dataset.episodes[:1]
        filtered_episode_count = (
            sum(
                1
                for idx, episode in enumerate(dataset.episodes)
                if _episode_matches_scene(episode, cfg.scene)
                and _episode_matches_id(idx, episode, cfg.episode_ids)
            )
            if cfg.scene or cfg.episode_ids
            else len(dataset.episodes)
        )
        total_eps = min(filtered_episode_count, cfg.num_episodes)
        is_eqa_dataset = str(hcfg.habitat.dataset.type).lower() == "openeqa-v1"
        answered_question_ids = (
            existing_eqa_answer_ids(Path(cfg.video_dir).parent) if cfg.resume and is_eqa_dataset else set()
        )
        finished_counts = {} if is_eqa_dataset else (self.logger.existing_episode_counts() if cfg.resume else {})
        self.episode_queue = mp.JoinableQueue()
        # default behavior
        for epoch in range(cfg.num_epochs):
            queued_for_epoch = 0
            # default behavior
            # Resume may enqueue fewer episodes than total_eps. Stop at the
            # dataset boundary instead of cycling forever waiting to fill it.
            for idx, episode in enumerate(EpisodeIterator(dataset.episodes, shuffle=False, cycle=False)):
                if (
                    (cfg.scene or cfg.episode_ids) and queued_for_epoch >= total_eps
                ) or (not cfg.scene and not cfg.episode_ids and idx >= total_eps):
                    break
                if not _episode_matches_scene(episode, cfg.scene):
                    continue
                if not _episode_matches_id(idx, episode, cfg.episode_ids):
                    continue
                if is_eqa_dataset:
                    question_id = episode_field(episode, ("question_id", "episode_id"), fallback=str(idx))
                    if str(question_id) in answered_question_ids:
                        continue
                    self.episode_queue.put((idx, episode))
                    queued_for_epoch += 1
                    continue
                # Multi-epoch resume: allow each episode_id to be evaluated once per epoch.
                # If an episode_id appears k times in the CSV, we consider its first k epochs done.
                if finished_counts.get(str(idx), 0) > epoch:
                    continue
                self.episode_queue.put((idx, episode))
                queued_for_epoch += 1
            if (cfg.scene or cfg.episode_ids) and queued_for_epoch == 0:
                print(f"No episodes queued for requested filters in epoch {epoch}.")

        self.gpu_managers: List[GPUWorkerManager] = []
        workers_per_gpu = ceil(cfg.num_workers / len(cfg.visible_gpus))
        for idx, gpu_id in enumerate(cfg.visible_gpus):
            start = idx * workers_per_gpu
            n_workers = min(cfg.num_workers - start, workers_per_gpu)
            self.gpu_managers.append(
                GPUWorkerManager(
                    cfg=cfg,
                    ctx_kwargs={
                        "logger": self.logger,
                        "stop_event": self.stop_event,
                        "total_episodes": total_eps,
                        "example_dataset": example_dataset,
                        "trace_dir": (
                            str(Path(cfg.results_csv).parent / "episode_traces")
                            if cfg.episode_trace
                            else None
                        ),
                    },
                    worker_class=EvalWorker,
                    worker_kwargs={"episode_queue": self.episode_queue},
                    worker_ids=list(range(start, start + n_workers)),
                    gpu_id=gpu_id,
                )
            )

    def request_stop(self) -> None:
        # Do not drain the entire episode queue here: this method is also
        # called from the SIGTERM handler, where blocking on queue IPC can
        # deadlock shutdown after a worker has failed.
        self.stop_event.set()

    def run(self) -> None:
        # Start workers
        for m in self.gpu_managers:
            m.start()

        # Wait for all episodes to be processed or stop requested
        try:
            self.episode_queue.join()
        except KeyboardInterrupt:
            pass
        finally:
            print("Shutting down workers...")
            self.request_stop()
            for m in self.gpu_managers:
                m.join()
            print("All workers shut down.")


class DataGenManager(EvalManager):
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        Path(cfg.video_dir).mkdir(parents=True, exist_ok=True)
        Path(Path(cfg.results_csv).parent).mkdir(parents=True, exist_ok=True)

        # Planners
        # No shared executor across threads; each worker gets its own.
        # Keep a factory for creating per-worker planners.

        # Dataset (Habitat) and CSV logger
        self.logger = CSVLogger(cfg.results_csv)
        self.stop_event = mp.Event()

        hcfg = get_config(config_path=cfg.habitat_cfg)
        print(hcfg)
        dataset = make_dataset(id_dataset=hcfg.habitat.dataset.type, config=hcfg.habitat.dataset)
        example_dataset = copy(dataset)
        example_dataset.episodes = dataset.episodes[:1]
        scene_episode_count = (
            sum(1 for episode in dataset.episodes if _episode_matches_scene(episode, cfg.scene))
            if cfg.scene
            else len(dataset.episodes)
        )
        total_eps = min(scene_episode_count, cfg.num_episodes)
        finished_counts = self.logger.existing_episode_counts() if cfg.resume else {}
        self.episode_queue = mp.JoinableQueue()
        # default behavior
        queued = 0
        for idx, episode in enumerate(EpisodeIterator(dataset.episodes, shuffle=True, group_by_scene=False, seed=100)):
            if not _episode_matches_scene(episode, cfg.scene):
                continue
            if finished_counts.get(str(idx), 0) > 0:
                continue
            if (cfg.scene and queued >= total_eps) or (not cfg.scene and idx >= total_eps):
                break
            self.episode_queue.put((idx, episode))
            queued += 1
        if cfg.scene and queued == 0:
            print(f"No episodes queued for scene filter '{cfg.scene}'.")

        # Shared model request/response queues
        self.data_queue: mp.Queue = mp.Queue()

        self.gpu_managers: List[GPUWorkerManager] = []
        workers_per_gpu = cfg.num_workers // len(cfg.visible_gpus)
        for idx, gpu_id in enumerate(cfg.visible_gpus):
            start = idx * workers_per_gpu
            n_workers = min(cfg.num_workers - start, workers_per_gpu)
            self.gpu_managers.append(
                GPUWorkerManager(
                    cfg=cfg,
                    ctx_kwargs={
                        "logger": self.logger,
                        "stop_event": self.stop_event,
                        "total_episodes": total_eps,
                        "example_dataset": example_dataset,
                    },
                    worker_class=DataGenWorker,
                    worker_kwargs={"episode_queue": self.episode_queue, "data_queue": self.data_queue},
                    worker_ids=list(range(start, start + n_workers)),
                    gpu_id=gpu_id,
                )
            )

        self.dataset_worker = DatasetWorker(self.data_queue, cfg.rl_data_path)

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        # Start GPU manager
        self.dataset_worker.start()

        # Start workers
        for m in self.gpu_managers:
            m.start()

        # Wait for all episodes to be processed or stop requested
        try:
            self.episode_queue.join()
        except KeyboardInterrupt:
            pass
        finally:
            print("Shutting down workers...")
            self.request_stop()
            self.data_queue.put(None)
            self.dataset_worker.join()
            for m in self.gpu_managers:
                m.join()
            print("All workers shut down.")
