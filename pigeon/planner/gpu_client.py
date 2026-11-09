from dataclasses import dataclass
from queue import Empty
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.shared_memory import ShmArray, attach_ndarray


@dataclass
class GPUWorkerClient:
    worker_id: int
    request_q: Any
    reply_q: Any

    def __post_init__(self) -> None:
        self._req_id = 100000
        self._episode_sampling_seed = None
        self._navigation_sample_index = 0
        # How long to wait for GPU replies per poll
        # Some models (e.g., first-time YOLO load) can take several seconds
        self._reply_timeout_sec = 30.0
        # Shared-memory slots reused across calls
        self._depth_slot = ShmArray()
        self._image_slot = ShmArray()
        self._goal_slot = ShmArray()

    def set_episode_seed(self, seed: Optional[int]) -> None:
        self._episode_sampling_seed = seed
        self._navigation_sample_index = 0

    def _add_sampling_seed(self, msg: Dict[str, Any]) -> None:
        if self._episode_sampling_seed is not None:
            msg["sampling_seed"] = (
                self._episode_sampling_seed * 1_000_003 + self._navigation_sample_index
            ) % (2**63 - 1)
            self._navigation_sample_index += 1

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _await(self, req_id: int) -> Dict[str, Any]:
        while True:
            try:
                resp = self.reply_q.get(timeout=self._reply_timeout_sec)
            except Empty:
                # Keep waiting for our response; model may still be initializing
                continue
            if resp.get("req_id") == req_id:
                return resp

    def pointnav(self, depth: np.ndarray, goal_rho_theta: np.ndarray, reset: bool = False) -> int:
        d = depth
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32, copy=False)
        self._depth_slot.write(d)

        g = np.asarray(goal_rho_theta, dtype=np.float32)
        self._goal_slot.write(g)

        rid = self._next_id()
        ddesc = self._depth_slot.desc()
        gdesc = self._goal_slot.desc()
        msg = {
            "req_id": rid,
            "worker_id": int(self.worker_id),
            "task": "pointnav_rnn",
            "depth_name": ddesc["name"],
            "depth_shape": ddesc["shape"],
            "depth_dtype": ddesc["dtype"],
            "goal_name": gdesc["name"],
            "goal_shape": gdesc["shape"],
            "goal_dtype": gdesc["dtype"],
        }
        if reset:
            msg["reset"] = True
        self._add_sampling_seed(msg)
        self.request_q.put(msg)
        resp = self._await(rid)
        if "action" in resp:
            return int(resp["action"])  # type: ignore[arg-type]
        # If we didn't receive a valid action, default to stop
        return 0

    def stairnav(self, depth: np.ndarray, goal_rho_theta: np.ndarray, reset: bool = False) -> int:
        d = depth
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32, copy=False)
        self._depth_slot.write(d)

        g = np.asarray(goal_rho_theta, dtype=np.float32)
        self._goal_slot.write(g)

        rid = self._next_id()
        ddesc = self._depth_slot.desc()
        gdesc = self._goal_slot.desc()
        msg = {
            "req_id": rid,
            "worker_id": int(self.worker_id),
            "task": "stairnav_rnn",
            "depth_name": ddesc["name"],
            "depth_shape": ddesc["shape"],
            "depth_dtype": ddesc["dtype"],
            "goal_name": gdesc["name"],
            "goal_shape": gdesc["shape"],
            "goal_dtype": gdesc["dtype"],
        }
        if reset:
            msg["reset"] = True
        self._add_sampling_seed(msg)
        self.request_q.put(msg)
        resp = self._await(rid)
        if "action" in resp:
            return int(resp["action"])  # type: ignore[arg-type]
        # If we didn't receive a valid action, default to stop
        return 0
    
    def yolov7(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        filter_class: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        img = np.asarray(image)
        self._image_slot.write(img)
        rid = self._next_id()
        idesc = self._image_slot.desc()
        msg: Dict[str, Any] = {
            "req_id": rid,
            "worker_id": int(self.worker_id),
            "task": "yolov7",
            "image_name": idesc["name"],
            "image_shape": idesc["shape"],
            "image_dtype": idesc["dtype"],
            "conf_thresh": float(conf),
        }
        if filter_class:
            msg["filter_class"] = list(filter_class)
        self.request_q.put(msg)
        return self._await(rid)

    def owl_v2(
        self,
        image: np.ndarray,
        threshold: float = 0.2,
        filter_class: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Call OWLv2 via GPU manager."""
        img = np.asarray(image)
        self._image_slot.write(img)
        rid = self._next_id()
        idesc = self._image_slot.desc()
        msg: Dict[str, Any] = {
            "req_id": rid,
            "worker_id": int(self.worker_id),
            "task": "owl_v2",
            "image_name": idesc["name"],
            "image_shape": idesc["shape"],
            "image_dtype": idesc["dtype"],
            "threshold": float(threshold),
        }
        if filter_class:
            msg["filter_class"] = list(filter_class)
        self.request_q.put(msg)
        return self._await(rid)

    def mobile_sam(self, image: np.ndarray, bbox_px: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        img = np.asarray(image)
        self._image_slot.write(img)
        rid = self._next_id()
        idesc = self._image_slot.desc()
        msg = {
            "req_id": rid,
            "worker_id": int(self.worker_id),
            "task": "mobile_sam",
            "image_name": idesc["name"],
            "image_shape": idesc["shape"],
            "image_dtype": idesc["dtype"],
            "bbox": [list(map(int, bbox_px))],
        }
        self.request_q.put(msg)
        resp = self._await(rid)
        if not bool(resp.get("found", False)):
            return None
        mname = resp.get("mask_name")
        mshape = tuple(resp.get("mask_shape", []))  # type: ignore
        mdtype = str(resp.get("mask_dtype"))
        if not mname or not mshape:
            return None
        shm, marr = attach_ndarray(mname, mshape, mdtype)
        try:
            return np.array(marr, copy=True)
        finally:
            shm.close()
            try:
                shm.unlink()
            except Exception:
                pass

    def itm_cosine(self, image: np.ndarray, text: str) -> float:
        """Compute BLIP2-ITM cosine similarity via GPU manager.

        Args:
            image: RGB image as numpy array (H,W,3) uint8.
            text: Prompt string.
        Returns:
            Cosine similarity as float.
        """
        img = np.asarray(image)
        self._image_slot.write(img)
        rid = self._next_id()
        idesc = self._image_slot.desc()
        msg = {
            "req_id": rid,
            "worker_id": int(self.worker_id),
            "task": "blip2itm",
            "image_name": idesc["name"],
            "image_shape": idesc["shape"],
            "image_dtype": idesc["dtype"],
            "text": str(text),
        }
        self.request_q.put(msg)
        resp = self._await(rid)
        try:
            return float(resp.get("cosine", 0.0))
        except Exception:
            return 0.0

    def close(self) -> None:
        """Release persistent shared-memory slots to minimize resource_tracker noise."""
        for slot in (
            getattr(self, "_depth_slot", None),
            getattr(self, "_image_slot", None),
            getattr(self, "_goal_slot", None),
        ):
            if slot is not None:
                slot.close()
