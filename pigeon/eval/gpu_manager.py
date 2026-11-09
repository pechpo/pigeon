import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict
import numpy as np

from ..utils.eval_randomness import policy_action


class GPUModelManager(mp.Process):
    """Simple single-process GPU model manager.

    - Loads required models once (here only pointnav_rnn).
    - Consumes requests from a single request queue.
    - Produces responses on per-worker reply queues.
    - All large tensors are passed via multiprocessing.shared_memory.
    """

    def __init__(
        self,
        request_q: "mp.Queue[Dict[str, Any]]",
        reply_queues: Dict[int, "mp.Queue[Dict[str, Any]]"],
        pointnav_ckpt: str,
        stairnav_ckpt: str,
    ) -> None:
        super().__init__(daemon=True)
        self._request_q = request_q
        self._reply_queues = reply_queues
        self._pointnav_ckpt = pointnav_ckpt
        self._stairnav_ckpt = stairnav_ckpt

        # Model placeholders
        self._pointnav = None
        self._stairnav = None
        self._yolov7 = None
        self._sam = None
        self._blip2itm = None
        self._owl_v2 = None
        # Torch/cuml lazy init flags
        self._cuml_ready = False
        # Per-worker RNN states for pointnav
        self._pn_states: Dict[int, Dict[str, Any]] = {}
        self._sn_states: Dict[int, Dict[str, Any]] = {}

    def _ensure_pointnav(self) -> None:
        if self._pointnav is None:
            from ..models.pointnav_rnn import WrappedPointNavHBPolicy

            self._pointnav = WrappedPointNavHBPolicy(self._pointnav_ckpt, device="cuda")

    def _ensure_stairnav(self) -> None:
        if self._stairnav is None:
            from ..models.stairnav_rnn import WrappedStairNavHBPolicy

            self._stairnav = WrappedStairNavHBPolicy(self._stairnav_ckpt, device="cuda")

    def _handle_pointnav(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        # Attach to shared memory
        d_shm = SharedMemory(name=msg["depth_name"])
        g_shm = SharedMemory(name=msg["goal_name"])
        try:
            depth = np.ndarray(
                shape=tuple(msg["depth_shape"]),
                dtype=np.dtype(msg["depth_dtype"]),
                buffer=d_shm.buf,
            )
            goal = np.ndarray(
                shape=tuple(msg["goal_shape"]),
                dtype=np.dtype(msg["goal_dtype"]),
                buffer=g_shm.buf,
            )
            # Ensure models are ready
            self._ensure_pointnav()

            # Load per-worker RNN state (or reset)
            wid = int(msg.get("worker_id", -1))
            reset = bool(msg.get("reset", False))
            state = None if reset else self._pn_states.get(wid)
            self._pointnav.set_state(state)

            act = int(policy_action(self._pointnav, depth, goal, msg.get("sampling_seed")))

            self._pn_states[wid] = self._pointnav.get_state()
        finally:
            d_shm.close()
            g_shm.close()

        return {"req_id": msg["req_id"], "action": act}

    def _handle_stairnav(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        # Attach to shared memory
        d_shm = SharedMemory(name=msg["depth_name"])
        g_shm = SharedMemory(name=msg["goal_name"])
        try:
            depth = np.ndarray(
                shape=tuple(msg["depth_shape"]),
                dtype=np.dtype(msg["depth_dtype"]),
                buffer=d_shm.buf,
            )
            goal = np.ndarray(
                shape=tuple(msg["goal_shape"]),
                dtype=np.dtype(msg["goal_dtype"]),
                buffer=g_shm.buf,
            )
            # Ensure models are ready
            self._ensure_stairnav()

            # Load per-worker RNN state (or reset)
            wid = int(msg.get("worker_id", -1))
            reset = bool(msg.get("reset", False))
            state = None if reset else self._sn_states.get(wid)
            self._stairnav.set_state(state)

            act = int(policy_action(self._stairnav, depth, goal, msg.get("sampling_seed")))

            self._sn_states[wid] = self._stairnav.get_state()
        finally:
            d_shm.close()
            g_shm.close()

        return {"req_id": msg["req_id"], "action": act}

    def _ensure_yolo(self) -> None:
        if self._yolov7 is None:
            from ..models.yolov7 import YOLOv7

            # Default weights path; adjust as needed
            self._yolov7 = YOLOv7(weights="data/yolov7-e6e.pt")

    def _ensure_owl_v2(self) -> None:
        if self._owl_v2 is None:
            from ..models.owlv2 import OWLv2

            self._owl_v2 = OWLv2(model_id="google/owlv2-base-patch16-ensemble", device="cuda")

    def _ensure_sam(self) -> None:
        if self._sam is None:
            from ..models.sam import MobileSAM
            import os

            ckpt = os.environ.get("MOBILE_SAM_CHECKPOINT", "data/mobile_sam.pt")
            self._sam = MobileSAM(sam_checkpoint=ckpt)

    def _ensure_blip2itm(self) -> None:
        if self._blip2itm is None:
            from ..models.blip2itm import BLIP2ITM
            # Loads on CUDA if available inside the GPU manager process
            self._blip2itm = BLIP2ITM()

    def _handle_blip2itm(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_blip2itm()
        img_shm = SharedMemory(name=msg["image_name"])
        try:
            image = np.ndarray(
                shape=tuple(msg["image_shape"]),
                dtype=np.dtype(msg["image_dtype"]),
                buffer=img_shm.buf,
            )
            text = str(msg.get("text", ""))
            try:
                cosine = float(self._blip2itm.cosine(image, text))
            except Exception:
                cosine = 0.0
            return {"req_id": msg["req_id"], "cosine": float(cosine)}
        finally:
            img_shm.close()


    def _handle_yolov7(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_yolo()
        img_shm = SharedMemory(name=msg["image_name"])
        try:
            image = np.ndarray(
                shape=tuple(msg["image_shape"]), dtype=np.dtype(msg["image_dtype"]), buffer=img_shm.buf
            )
            conf = float(msg.get("conf_thresh", 0.25))
            dets = self._yolov7.predict(image, conf_thres=conf)
            # Optional filtering by one or more class names (list of strings)
            fcls = msg.get("filter_class")
            if fcls:
                targets = [str(c).lower() for c in fcls]
                phrases = [str(p).lower() for p in dets.phrases]
                idxs = [i for i, ps in enumerate(phrases) if any(t in ps for t in targets)]
                if not idxs:
                    return {"req_id": msg["req_id"], "found": False}
                import torch

                keep = torch.zeros_like(dets.logits, dtype=torch.bool)
                for i in idxs:
                    if 0 <= i < len(keep):
                        keep[i] = True
                dets._filter(keep)
            if dets.num_detections == 0:
                return {"req_id": msg["req_id"], "found": False}

            import torch

            # Indices sorted by score (descending)
            scores_tensor = dets.logits
            boxes_tensor = dets.boxes
            phrases = list(dets.phrases)

            best_i = int(torch.argmax(scores_tensor).item())
            best_box = boxes_tensor[best_i].detach().cpu().numpy().tolist()
            best_score = float(scores_tensor[best_i].detach().cpu().item())
            best_phrase = phrases[best_i]

            # Expose all detections for multi-object use, while keeping
            # the original single-detection fields for compatibility.
            all_boxes = boxes_tensor.detach().cpu().numpy().tolist()
            all_scores = scores_tensor.detach().cpu().numpy().tolist()

            return {
                "req_id": msg["req_id"],
                "found": True,
                "bbox": best_box,
                "score": best_score,
                "class": best_phrase,
                "bboxes": all_boxes,
                "scores": all_scores,
                "classes": phrases,
            }
        finally:
            img_shm.close()

    def _handle_owl_v2(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_owl_v2()
        img_shm = SharedMemory(name=msg["image_name"])
        try:
            import torch

            image = np.ndarray(
                shape=tuple(msg["image_shape"]), dtype=np.dtype(msg["image_dtype"]), buffer=img_shm.buf
            )
            fcls = msg.get("filter_class")
            queries = fcls if isinstance(fcls, list) else ([fcls] if fcls is not None else None)
            threshold = float(msg.get("threshold", 0.1))
            dets = self._owl_v2.predict(image=image, classes=queries, threshold=threshold)
            if dets.num_detections == 0:
                return {"req_id": msg["req_id"], "found": False}

            boxes_tensor = dets.boxes
            scores_tensor = dets.logits
            phrases = list(dets.phrases)

            best_i = int(torch.argmax(scores_tensor).item())
            boxes_np = np.clip(boxes_tensor.detach().cpu().numpy(), 0.0, 1.0).tolist()
            scores_np = scores_tensor.detach().cpu().numpy().tolist()
            best_box = boxes_np[best_i]
            best_score = float(scores_np[best_i])
            best_phrase = phrases[best_i]
            return {
                "req_id": msg["req_id"],
                "found": True,
                "bbox": best_box,
                "score": best_score,
                "class": best_phrase,
                "bboxes": boxes_np,
                "scores": scores_np,
                "classes": phrases,
            }
        finally:
            img_shm.close()

    def _handle_sam(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_sam()
        img_shm = SharedMemory(name=msg["image_name"])
        try:
            image = np.ndarray(
                shape=tuple(msg["image_shape"]), dtype=np.dtype(msg["image_dtype"]), buffer=img_shm.buf
            )
            bbox = msg["bbox"]
            masks = self._sam.segment_bbox(image, bbox)
            if masks is None or masks.shape[0] == 0:
                return {"req_id": msg["req_id"], "found": False}
            # Return first mask via shared memory as uint8
            m = masks[0].astype(np.uint8)
            out_shm = SharedMemory(create=True, size=int(m.size * m.dtype.itemsize))
            out_arr = np.ndarray(shape=m.shape, dtype=m.dtype, buffer=out_shm.buf)
            np.copyto(out_arr, m)
            return {
                "req_id": msg["req_id"],
                "found": True,
                "mask_name": out_shm.name,
                "mask_shape": list(m.shape),
                "mask_dtype": str(m.dtype),
            }
        finally:
            img_shm.close()

    def run(self) -> None:
        # Lazy-load on first request
        while True:
            msg = self._request_q.get()
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "shutdown":
                break

            task = msg.get("task")
            if task == "pointnav_rnn":
                resp = self._handle_pointnav(msg)
            elif task == "stairnav_rnn":
                resp = self._handle_stairnav(msg)
            elif task == "yolov7":
                resp = self._handle_yolov7(msg)
            elif task == "owl_v2":
                resp = self._handle_owl_v2(msg)
            elif task == "mobile_sam":
                resp = self._handle_sam(msg)
            elif task == "blip2itm":
                resp = self._handle_blip2itm(msg)
            else:
                resp = {"req_id": msg.get("req_id"), "error": f"unknown task {task}"}

            # Route response to the worker-specific queue
            try:
                wid = int(msg["worker_id"])  # require valid worker_id; 0-based ids
            except Exception:
                wid = -1
            q = self._reply_queues.get(wid)
            if q is None:
                print(f"[GPUManager] Unknown worker_id={wid} for req_id={msg.get('req_id')} - dropping response")
            else:
                q.put(resp)
