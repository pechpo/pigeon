from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch._inductor.config
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.models.owlv2.modeling_owlv2 import Owlv2ObjectDetectionOutput

from pigeon.models.detections import ObjectDetections


class OWLv2:
    def __init__(self, model_id: str = "google/owlv2-base-patch16-ensemble", device: str = "cuda"):
        torch_device = torch.device(device)
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id, torch_dtype=torch.float16).to(torch_device)
        self.model.eval()
        self.device = torch_device
        # Cache text encodings since prompts rarely change.
        self._text_cache: Dict[Tuple[str, ...], Dict[str, torch.Tensor]] = {}

    def _get_cached_text(self, queries: List[str]) -> Dict[str, torch.Tensor]:
        """Return device-ready text ids, masks, and embeddings for the given queries."""
        key = tuple(queries)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached

        text_inputs = self.processor(text=[queries], return_tensors="pt")
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.inference_mode():
            text_features = self.model.owlv2.get_text_features(
                input_ids=text_inputs["input_ids"], attention_mask=text_inputs.get("attention_mask")
            )
        text_embeds = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

        cached = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs.get("attention_mask"),
            "text_embeds": text_embeds,
        }
        self._text_cache[key] = cached
        return cached

    def predict(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None,
        threshold: float = 0.1,
    ) -> ObjectDetections:
        """Run OWLv2 zero-shot detection on an RGB image."""
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Expected RGB image ndarray for OWLv2 predict().")
        image_rgb = image
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("OWLv2 requires an HxWx3 RGB image.")
        image_pil = Image.fromarray(image_rgb)
        queries = classes if classes and len(classes) > 0 else ["person"]
        text_labels = [queries]

        # Reuse text embeddings when the queries repeat.
        text_state = self._get_cached_text(queries)
        use_custom_size = True 
        if use_custom_size:
            image_inputs = self.processor(images=image_pil, return_tensors="pt", size={"height": 480, "width": 640})
        else:
            image_inputs = self.processor(images=image_pil, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self.device)

        with torch.inference_mode():
            feature_map, vision_outputs = self.model.image_embedder(pixel_values=pixel_values, interpolate_pos_encoding=use_custom_size)
            batch_size, h, w, hidden = feature_map.shape
            image_feats = feature_map.reshape(batch_size, h * w, hidden)

            input_ids = text_state["input_ids"]
            max_queries = input_ids.shape[0] // batch_size
            query_embeds = text_state["text_embeds"].reshape(batch_size, max_queries, -1)
            query_mask = input_ids.reshape(batch_size, max_queries, -1)[..., 0] > 0

            pred_logits, class_embeds = self.model.class_predictor(
                image_feats=image_feats, query_embeds=query_embeds, query_mask=query_mask
            )
            objectness_logits = self.model.objectness_predictor(image_feats)
            pred_boxes = self.model.box_predictor(image_feats, feature_map, interpolate_pos_encoding=use_custom_size)

            outputs = Owlv2ObjectDetectionOutput(
                logits=pred_logits,
                objectness_logits=objectness_logits,
                pred_boxes=pred_boxes,
                text_embeds=query_embeds,
                image_embeds=feature_map,
                class_embeds=class_embeds,
                text_model_output=None,
                vision_model_output=vision_outputs,
            )
        target_sizes = torch.tensor([(image_pil.height, image_pil.width)], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold, text_labels=text_labels
        )
        result = results[0]
        boxes_px = result["boxes"]
        scores = result["scores"]
        labels = [str(l) for l in result["text_labels"]]

        if boxes_px.numel() == 0:
            boxes_norm = torch.zeros((0, 4), dtype=torch.float32)
            scores_t = torch.zeros((0,), dtype=torch.float32)
            phrases: List[str] = []
        else:
            norm = torch.tensor([image_pil.width, image_pil.height, image_pil.width, image_pil.height], device=boxes_px.device)
            boxes_norm = (boxes_px.to(torch.float32) / norm).cpu()
            scores_t = scores.to(torch.float32).cpu()
            phrases = labels

        return ObjectDetections(boxes=boxes_norm, logits=scores_t, phrases=phrases, image_source=image_rgb, fmt="xyxy")


def _parse_classes(raw: str) -> List[str]:
    return [c.strip() for c in raw.replace(";", ",").split(",") if c.strip()]


def _gather_images(src: Path) -> List[Path]:
    if src.is_file():
        return [src]
    return sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def run_offline_inference(
    data_path: Path,
    classes: List[str],
    threshold: float,
    model_id: str,
    device: str,
    output_dir: Optional[Path],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src = data_path if data_path.is_absolute() else repo_root / data_path
    if not src.exists():
        raise FileNotFoundError(f"Path not found: {src}")
    out_dir = output_dir
    if out_dir is None:
        out_dir = (src.parent / f"{src.stem}_owlv2") if src.is_file() else (src.parent / f"{src.name}_owlv2")
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = OWLv2(model_id=model_id, device=device)
    img_paths = _gather_images(src)
    if not img_paths:
        raise ValueError(f"No images found under: {src}")

    for img_path in img_paths:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"Skip unreadable image: {img_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        dets = detector.predict(image=image_rgb, classes=classes, threshold=threshold)
        annotated = dets.annotated_frame if dets.annotated_frame is not None else image_rgb
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"Wrote: {out_path} | dets: {dets.num_detections}")
        if dets.num_detections:
            print(dets)
