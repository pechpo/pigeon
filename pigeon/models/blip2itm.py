from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

from transformers import Blip2ForImageTextRetrieval, Blip2Processor


class BLIP2ITM:
    """BLIP2 image-text matching via HuggingFace BLIP-2 ITM only."""

    def __init__(
        self,
        model_id: str = "Salesforce/blip2-itm-vit-g",  # maybe Salesforce/blip2-itm-vit-g-coco better?
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        if Blip2Processor is None or Blip2ForImageTextRetrieval is None:
            raise ImportError("transformers with Blip2 is required for BLIP2ITM.")

        self.device = device
        self._hf_blip2_proc = Blip2Processor.from_pretrained(model_id, torch_dtype=torch.float16)
        self._hf_blip2_model = Blip2ForImageTextRetrieval.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
        """
        pil_img = Image.fromarray(image)
        inputs = self._hf_blip2_proc(images=pil_img, text=txt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._hf_blip2_model(**inputs)
        # logits_per_image shape: [batch, num_text]; single item here
        cosine = float(outputs.logits_per_image[0, 0].item())

        return cosine


class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        print(f"BLIP2ITMClient.cosine: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])
