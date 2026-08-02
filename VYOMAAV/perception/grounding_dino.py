"""Grounding DINO Open-Vocabulary Object Detection Subsystem."""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class GroundingDINOPredictor:
    """Predicts 2D bounding boxes and text labels for dynamic open-world scene entities."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_id = model_id
        print(f"Loading Grounding DINO ({self.model_id}) onto {self.device}...")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
            print("Grounding DINO loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load Grounding DINO weights ({e}).")
            self.processor = None
            self.model = None

    def predict(
        self,
        image: Image.Image,
        text_prompt: str = "chair . lamp . table . vase . tv . console . furniture . decor .",
        box_threshold: float = 0.20,
        text_threshold: float = 0.20
    ) -> Dict[str, Any]:
        """Detects all dynamic objects in the scene."""
        if self.model is None or self.processor is None:
            w, h = image.size
            return {"boxes": [[0, 0, w, h]], "labels": ["object"]}

        formatted_prompt = text_prompt.lower().strip()
        if not formatted_prompt.endswith("."):
            formatted_prompt += " ."

        try:
            inputs = self.processor(images=image, text=formatted_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = [torch.tensor([image.size[1], image.size[0]])]

            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes
                )[0]
            except TypeError:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes
                )[0]

            boxes = results["boxes"].cpu().numpy().tolist()
            labels = results.get("labels", [f"object_{i}" for i in range(len(boxes))])

            return {"boxes": boxes, "labels": labels}

        except Exception as exc:
            print(f"Grounding DINO inference warning ({exc}). Using full image bounding frame.")
            w, h = image.size
            return {"boxes": [[0, 0, w, h]], "labels": ["scene_object"]}


GroundingDINOPipeline = GroundingDINOPredictor
