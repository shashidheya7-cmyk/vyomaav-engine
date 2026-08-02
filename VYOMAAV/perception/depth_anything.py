"""Depth Anything V2 Perception Module for Dense Depth Estimation."""

import torch
import numpy as np
from PIL import Image
from typing import Optional
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthAnythingV2Predictor:
    """Predicts dense spatial depth maps from RGB images using Depth Anything V2."""

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Base-hf", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_id = model_id
        print(f"Loading Depth Anything V2 ({self.model_id}) onto {self.device}...")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device)
            print("Depth Anything V2 loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load Depth Anything V2 weights ({e}). Using procedural depth gradient.")
            self.processor = None
            self.model = None

    def predict_depth(self, image: Image.Image) -> np.ndarray:
        """Generates a dense float32 depth map for an input image."""
        if self.model is not None and self.processor is not None:
            try:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predicted_depth = outputs.predicted_depth

                # Rescale depth map back to original image dimensions (height, width)
                h, w = image.size[1], image.size[0]
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()
                return prediction
            except Exception as exc:
                print(f"Depth inference error ({exc}). Falling back to spatial gradient.")

        # Procedural fallback depth gradient
        w, h = image.size
        return np.tile(np.linspace(0.1, 1.0, h)[:, None], (1, w)).astype(np.float32)


# Alias for backward compatibility
DepthAnythingPredictor = DepthAnythingV2Predictor
