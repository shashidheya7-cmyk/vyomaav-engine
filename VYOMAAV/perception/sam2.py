"""SAM2 (Segment Anything 2) Subsystem for Fine-Grained Object Masking."""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional


class SAM2Predictor:
    """Generates pixel-accurate object masks given 2D bounding boxes using SAM2."""

    def __init__(self, model_id: str = "facebook/sam2-hiera-tiny", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_id = model_id
        print(f"Loading SAM2 ({self.model_id}) onto {self.device}...")

        try:
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
            print("SAM2 loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load SAM2 weights ({e}). Using bounding box masks.")
            self.processor = None
            self.model = None

    def segment_boxes(self, image: Image.Image, boxes: List[List[float]]) -> List[np.ndarray]:
        """Generates binary masks (height, width) for a list of bounding boxes [x1, y1, x2, y2]."""
        w, h = image.size
        masks = []

        if self.model is not None and self.processor is not None and len(boxes) > 0:
            try:
                input_boxes = [[box] for box in boxes]
                inputs = self.processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)

                if hasattr(self.processor, "post_process_masks"):
                    pred_masks = self.processor.post_process_masks(
                        outputs.pred_masks,
                        inputs.original_sizes,
                        inputs.reshaped_input_sizes
                    )
                    for i in range(len(boxes)):
                        mask_tensor = pred_masks[i][0][0]
                        mask_np = (mask_tensor.cpu().numpy() > 0.0).astype(np.uint8)
                        masks.append(mask_np)

                    if len(masks) == len(boxes):
                        return masks

            except Exception as exc:
                print(f"SAM2 segmentation warning ({exc}). Generating bounding box spatial masks.")

        # Spatial bounding box binary mask fallback
        for box in boxes:
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)

        return masks


SAM2Pipeline = SAM2Predictor
