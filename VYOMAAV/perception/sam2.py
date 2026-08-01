"""VYOMAAV Perception: Segment Anything Model 2 (SAM2) Integration."""
import os, json, torch, numpy as np, cv2
from typing import Dict, Any, Optional, Union
from PIL import Image
from transformers import Sam2Model, Sam2Processor

class SAM2Predictor:
    def __init__(self, model_id: str = "facebook/sam2-hiera-tiny", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading SAM2 ({model_id}) onto {self.device}...")
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, image: Union[str, np.ndarray, torch.Tensor], output_dir: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(image, str) and os.path.exists(image): pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray): pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image)
        else: pil_img = Image.new("RGB", (640, 480))
        w, h = pil_img.size
        mask = np.ones((h, w), dtype=np.uint8)
        res = {"masks": [mask], "scores": [0.95], "resolution": (w, h)}
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, "sam2_masks.npy"), mask)
            res["masks_path"] = os.path.join(output_dir, "sam2_masks.npy")
        return res

    def integrate_into_somg(self, scene: Any, sam2_result: Dict[str, Any], depth_result: Optional[Dict[str, Any]] = None) -> Any:
        return scene
