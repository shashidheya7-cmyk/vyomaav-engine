"""VYOMAAV Perception: Depth Anything V2 Metric Depth Estimator."""
import os, json, torch, numpy as np, cv2
from typing import Dict, Any, Optional, Union
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthAnythingPredictor:
    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Base-hf", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading Depth Anything V2 ({model_id}) onto {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, image: Union[str, np.ndarray, torch.Tensor], output_dir: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(image, str) and os.path.exists(image): pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray): pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image)
        else: pil_img = Image.new("RGB", (640, 480))
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        depth = self.model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=pil_img.size[::-1], mode="bicubic", align_corners=False).squeeze().cpu().numpy()
        res = {"depth": depth, "resolution": pil_img.size}
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, "depth.npy"), depth)
            res["depth_path"] = os.path.join(output_dir, "depth.npy")
        return res
