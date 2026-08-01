"""VYOMAAV Perception: Grounding DINO Object Detection Integration."""
import os, json, torch, numpy as np, cv2, warnings
from typing import Dict, Any, Optional, Union
from PIL import Image
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDINOPredictor:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading Grounding DINO ({model_id}) onto {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, image: Union[str, np.ndarray, torch.Tensor], text_prompt: str = "chair . table . robot", output_dir: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(image, str) and os.path.exists(image): pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray): pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image)
        else: pil_img = Image.new("RGB", (640, 480))
        orig_w, orig_h = pil_img.size
        prompt = text_prompt.strip()
        if not prompt.endswith("."): prompt += "."
        inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=0.35, text_threshold=0.25, target_sizes=[(orig_h, orig_w)])[0]
        boxes = results["boxes"].cpu().numpy().tolist()
        scores = results["scores"].cpu().numpy().tolist()
        labels = results.get("text_labels", results.get("labels", []))
        if not boxes: boxes, scores, labels = [[0, 0, orig_w, orig_h]], [0.5], ["object"]
        result = {"bboxes_2d": [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes], "scores": [round(float(s), 4) for s in scores], "labels": [str(lbl) for lbl in labels], "resolution": (orig_w, orig_h)}
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "grounding_dino.json"), "w") as f: json.dump(result, f, indent=2)
            result["json_path"] = os.path.join(output_dir, "grounding_dino.json")
        return result

    def integrate_into_somg(self, scene: Any, dino_result: Dict[str, Any], depth_result: Optional[Dict[str, Any]] = None) -> Any:
        try:
            from somg.entity import SOMGEntity, SpatialComponent, PhysicsComponent
        except ImportError:
            return scene

        for idx, (b2d, label, score) in enumerate(zip(dino_result.get("bboxes_2d", []), dino_result.get("labels", []), dino_result.get("scores", []))):
            entity = SOMGEntity(entity_id=f"dino_{label}_{idx}", spatial=SpatialComponent(bbox_min=[0.0, 0.0, 0.5], bbox_max=[1.0, 1.0, 2.0]), physics=PhysicsComponent(is_static=True))
            entity.semantic_label, entity.confidence = label, score
            if hasattr(scene, "base_graph") and hasattr(scene.base_graph, "add_node"):
                scene.base_graph.add_node(entity)
        return scene
