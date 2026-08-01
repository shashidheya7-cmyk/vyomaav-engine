"""VYOMAAV Perception: CLIP Feature Embedder Integration."""
import os, torch, numpy as np, cv2
from typing import Dict, List, Any, Optional, Union
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPPredictor:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading CLIP ({model_id}) onto {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def _extract_tensor(self, features: Any) -> torch.Tensor:
        if isinstance(features, torch.Tensor): return features
        if hasattr(features, "image_embeds") and features.image_embeds is not None: return features.image_embeds
        if hasattr(features, "text_embeds") and features.text_embeds is not None: return features.text_embeds
        if hasattr(features, "pooler_output") and features.pooler_output is not None: return features.pooler_output
        return features[0]

    @torch.no_grad()
    def get_image_embedding(self, image: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(image, str) and os.path.exists(image): pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray): pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image)
        else: pil_img = Image.new("RGB", (224, 224))
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        features = self._extract_tensor(self.model.get_image_features(**inputs))
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    def integrate_into_somg(self, scene: Any, entity_id: str, embedding: np.ndarray) -> Any:
        if hasattr(scene, "base_graph") and hasattr(scene.base_graph, "nodes"):
            nodes = scene.base_graph.nodes
            if isinstance(nodes, dict) and entity_id in nodes:
                nodes[entity_id].clip_embedding = embedding.tolist()
        return scene
