"""Microsoft TRELLIS image-to-3D mesh reconstruction provider."""

from __future__ import annotations

import gc
from typing import Any, Tuple, Optional
import numpy as np
import trimesh
from PIL import Image

from ....core.exceptions import FusionEngineError
from ....registry.registry import FUSION_REGISTRY
from ..base import BaseFusionProvider


@FUSION_REGISTRY.register_module("TRELLIS")
class TRELLISProvider(BaseFusionProvider):
    """Run Microsoft's image-conditioned TRELLIS pipeline to produce 3D surface meshes."""

    def initialize(self) -> None:
        try:
            import torch
            from trellis.pipelines import TrellisImageTo3DPipeline
        except ModuleNotFoundError as exc:
            raise FusionEngineError("TRELLIS requires PyTorch and its official repository package.") from exc
        
        self.device = str(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model_id = str(self.config.get("model_id", "microsoft/TRELLIS-image-large"))
        
        try:
            print(f"Loading TRELLIS ({model_id}) onto {self.device}...")
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_id).to(self.device)
        except Exception as exc:
            raise FusionEngineError(f"Failed to load TRELLIS weights ({model_id}): {exc}") from exc

    def extract_trellis_mesh(self, pipeline_output: Any) -> trimesh.Trimesh:
        """Extracts a watertight 3D surface mesh directly from TRELLIS output."""
        meshes = pipeline_output.get("meshes") if isinstance(pipeline_output, dict) else getattr(pipeline_output, "meshes", None)
        raw_mesh = meshes[0] if meshes else pipeline_output
        
        vertices = raw_mesh.vertices.cpu().numpy() if hasattr(raw_mesh.vertices, "cpu") else np.asarray(raw_mesh.vertices)
        faces = raw_mesh.faces.cpu().numpy() if hasattr(raw_mesh.faces, "cpu") else np.asarray(raw_mesh.faces)
        
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    def reconstruct(self, views: Any, cameras: dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Executes TRELLIS generation and returns 3D vertices, faces, and colors."""
        try:
            pil_img = Image.fromarray(np.clip(views[0].detach().cpu().permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8))
            outputs = self.pipeline.run(
                pil_img,
                seed=int(self.config.get("seed", 42)),
                sparse_structure_sampler_params={"steps": int(self.config.get("sparse_steps", 12))},
                slat_sampler_params={"steps": int(self.config.get("slat_steps", 12))}
            )
            mesh = self.extract_trellis_mesh(outputs)
            return mesh.vertices, mesh.faces, None
        except Exception as exc:
            raise FusionEngineError(f"TRELLIS mesh reconstruction failed: {exc}") from exc

    def cleanup(self) -> None:
        if hasattr(self, "pipeline"):
            del self.pipeline
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass
