"""VYOMAAV Modular Master Neural Completion Engine."""

import trimesh
from typing import Dict, Any, Optional
from core.types import FullGeometryPrediction, CompletionPrediction
from neural_completion.trellis import TrellisCompletionAdapter
from neural_completion.hunyuan3d import Hunyuan3DCompletionAdapter
from neural_completion.fusion import CompletionFusionEngine


class MasterNeuralCompletionEngine:
    """Master orchestrator for multi-backend generative neural completion."""

    def __init__(self, backend: str = "trellis", device: Optional[str] = None):
        self.device = device or "cuda"
        self.backend_name = backend.lower()
        print(f"Initializing Master Neural Completion Engine [{self.backend_name.upper()}]...")

        self.backends = {
            "trellis": TrellisCompletionAdapter(device=self.device),
            "hunyuan3d": Hunyuan3DCompletionAdapter(device=self.device)
        }
        self.fusion_engine = CompletionFusionEngine()

    def complete_geometry(
        self,
        partial_mesh: trimesh.Trimesh,
        geom_pred: FullGeometryPrediction
    ) -> CompletionPrediction:
        """Executes selected completion backend and preserves observed vs inferred provenance."""
        primary_backend = self.backends.get(self.backend_name, self.backends["trellis"])
        pred = primary_backend.complete(partial_mesh, geom_pred)
        
        return self.fusion_engine.fuse_predictions([pred])
