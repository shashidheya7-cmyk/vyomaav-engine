"""VYOMAAV Abstract Neural Completion Model Interface."""

import trimesh
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from core.types import FullGeometryPrediction, CompletionPrediction


class INeuralGeometryCompleter(ABC):
    """Abstract interface for generative 3D shape completion backends (TRELLIS, Hunyuan3D, OpenLRM)."""

    @abstractmethod
    def complete(
        self,
        partial_mesh: trimesh.Trimesh,
        geom_pred: FullGeometryPrediction
    ) -> CompletionPrediction:
        """Takes partial front-facing mesh and predicts complete watertight 3D manifold."""
        pass
