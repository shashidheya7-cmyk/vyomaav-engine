"""VYOMAAV Neural Geometry Completion Interfaces."""

import trimesh
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from core.types import FullGeometryPrediction


class INeuralGeometryCompleter(ABC):
    """Abstract interface for generative 3D shape completion backends (TRELLIS, Hunyuan3D, OpenLRM)."""

    @abstractmethod
    def complete_geometry(
        self,
        partial_mesh: trimesh.Trimesh,
        geom_prediction: FullGeometryPrediction
    ) -> trimesh.Trimesh:
        """Takes partial front-facing mesh and predicts complete watertight 3D manifold."""
        pass
