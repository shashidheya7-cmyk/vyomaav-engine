"""VYOMAAV Reconstruction Engine: Pluggable Backend Interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np

class BaseReconstructionBackend(ABC):
    """Abstract interface for pluggable 3D reconstruction backends (TRELLIS, Hunyuan3D, etc.)."""

    def __init__(self, backend_name: str, device: Optional[str] = None):
        self.backend_name = backend_name
        self.device = device or "cpu"

    @abstractmethod
    def reconstruct(
        self,
        image: Any,
        depth: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        semantics: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generates 3D Mesh geometry, normals, topology, and UV mappings."""
        pass

    @abstractmethod
    def integrate_into_somg(self, scene: Any, entity_id: str, reconstruction_result: Dict[str, Any]) -> Any:
        """Attaches reconstructed 3D geometry into an SOMG entity node."""
        pass
