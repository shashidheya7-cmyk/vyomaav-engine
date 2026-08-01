"""VYOMAAV Spatial Object Mesh Graph (SOMG) Entity Definitions."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SpatialComponent:
    bbox_min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bbox_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    @property
    def center(self) -> List[float]:
        return [(a + b) / 2.0 for a, b in zip(self.bbox_min, self.bbox_max)]

    @property
    def dimensions(self) -> List[float]:
        return [abs(b - a) for a, b in zip(self.bbox_min, self.bbox_max)]

@dataclass
class PhysicsComponent:
    is_static: bool = True
    mass: float = 1.0
    friction: float = 0.5

class SOMGEntity:
    def __init__(
        self,
        entity_id: str,
        spatial: Optional[SpatialComponent] = None,
        physics: Optional[PhysicsComponent] = None
    ):
        self.entity_id = entity_id
        self.spatial = spatial or SpatialComponent()
        self.physics = physics or PhysicsComponent()
        self.semantic_label: str = "object"
        self.confidence: float = 1.0
        self.clip_embedding: List[float] = []
        self.mesh_geometry: Optional[Dict[str, Any]] = None
        self.attributes: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "semantic_label": self.semantic_label,
            "confidence": self.confidence,
            "bbox_min": self.spatial.bbox_min,
            "bbox_max": self.spatial.bbox_max,
            "clip_embedding_len": len(self.clip_embedding),
            "is_static": self.physics.is_static
        }
