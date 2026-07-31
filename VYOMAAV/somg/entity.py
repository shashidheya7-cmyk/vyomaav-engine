"""
VYOMAAV Base Model Engine
Module: somg.entity (Refactored V2)

Preserves discrete component objects in runtime SOMGEntities and adds version tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SemanticComponent:
    label: str = ""
    class_id: int = 0
    confidence: float = 1.0


@dataclass
class SpatialComponent:
    bbox_min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bbox_max: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    transform_matrix: Optional[List[float]] = None
    sdf_ref: Optional[str] = None


@dataclass
class MaterialComponent:
    material_type: str = "generic"
    roughness: float = 0.5
    metallic: float = 0.0
    albedo_rgb: Optional[List[float]] = None


@dataclass
class PhysicsComponent:
    mass_kg: float = 1.0
    friction: float = 0.5
    is_static: bool = False
    restitution: Optional[float] = None


@dataclass
class UncertaintyComponent:
    aleatoric_noise: float = 0.0
    epistemic_risk: float = 0.0
    is_inferred: bool = False


@dataclass
class SOMGEntity:
    """Runtime entity maintaining versioned discrete component instances."""
    entity_id: str
    version: int = 1
    semantic: SemanticComponent = field(default_factory=SemanticComponent)
    spatial: SpatialComponent = field(default_factory=SpatialComponent)
    material: MaterialComponent = field(default_factory=MaterialComponent)
    physics: PhysicsComponent = field(default_factory=PhysicsComponent)
    uncertainty: UncertaintyComponent = field(default_factory=UncertaintyComponent)

    def increment_version(self) -> 'SOMGEntity':
        """Creates a version-incremented copy of the entity."""
        return SOMGEntity(
            entity_id=self.entity_id,
            version=self.version + 1,
            semantic=self.semantic,
            spatial=self.spatial,
            material=self.material,
            physics=self.physics,
            uncertainty=self.uncertainty
        )