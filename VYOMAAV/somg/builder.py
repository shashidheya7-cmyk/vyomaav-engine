"""
VYOMAAV Base Model Engine
Module: somg.builder

Converts raw perception detections (bounding boxes, centroids, semantic classifications,
confidence scores) into structured SOMGEntity objects.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from somg.entity import (
    SOMGEntity, SemanticComponent, SpatialComponent,
    MaterialComponent, PhysicsComponent, UncertaintyComponent
)


@dataclass
class PerceptionObservation:
    """Raw observation frame output emitted by perception models."""
    observation_id: str
    label: str
    class_id: int
    confidence: float
    bbox_min: List[float]  # [x_min, y_min, z_min]
    bbox_max: List[float]  # [x_max, y_max, z_max]
    estimated_mass_kg: float = 1.0
    material_type: str = "generic"
    aleatoric_noise: float = 0.05


class SOMGEntityBuilder:
    """Transforms raw perception observations into strongly typed SOMGEntities."""

    @staticmethod
    def from_observation(obs: PerceptionObservation, entity_id_prefix: str = "ent") -> SOMGEntity:
        entity_id = f"{entity_id_prefix}_{obs.label}_{obs.observation_id}"
        
        return SOMGEntity(
            entity_id=entity_id,
            version=1,
            semantic=SemanticComponent(
                label=obs.label,
                class_id=obs.class_id,
                confidence=obs.confidence
            ),
            spatial=SpatialComponent(
                bbox_min=list(obs.bbox_min),
                bbox_max=list(obs.bbox_max)
            ),
            material=MaterialComponent(
                material_type=obs.material_type
            ),
            physics=PhysicsComponent(
                mass_kg=obs.estimated_mass_kg,
                is_static=False
            ),
            uncertainty=UncertaintyComponent(
                aleatoric_noise=obs.aleatoric_noise,
                epistemic_risk=1.0 - obs.confidence,
                is_inferred=False
            )
        )