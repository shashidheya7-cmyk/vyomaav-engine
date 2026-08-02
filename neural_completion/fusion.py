"""VYOMAAV Completion Fusion & Provenance Tracking Engine."""

import trimesh
import numpy as np
from typing import Dict, Any, List
from core.types import CompletionPrediction


class CompletionFusionEngine:
    """Fuses outputs from multiple completion backends while tracking ground-truth provenance."""

    def fuse_predictions(self, predictions: List[CompletionPrediction]) -> CompletionPrediction:
        """Selects or blends completion predictions based on confidence scores."""
        if not predictions:
            raise ValueError("No completion predictions provided for fusion.")

        # Select highest confidence prediction
        best_pred = max(predictions, key=lambda p: p.completion_confidence)
        
        best_pred.metadata["fused_multi_backend_count"] = len(predictions)
        best_pred.metadata["selected_primary_backend"] = best_pred.metadata.get("backend", "unknown")

        return best_pred
