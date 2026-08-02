"""VYOMAAV Dedicated Geometry Fusion Engine."""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from core.coordinates import CoordinateTransformEngine
from core.types import FullGeometryPrediction


class DedicatedGeometryFusionEngine:
    """Fuses multi-source depth, normals, and scale into a unified canonical geometry contract."""

    def fuse(
        self,
        rel_depth: np.ndarray,
        metric_depth: np.ndarray,
        scale_uncertainty: float,
        gradient_normals: np.ndarray,
        learned_normals: Optional[np.ndarray],
        curvature: np.ndarray,
        confidence_map: np.ndarray,
        K: np.ndarray
    ) -> FullGeometryPrediction:
        """Applies confidence-weighted spatial fusion and canonical coordinate transformation."""
        
        # 1. Normal Vector Fusion (Blend learned normals with gradient normals based on confidence)
        if learned_normals is not None:
            blend_weight = np.expand_dims(confidence_map, axis=-1)
            fused_normals_cam = blend_weight * learned_normals + (1.0 - blend_weight) * gradient_normals
            norm_len = np.linalg.norm(fused_normals_cam, axis=-1, keepdims=True) + 1e-6
            fused_normals_cam /= norm_len
        else:
            fused_normals_cam = gradient_normals

        # 2. Transform Normals to Canonical World Coordinate Frame (+X Right, +Y Up, -Z Forward)
        canonical_normals = CoordinateTransformEngine.transform_normals_to_canonical(fused_normals_cam)

        # 3. Derive Geometric Edge Discontinuities
        depth_uint8 = cv2.normalize(metric_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges = cv2.Canny(depth_uint8, 40, 120).astype(np.float32) / 255.0

        # 4. Compute Spatial Uncertainty Map
        combined_uncertainty = (1.0 - confidence_map) + float(scale_uncertainty)
        uncertainty_map = np.clip(combined_uncertainty, 0.0, 1.0).astype(np.float32)

        return FullGeometryPrediction(
            depth=rel_depth.astype(np.float32),
            metric_depth=metric_depth.astype(np.float32),
            normals=canonical_normals.astype(np.float32),
            depth_confidence=confidence_map.astype(np.float32),
            normal_confidence=confidence_map.astype(np.float32),
            geometry_confidence=confidence_map.astype(np.float32),
            uncertainty=uncertainty_map,
            camera_K=K,
            curvature=curvature.astype(np.float32),
            edges=edges,
            occlusion=edges * 0.5,
            metadata={
                "canonical_coordinate_system": "OpenGL_World (+X Right, +Y Up, -Z Forward)",
                "scale_uncertainty_score": float(scale_uncertainty),
                "fusion_mode": "Confidence_Weighted_Canonical_Blend"
            }
        )
