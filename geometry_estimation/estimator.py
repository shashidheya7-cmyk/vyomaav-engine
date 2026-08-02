"""VYOMAAV Modular Master Geometry Orchestrator."""

import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from core.types import FullGeometryPrediction

from perception.depth_anything import DepthAnythingV2Predictor
from geometry_estimation.curvature import SurfaceCurvatureEngine
from geometry_estimation.confidence import MultiFactorConfidenceEngine
from geometry_estimation.fusion import DedicatedGeometryFusionEngine


class GeometryEstimationEngine:
    """Master orchestrator for the Geometry Foundation Layer."""

    def __init__(self, device: Optional[str] = None):
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Initializing Geometry Foundation Engine on [{self.device.upper()}]...")

        self.depth_predictor = DepthAnythingV2Predictor(device=self.device)
        self.curvature_engine = SurfaceCurvatureEngine()
        self.confidence_engine = MultiFactorConfidenceEngine()
        self.fusion_engine = DedicatedGeometryFusionEngine()

    def estimate_gradient_normals(self, depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Calculates camera-space surface normals from depth Sobel gradients."""
        f_x, f_y = K[0, 0], K[1, 1]
        dz_dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        dz_dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

        normals = np.stack([-dz_dx * (f_x / 1000.0), -dz_dy * (f_y / 1000.0), np.ones_like(depth_map)], axis=-1)
        norm_len = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6
        return normals / norm_len

    def estimate_metric_scale_with_uncertainty(self, rel_depth: np.ndarray) -> Tuple[np.ndarray, float]:
        """Recovers physical distance metric (meters) and reports scale uncertainty variance."""
        d_min, d_max = np.min(rel_depth), np.max(rel_depth)
        norm_depth = (rel_depth - d_min) / (d_max - d_min + 1e-6)

        # Baseline physical metric scale (meters)
        metric_depth = norm_depth * 3.5 + 0.5
        
        # Estimate scale variance/uncertainty based on depth range dynamic span
        scale_uncertainty = 0.15 if (d_max - d_min) > 0.5 else 0.40
        return metric_depth.astype(np.float32), scale_uncertainty

    def predict_full_geometry(self, image: Image.Image, K: np.ndarray) -> FullGeometryPrediction:
        """Executes full geometry pipeline and passes results to the Fusion Engine."""
        # 1. Relative depth
        rel_depth = self.depth_predictor.predict_depth(image)

        # 2. Metric scale recovery with explicit uncertainty estimation
        metric_depth, scale_uncertainty = self.estimate_metric_scale_with_uncertainty(rel_depth)

        # 3. Gradient-based surface normals
        grad_normals = self.estimate_gradient_normals(metric_depth, K)

        # 4. Surface curvature
        curvature = self.curvature_engine.compute_curvature(metric_depth)

        # 5. Multi-factor confidence fusion
        confidence = self.confidence_engine.fuse_confidence(metric_depth, grad_normals, curvature)

        # 6. Pass all signals into the Dedicated Geometry Fusion Engine
        return self.fusion_engine.fuse(
            rel_depth=rel_depth,
            metric_depth=metric_depth,
            scale_uncertainty=scale_uncertainty,
            gradient_normals=grad_normals,
            learned_normals=None,  # Placeholder for Marigold / Metric3D learned normals
            curvature=curvature,
            confidence_map=confidence,
            K=K
        )
