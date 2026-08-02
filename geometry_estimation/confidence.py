"""VYOMAAV Multi-Factor Confidence & Uncertainty Fusion Engine."""

import cv2
import numpy as np


class MultiFactorConfidenceEngine:
    """Fuses depth variance, normal variance, and edge gradients into a robust confidence map."""

    def fuse_confidence(
        self,
        depth_map: np.ndarray,
        normals: np.ndarray,
        curvature: np.ndarray
    ) -> np.ndarray:
        """Fuses multiple diagnostic cues into a composite confidence grid C in [0, 1]."""
        # 1. Depth Gradient Confidence (low confidence at high depth discontinuities)
        depth_grad = cv2.Laplacian(depth_map, cv2.CV_32F)
        conf_depth = np.exp(-np.abs(depth_grad) / (np.std(depth_grad) + 1e-6))

        # 2. Normal Consistency Confidence (low confidence where normal Z flips sharply)
        normal_z = np.abs(normals[:, :, 2])
        conf_normals = np.clip(normal_z, 0.2, 1.0)

        # 3. Curvature Penalty (penalize extreme noise regions)
        conf_curvature = 1.0 - np.clip(curvature, 0.0, 0.8)

        # Weighted Composite Fusion
        fused_confidence = 0.5 * conf_depth + 0.3 * conf_normals + 0.2 * conf_curvature
        return np.clip(fused_confidence, 0.05, 1.0).astype(np.float32)
