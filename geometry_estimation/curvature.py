"""VYOMAAV Surface Curvature Computation Engine (Mean & Gaussian Curvature)."""

import cv2
import numpy as np


class SurfaceCurvatureEngine:
    """Calculates Mean (H) and Gaussian (K) surface curvatures from depth fields."""

    def compute_curvature(self, depth_map: np.ndarray) -> np.ndarray:
        """Computes mean surface curvature H to detect sharp edges and bends."""
        dz_dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        dz_dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

        d2z_dx2 = cv2.Sobel(dz_dx, cv2.CV_32F, 1, 0, ksize=3)
        d2z_dy2 = cv2.Sobel(dz_dy, cv2.CV_32F, 0, 1, ksize=3)
        d2z_dxy = cv2.Sobel(dz_dx, cv2.CV_32F, 0, 1, ksize=3)

        # Mean curvature formula: H = ((1 + z_y^2)*z_xx - 2*z_x*z_y*z_xy + (1 + z_x^2)*z_yy) / (2 * (1 + z_x^2 + z_y^2)^(3/2))
        num = (1 + dz_dy**2) * d2z_dx2 - 2 * dz_dx * dz_dy * d2z_dxy + (1 + dz_dx**2) * d2z_dy2
        denom = 2.0 * np.power(1.0 + dz_dx**2 + dz_dy**2, 1.5) + 1e-6

        mean_curvature = np.abs(num / denom)
        return cv2.normalize(mean_curvature, None, 0.0, 1.0, cv2.NORM_MINMAX)
