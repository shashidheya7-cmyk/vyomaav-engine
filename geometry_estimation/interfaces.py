"""VYOMAAV Granular Backend Interfaces for Geometry Subsystems."""

import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict, Any, Optional, Tuple


class IDepthPredictor(ABC):
    """Interface for relative depth estimators (Depth Anything, Depth Anything V2)."""

    @abstractmethod
    def predict_depth(self, image: Image.Image) -> np.ndarray:
        pass


class IScalePredictor(ABC):
    """Interface for physical metric scale recovery estimators (Metric3D, Ground Plane)."""

    @abstractmethod
    def recover_scale(self, relative_depth: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, float]:
        """Returns (metric_depth_map, scale_uncertainty_variance)."""
        pass


class INormalPredictor(ABC):
    """Interface for surface normal predictors (Gradient, Metric3D, Marigold)."""

    @abstractmethod
    def predict_normals(self, depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
        pass


class ICurvaturePredictor(ABC):
    """Interface for surface curvature estimators."""

    @abstractmethod
    def compute_curvature(self, depth_map: np.ndarray) -> np.ndarray:
        pass


class IConfidencePredictor(ABC):
    """Interface for uncertainty and confidence estimation."""

    @abstractmethod
    def compute_confidence(
        self, depth: np.ndarray, normals: np.ndarray, curvature: np.ndarray
    ) -> np.ndarray:
        pass
