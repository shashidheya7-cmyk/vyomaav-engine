"""VYOMAAV Production-Grade Geometry, Registration, & Completion Diagnostic Contracts."""

import trimesh
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class MatchPrediction:
    """Standardized correspondence prediction with uncertainty metrics."""
    keypoints1: np.ndarray          # Nx2 array
    keypoints2: np.ndarray          # Nx2 array
    match_confidence: np.ndarray    # N-length confidence scores [0, 1]
    inlier_mask: np.ndarray         # N-length boolean RANSAC mask
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def inlier_count(self) -> int:
        return int(np.sum(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        total = len(self.inlier_mask)
        return float(self.inlier_count / total) if total > 0 else 0.0


@dataclass
class PosePrediction:
    """Camera extrinsics prediction with full covariance and error diagnostics."""
    R: np.ndarray                   # 3x3 Rotation
    t: np.ndarray                   # 3x1 Translation
    confidence: float               # Aggregate confidence [0, 1]
    inlier_ratio: float
    reprojection_error: float
    essential_matrix: Optional[np.ndarray] = None
    fundamental_matrix: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FullGeometryPrediction:
    """Research-grade geometry prediction object carrying multi-factor diagnostics."""
    depth: np.ndarray                           # Relative depth grid HxW
    metric_depth: np.ndarray                    # True physical distance grid HxW (meters)
    normals: np.ndarray                         # Surface normal vectors HxWx3
    depth_confidence: np.ndarray                # Depth uncertainty grid HxW [0, 1]
    normal_confidence: np.ndarray               # Normal reliability grid HxW [0, 1]
    geometry_confidence: np.ndarray             # Composite fused confidence HxW [0, 1]
    uncertainty: np.ndarray                     # Combined spatial variance map HxW
    camera_K: np.ndarray                        # Intrinsic calibration matrix 3x3

    curvature: Optional[np.ndarray] = None      # Surface mean/Gaussian curvature HxW
    edges: Optional[np.ndarray] = None          # Geometric depth/normal discontinuities HxW
    occlusion: Optional[np.ndarray] = None      # Occlusion likelihood map HxW
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionPrediction:
    """Standardized Neural Completion prediction preserving ground-truth vs inferred provenance."""
    observed_mesh: trimesh.Trimesh              # Camera-captured TSDF ground truth geometry
    inferred_mesh: trimesh.Trimesh              # Generative neural prior completed geometry
    fused_mesh: trimesh.Trimesh                 # Unified combined mesh
    vertex_provenance_mask: np.ndarray          # Boolean mask: False = Observed, True = AI Inferred
    completion_confidence: float               # Model confidence score [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)
