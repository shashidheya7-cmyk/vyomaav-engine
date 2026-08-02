"""VYOMAAV Canonical Spatial Coordinate System & Transform Engine."""

import numpy as np
from enum import Enum
from typing import Tuple


class CoordinateFrame(Enum):
    CAMERA_OPENCV = "camera_opencv"  # +X Right, +Y Down, +Z Forward (Depth)
    WORLD_CANONICAL = "world_canonical"  # +X Right, +Y Up, -Z Forward (OpenGL/Blender Standard)


class CoordinateTransformEngine:
    """Converts 3D points, normals, and camera poses between Camera and Canonical World Space."""

    # Transformation matrix from Camera (OpenCV) to Canonical World (OpenGL/Blender)
    T_CAM_TO_WORLD = np.array([
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0]
    ], dtype=np.float32)

    @classmethod
    def transform_points_to_canonical(cls, points_cam: np.ndarray) -> np.ndarray:
        """Transforms N x 3 points from Camera frame to Canonical World frame."""
        if points_cam.shape[-1] != 3:
            raise ValueError(f"Expected N x 3 points tensor, got shape {points_cam.shape}")
        
        # Flip Y and Z axes
        points_canonical = points_cam.copy()
        points_canonical[..., 1] *= -1.0
        points_canonical[..., 2] *= -1.0
        return points_canonical

    @classmethod
    def transform_normals_to_canonical(cls, normals_cam: np.ndarray) -> np.ndarray:
        """Transforms surface normal vectors into Canonical World frame and re-normalizes."""
        normals_world = cls.transform_points_to_canonical(normals_cam)
        norm_len = np.linalg.norm(normals_world, axis=-1, keepdims=True) + 1e-6
        return normals_world / norm_len
