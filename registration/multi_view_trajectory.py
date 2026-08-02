"""VYOMAAV Multi-View Camera Trajectory Simulator & Scalability Engine."""

import numpy as np
from typing import List, Tuple
from core.types import PosePrediction


class MultiViewTrajectoryEngine:
    """Generates synthetic camera orbits (N frames) to stress-test TSDF volumetric fusion scalability."""

    def generate_circular_orbit(
        self,
        num_frames: int = 12,
        radius: float = 2.5,
        height: float = 0.5
    ) -> List[PosePrediction]:
        """Generates N-view camera orbit around world origin (0, 0, 1.5)."""
        poses = []
        angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

        for i, theta in enumerate(angles):
            # Camera Center in World Space
            cam_x = radius * np.cos(theta)
            cam_y = radius * np.sin(theta)
            cam_z = height

            # Rotation matrix looking toward origin (0, 0, 1.5)
            forward = np.array([-cam_x, -cam_y, 1.5 - cam_z], dtype=np.float32)
            forward /= np.linalg.norm(forward) + 1e-6

            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(forward, up)
            right /= np.linalg.norm(right) + 1e-6
            up = np.cross(right, forward)

            R_world = np.stack([right, up, -forward], axis=-1)
            t_world = np.array([[cam_x], [cam_y], [cam_z]], dtype=np.float32)

            pose = PosePrediction(
                R=R_world,
                t=t_world,
                confidence=0.92,
                inlier_ratio=0.89,
                reprojection_error=0.35,
                metadata={"frame_index": i, "orbit_angle_rad": float(theta)}
            )
            poses.append(pose)

        return poses
