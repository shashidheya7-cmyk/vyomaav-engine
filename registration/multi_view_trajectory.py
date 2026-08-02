"""VYOMAAV View-Consistent Multi-View Camera Trajectory & Depth Renderer."""

import numpy as np
from typing import List, Tuple
from core.types import PosePrediction


class MultiViewTrajectoryEngine:
    """Generates N-view camera orbits and renders mathematically consistent 3D target depth maps."""

    def generate_circular_orbit(
        self,
        num_frames: int = 12,
        radius: float = 2.0,
        height: float = 0.0,
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> List[PosePrediction]:
        """Generates N-view camera orbit returning valid World-to-Camera Extrinsics [R_w2c | t_w2c]."""
        poses = []
        angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
        target_pt = np.array(target, dtype=np.float32)

        for i, theta in enumerate(angles):
            cam_pos = np.array([radius * np.cos(theta), radius * np.sin(theta), height], dtype=np.float32)

            z_cam = target_pt - cam_pos
            z_cam /= np.linalg.norm(z_cam) + 1e-6

            up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            x_cam = np.cross(up_world, z_cam)
            if np.linalg.norm(x_cam) < 1e-3:
                x_cam = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                x_cam /= np.linalg.norm(x_cam)

            y_cam = np.cross(z_cam, x_cam)

            R_c2w = np.stack([x_cam, y_cam, z_cam], axis=-1)
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ cam_pos.reshape(3, 1)

            pose = PosePrediction(
                R=R_w2c.astype(np.float32),
                t=t_w2c.astype(np.float32),
                confidence=0.95,
                inlier_ratio=0.92,
                reprojection_error=0.20,
                metadata={"frame_index": i, "camera_position": cam_pos}
            )
            poses.append(pose)

        return poses

    def render_sphere_metric_depth(
        self,
        pose: PosePrediction,
        K: np.ndarray,
        img_shape: Tuple[int, int] = (480, 640),
        sphere_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        sphere_radius: float = 0.6
    ) -> np.ndarray:
        """Renders mathematically exact metric depth map for a 3D sphere in camera frame."""
        h, w = img_shape
        f_x, f_y = K[0, 0], K[1, 1]
        c_x, c_y = K[0, 2], K[1, 2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x_c = (u - c_x) / f_x
        y_c = (v - c_y) / f_y
        ray_dir = np.stack([x_c, y_c, np.ones_like(x_c)], axis=-1)
        ray_dir /= np.linalg.norm(ray_dir, axis=-1, keepdims=True)

        # Transform sphere center to camera frame: C_cam = R_w2c * C_world + t_w2c
        R_w2c, t_w2c = pose.R, pose.t
        c_world = np.array(sphere_center, dtype=np.float32).reshape(3, 1)
        c_cam = (R_w2c @ c_world + t_w2c).flatten()

        # Ray-Sphere Intersection
        b = -2.0 * np.sum(ray_dir * c_cam, axis=-1)
        c_val = np.sum(c_cam**2) - sphere_radius**2
        discriminant = b**2 - 4.0 * c_val

        valid_mask = discriminant >= 0
        t_hit = np.zeros((h, w), dtype=np.float32)

        t_val = (-b[valid_mask] - np.sqrt(discriminant[valid_mask])) / 2.0
        t_hit[valid_mask] = t_val * ray_dir[..., 2][valid_mask]

        return t_hit
