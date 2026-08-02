"""VYOMAAV Confidence, Distance, and Grazing-Angle Weighted TSDF Volumetric Grid."""

import trimesh
import numpy as np
from typing import Dict, Any, Tuple, Optional
from core.types import FullGeometryPrediction


class TSDFVolumeGrid:
    """Multi-observation TSDF volume grid with confidence, grazing-angle, and distance-weighted fusion."""

    def __init__(
        self,
        grid_dim: int = 128,
        trunc_margin: float = 0.08,
        bounds_min: Tuple[float, float, float] = (-1.2, -1.2, -1.2),
        bounds_max: Tuple[float, float, float] = (1.2, 1.2, 1.2)
    ):
        self.grid_dim = grid_dim
        self.trunc_margin = trunc_margin
        self.bounds_min = np.array(bounds_min, dtype=np.float32)
        self.bounds_max = np.array(bounds_max, dtype=np.float32)

        self.tsdf_grid = np.ones((grid_dim, grid_dim, grid_dim), dtype=np.float32)
        self.weight_grid = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.float32)
        self.obs_count_grid = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.int32)
        self.variance_grid = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.float32)

        x = np.linspace(self.bounds_min[0], self.bounds_max[0], grid_dim)
        y = np.linspace(self.bounds_min[1], self.bounds_max[1], grid_dim)
        z = np.linspace(self.bounds_min[2], self.bounds_max[2], grid_dim)
        
        xs, ys, zs = np.meshgrid(x, y, z, indexing="ij")
        self.voxel_centers = np.stack([xs.flatten(), ys.flatten(), zs.flatten()], axis=-1)

    def integrate_prediction(
        self,
        geom_pred: FullGeometryPrediction,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        min_confidence_threshold: float = 0.35
    ) -> None:
        """Accumulates depth observations using angle, distance, and confidence weighting."""
        R_w2c = R if R is not None else np.eye(3, dtype=np.float32)
        t_w2c = t if t is not None else np.zeros((3, 1), dtype=np.float32)

        K = geom_pred.camera_K
        depth_map = geom_pred.metric_depth
        confidence_map = geom_pred.geometry_confidence
        normals_map = geom_pred.normals
        h, w = depth_map.shape

        f_x, f_y = K[0, 0], K[1, 1]
        c_x, c_y = K[0, 2], K[1, 2]

        # 1. Transform World Voxel Centers to Camera Frame: P_cam = R_w2c * P_world + t_w2c
        vox_cam = (R_w2c @ self.voxel_centers.T + t_w2c).T
        p_x, p_y, p_z = vox_cam[:, 0], vox_cam[:, 1], vox_cam[:, 2]
        valid_z = p_z > 0.05

        # 2. Camera Frame to Pixel Space (u, v)
        u_proj = np.round((p_x * f_x / (p_z + 1e-6)) + c_x).astype(np.int32)
        v_proj = np.round((p_y * f_y / (p_z + 1e-6)) + c_y).astype(np.int32)

        valid_uv = (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h) & valid_z
        valid_indices = np.where(valid_uv)[0]
        
        u_valid, v_valid = u_proj[valid_indices], v_proj[valid_indices]
        obs_depth = depth_map[v_valid, u_valid]
        obs_confidence = confidence_map[v_valid, u_valid]

        # Filter Low-Confidence Projections
        conf_mask = (obs_confidence >= min_confidence_threshold) & (obs_depth > 0.05)
        final_valid_indices = valid_indices[conf_mask]
        
        obs_depth = obs_depth[conf_mask]
        obs_confidence = obs_confidence[conf_mask]
        u_final, v_final = u_valid[conf_mask], v_valid[conf_mask]
        p_z_valid = p_z[final_valid_indices]

        # 3. Compute Viewing Angle Weighting (cos_theta = ray_dir . surface_normal)
        ray_dirs = vox_cam[final_valid_indices]
        ray_dirs /= np.linalg.norm(ray_dirs, axis=-1, keepdims=True) + 1e-6
        
        obs_normals = normals_map[v_final, u_final]
        cos_theta = np.abs(np.sum(-ray_dirs * obs_normals, axis=-1))
        cos_theta = np.clip(cos_theta, 0.1, 1.0)

        # 4. Distance Attenuation Weighting (1 / z^2 normalized to depth range)
        dist_weight = np.clip((1.0 / (obs_depth + 1e-3))**2, 0.1, 5.0)

        # 5. Composite Physical Weight: W = C_geom * cos_theta * W_dist
        composite_weights = obs_confidence * cos_theta * dist_weight

        # 6. Signed Distance Function (SDF) calculation
        dist = obs_depth - p_z_valid
        valid_sdf = (dist >= -self.trunc_margin) & (dist <= self.trunc_margin)

        tsdf_vals = np.clip(dist[valid_sdf] / self.trunc_margin, -1.0, 1.0)
        weights = composite_weights[valid_sdf]
        update_indices = final_valid_indices[valid_sdf]

        # 7. Volumetric Update
        tsdf_flat = self.tsdf_grid.flatten()
        weight_flat = self.weight_grid.flatten()
        obs_flat = self.obs_count_grid.flatten()
        var_flat = self.variance_grid.flatten()

        old_tsdf = tsdf_flat[update_indices]
        old_weight = weight_flat[update_indices]

        new_weight = old_weight + weights
        new_tsdf = (old_tsdf * old_weight + tsdf_vals * weights) / (new_weight + 1e-6)

        delta = tsdf_vals - new_tsdf
        new_var = var_flat[update_indices] + delta**2 * weights

        tsdf_flat[update_indices] = new_tsdf
        weight_flat[update_indices] = new_weight
        obs_flat[update_indices] += 1
        var_flat[update_indices] = new_var

        dim = self.grid_dim
        self.tsdf_grid = tsdf_flat.reshape((dim, dim, dim))
        self.weight_grid = weight_flat.reshape((dim, dim, dim))
        self.obs_count_grid = obs_flat.reshape((dim, dim, dim))
        self.variance_grid = var_flat.reshape((dim, dim, dim))

    def audit_grid_health(self) -> Dict[str, Any]:
        """Returns diagnostic metadata on TSDF volume grid health."""
        observed_mask = self.weight_grid > 0
        surface_mask = (np.abs(self.tsdf_grid) < 0.95) & observed_mask

        return {
            "min_tsdf": float(np.min(self.tsdf_grid)),
            "max_tsdf": float(np.max(self.tsdf_grid)),
            "mean_tsdf": float(np.mean(self.tsdf_grid[observed_mask])) if np.any(observed_mask) else 1.0,
            "total_voxels": int(self.grid_dim**3),
            "observed_voxels": int(np.sum(observed_mask)),
            "surface_voxels": int(np.sum(surface_mask)),
            "observation_ratio": float(np.sum(observed_mask) / (self.grid_dim**3))
        }

    def export_point_cloud(self, filepath: str) -> bool:
        """Exports near-surface integrated voxels as a 3D PLY point cloud."""
        surface_mask = (np.abs(self.tsdf_grid.flatten()) < 0.95) & (self.weight_grid.flatten() > 0)
        pts = self.voxel_centers[surface_mask]

        if len(pts) == 0:
            return False

        cloud = trimesh.PointCloud(pts)
        cloud.export(filepath)
        return True
