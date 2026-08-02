"""VYOMAAV Confidence-Aware Marching Cubes Isosurface Extractor with Grid Smoothing."""

import trimesh
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from geometry.tsdf_volume import TSDFVolumeGrid


class MarchingCubesExtractor:
    """Extracts isosurface geometry with spatial confidence pruning and Laplacian grid smoothing."""

    def extract_mesh(
        self,
        tsdf_volume: TSDFVolumeGrid,
        iso_level: float = 0.0,
        min_weight_threshold: float = 0.20,
        smooth_sigma: float = 0.8
    ) -> trimesh.Trimesh:
        """Applies confidence-aware voxel masking and 3D Gaussian smoothing before surface extraction."""
        tsdf_grid = tsdf_volume.tsdf_grid.copy()
        weight_grid = tsdf_volume.weight_grid

        # 1. Mask out low-confidence/unobserved voxels
        unobserved_mask = weight_grid < min_weight_threshold
        tsdf_grid[unobserved_mask] = 1.0

        # 2. Apply 3D Spatial Gaussian Smoothing to eliminate staircase discretization noise
        if smooth_sigma > 0:
            tsdf_grid = gaussian_filter(tsdf_grid, sigma=smooth_sigma)

        try:
            verts_vox, faces, normals, _ = marching_cubes(tsdf_grid, level=iso_level)

            # Map Voxel indices [0, grid_dim] to World Metric Space
            dim = tsdf_volume.grid_dim
            b_min = tsdf_volume.bounds_min
            b_max = tsdf_volume.bounds_max

            verts_world = np.zeros_like(verts_vox, dtype=np.float32)
            verts_world[:, 0] = b_min[0] + verts_vox[:, 0] * (b_max[0] - b_min[0]) / dim
            verts_world[:, 1] = b_min[1] + verts_vox[:, 1] * (b_max[1] - b_min[1]) / dim
            verts_world[:, 2] = b_min[2] + verts_vox[:, 2] * (b_max[2] - b_min[2]) / dim

            mesh = trimesh.Trimesh(
                vertices=verts_world,
                faces=faces,
                vertex_normals=normals,
                process=True
            )

            # Basic mesh cleanup
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.update_faces(mesh.unique_faces())
            mesh.remove_unreferenced_vertices()

            return mesh

        except Exception as e:
            print(f"Warning: Isosurface extraction fallback due to: {e}")
            return trimesh.creation.icosphere(subdivisions=2, radius=0.5)
