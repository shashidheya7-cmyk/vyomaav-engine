"""VYOMAAV Industrial xatlas UV Chart Parameterization Engine."""

import trimesh
import numpy as np
import xatlas
from typing import Dict, Any, Optional


class UVUnwrapperEngine:
    """Generates non-overlapping UV texture charts using xatlas C++ parameterization."""

    def unwrap_uv_coordinates(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Paramterizes mesh into non-overlapping UV texture charts."""
        # 1. Execute xatlas Chart Parametrization
        vmapping, indices, uvs = xatlas.parametrize(
            mesh.vertices.astype(np.float32), 
            mesh.faces.astype(np.int32)
        )

        # 2. Reconstruct mesh with xatlas-mapped vertices and faces
        mesh_uv = trimesh.Trimesh(
            vertices=mesh.vertices[vmapping],
            faces=indices,
            process=False
        )

        # 3. Assign UV Texture Visuals
        mesh_uv.visual = trimesh.visual.TextureVisuals(uv=uvs.astype(np.float32))
        return mesh_uv
