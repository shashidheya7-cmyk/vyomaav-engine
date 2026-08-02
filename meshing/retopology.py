"""VYOMAAV Native Quadric Error Metric (QEM) Retopology Engine."""

import trimesh
import numpy as np
from typing import Dict, Any, Optional
import fast_simplification


class MeshRetopologyEngine:
    """Decimates mesh polygon count using native C++ Fast-Quadric-Mesh-Simplification."""

    def decimate(self, mesh: trimesh.Trimesh, target_face_count: int = 3500) -> trimesh.Trimesh:
        """Applies true QEM simplification down to target_face_count while preserving feature edges."""
        if len(mesh.faces) <= target_face_count:
            return mesh

        target_reduction = 1.0 - (target_face_count / len(mesh.faces))
        target_reduction = max(0.05, min(0.95, target_reduction))

        # Native C++ Quadric Simplification Execution
        points_out, faces_out = fast_simplification.simplify(
            mesh.vertices.astype(np.float32), 
            mesh.faces.astype(np.int32), 
            target_reduction
        )

        decimated = trimesh.Trimesh(
            vertices=points_out,
            faces=faces_out,
            process=True
        )
        decimated.update_faces(decimated.nondegenerate_faces())
        decimated.remove_unreferenced_vertices()
        
        return decimated
