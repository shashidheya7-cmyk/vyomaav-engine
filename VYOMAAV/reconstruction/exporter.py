"""VYOMAAV Asset Exporter: Writes reconstructed 3D meshes (.obj), Gaussian splats (.ply), and scene graphs."""

import os
import json
import numpy as np
from typing import Dict, Any, Optional

class MeshExporter:
    """Exports fused geometry to standard Wavefront .OBJ and 3D Gaussian .PLY files."""

    @staticmethod
    def export_to_obj(vertices: list, faces: list, output_filepath: str) -> str:
        """Writes vertex coordinates and triangle faces to a Wavefront .OBJ file."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        with open(output_filepath, "w") as f:
            f.write("# VYOMAAV Engine Reconstructed 3D Mesh\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                # 1-based indexing for OBJ format
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        return output_filepath

    @staticmethod
    def export_gaussians_to_ply(positions: list, output_filepath: str) -> str:
        """Writes 3D Gaussian splat point positions to a PLY file for point-cloud/splat viewers."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        pts = np.array(positions, dtype=np.float32)

        with open(output_filepath, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for pt in pts:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

        return output_filepath
