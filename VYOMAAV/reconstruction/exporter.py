"""VYOMAAV Reconstruction Exporter: Exports Mesh Geometry to Wavefront OBJ / PLY."""

import os
from typing import Dict, Any

class MeshExporter:
    """Exports reconstructed 3D mesh arrays to OBJ and PLY formats."""

    @staticmethod
    def export_to_obj(mesh_data: Dict[str, Any], file_path: str) -> str:
        """Writes vertices, normals, UVs, and face indices to a standard .obj file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        vertices = mesh_data.get("vertices", [])
        normals = mesh_data.get("normals", [])
        uvs = mesh_data.get("uvs", [])
        faces = mesh_data.get("faces", [])

        with open(file_path, "w") as f:
            f.write("# VYOMAAV Wavefront OBJ Exporter\n")
            f.write(f"# Backend: {mesh_data.get('backend', 'TRELLIS')}\n\n")

            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            for vn in normals:
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")

            for vt in uvs:
                f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")

            for face in faces:
                f1, f2, f3 = face[0] + 1, face[1] + 1, face[2] + 1
                f.write(f"f {f1}/{f1}/{f1} {f2}/{f2}/{f2} {f3}/{f3}/{f3}\n")

        return file_path
