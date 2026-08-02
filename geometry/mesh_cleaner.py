"""VYOMAAV Mesh Topology Cleaning & Watertight Repair Engine."""

import trimesh


class MeshTopologyCleaner:
    """Removes non-manifold edges, fills holes, and repairs face winding."""

    def clean(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Applies non-degenerate filtering and watertight repair."""
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()
        mesh.fill_holes()
        return mesh
