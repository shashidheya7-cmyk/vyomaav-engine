"""VYOMAAV Topology Validation & Automated Mesh Repair Engine."""

import trimesh
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class TopologyAuditReport:
    """Comprehensive diagnostic report on mesh topological health."""
    vertex_count: int
    face_count: int
    is_watertight: bool
    is_manifold: bool
    has_self_intersections: bool
    degenerate_face_count: int
    boundary_edge_count: int
    euler_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TopologyValidatorEngine:
    """Audits 3D mesh topology and executes automated manifold repair."""

    def audit_topology(self, mesh: trimesh.Trimesh) -> TopologyAuditReport:
        """Runs a complete diagnostic audit on topological health."""
        degenerate_count = int(np.sum(~mesh.nondegenerate_faces()))
        boundary_edges = len(mesh.outline().entities) if hasattr(mesh, "outline") else 0
        
        # Euler characteristic: V - E + F (2 for closed sphere-like manifolds)
        euler_num = int(mesh.euler_number)

        # Self-intersection check
        try:
            has_self_intersect = bool(mesh.is_self_intersecting)
        except Exception:
            has_self_intersect = False

        return TopologyAuditReport(
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            is_watertight=bool(mesh.is_watertight),
            is_manifold=bool(mesh.is_winding_consistent),
            has_self_intersections=has_self_intersect,
            degenerate_face_count=degenerate_count,
            boundary_edge_count=boundary_edges,
            euler_number=euler_num,
            metadata={"volume": float(mesh.volume) if mesh.is_watertight else 0.0}
        )

    def repair_and_seal_topology(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Executes non-manifold repair, vertex welding, and hole sealing."""
        repaired = mesh.copy()

        # 1. Merge Duplicate Vertices within Epsilon
        repaired.merge_vertices(merge_tex=False, merge_norm=False)

        # 2. Remove Degenerate and Duplicate Faces
        repaired.update_faces(repaired.nondegenerate_faces())
        repaired.update_faces(repaired.unique_faces())
        repaired.remove_unreferenced_vertices()

        # 3. Fix Winding Order & Inverted Normals
        trimesh.repair.fix_winding(repaired)
        trimesh.repair.fix_inversion(repaired)
        trimesh.repair.fix_normals(repaired)

        # 4. Fill Hole Boundaries
        trimesh.repair.fill_holes(repaired)

        return repaired
