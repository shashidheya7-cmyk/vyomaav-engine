"""VYOMAAV Quantitative Benchmark & Evaluation Engine."""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional


@dataclass
class EvaluationReport:
    """Quantitative metric report comparing predictions against ground truth."""
    # Depth Metrics
    abs_rel_error: float            # Absolute Relative Error: mean(|z - z*| / z*)
    rmse: float                     # Root Mean Squared Error (meters)
    scale_error: float              # Scale drift ratio
    
    # 3D Mesh & Pointcloud Metrics
    chamfer_distance: float         # Bi-directional nearest neighbor distance (meters)
    f_score: float                  # Percentage of points within threshold tau (e.g. 5cm)
    normal_consistency: float       # Mean cosine similarity between surface normals [0, 1]
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkEvaluationEngine:
    """Calculates quantitative reconstruction metrics against ground-truth benchmarks."""

    def evaluate_depth(self, pred_depth: np.ndarray, gt_depth: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """Calculates AbsRel, RMSE, and Scale Error between predicted and GT depth maps."""
        if valid_mask is None:
            valid_mask = (gt_depth > 0.01) & (pred_depth > 0.01)

        p = pred_depth[valid_mask]
        g = gt_depth[valid_mask]

        if len(g) == 0:
            return 0.0, 0.0, 0.0

        # 1. Absolute Relative Error (AbsRel)
        abs_rel = float(np.mean(np.abs(p - g) / g))

        # 2. Root Mean Squared Error (RMSE)
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))

        # 3. Scale Error Ratio
        scale_error = float(np.mean(p) / np.mean(g))

        return abs_rel, rmse, scale_error

    def evaluate_mesh_geometry(
        self,
        pred_mesh: trimesh.Trimesh,
        gt_mesh: trimesh.Trimesh,
        threshold_meters: float = 0.05,
        num_sample_points: int = 50000
    ) -> Tuple[float, float, float]:
        """Calculates Chamfer Distance, F-Score@5cm, and Normal Consistency."""
        if len(pred_mesh.vertices) == 0 or len(gt_mesh.vertices) == 0:
            return 0.0, 0.0, 0.0

        # 1. Uniformly Sample 3D Points & Normals from Surface Meshes
        p_points, p_face_idx = pred_mesh.sample(num_sample_points, return_index=True)
        g_points, g_face_idx = gt_mesh.sample(num_sample_points, return_index=True)

        p_normals = pred_mesh.face_normals[p_face_idx]
        g_normals = gt_mesh.face_normals[g_face_idx]

        # 2. Build Spatial KD-Trees for Nearest Neighbor Searching
        kdtree_pred = cKDTree(p_points)
        kdtree_gt = cKDTree(g_points)

        # Distance from Pred -> GT
        dist_p2g, idx_p2g = kdtree_gt.query(p_points)
        # Distance from GT -> Pred
        dist_g2p, _ = kdtree_pred.query(g_points)

        # 3. Bi-directional Chamfer Distance
        chamfer = float(np.mean(dist_p2g) + np.mean(dist_g2p)) / 2.0

        # 4. F-Score at threshold tau (e.g., 5cm = 0.05m)
        precision = np.mean(dist_p2g < threshold_meters)
        recall = np.mean(dist_g2p < threshold_meters)
        f_score = float(2 * (precision * recall) / (precision + recall + 1e-8))

        # 5. Normal Consistency (Cosine similarity of nearest neighbor normals)
        matched_gt_normals = g_normals[idx_p2g]
        cos_sim = np.abs(np.sum(p_normals * matched_gt_normals, axis=-1))
        normal_consistency = float(np.mean(cos_sim))

        return chamfer, f_score, normal_consistency

    def run_full_benchmark(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        pred_mesh: trimesh.Trimesh,
        gt_mesh: trimesh.Trimesh
    ) -> EvaluationReport:
        """Executes full evaluation suite and returns standardized diagnostic report."""
        abs_rel, rmse, scale_err = self.evaluate_depth(pred_depth, gt_depth)
        chamfer, f_score, norm_cons = self.evaluate_mesh_geometry(pred_mesh, gt_mesh)

        return EvaluationReport(
            abs_rel_error=abs_rel,
            rmse=rmse,
            scale_error=scale_err,
            chamfer_distance=chamfer,
            f_score=f_score,
            normal_consistency=norm_cons,
            metadata={
                "benchmark_protocol": "ScanNet++/Replica_Standard_5cm",
                "sample_points": 50000
            }
        )
