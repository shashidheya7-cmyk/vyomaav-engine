"""VYOMAAV Comprehensive Quantitative Benchmark & Diagnostic Audit Engine."""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional


@dataclass
class ExtendedEvaluationReport:
    """Extended quantitative report covering geometry, pose, topology, and volumetric metrics."""
    abs_rel_error: float
    rmse: float
    scale_error: float
    chamfer_distance: float
    hausdorff_distance: float
    f_score: float
    normal_consistency: float
    volumetric_iou: float
    pred_occupied_voxels: int
    gt_occupied_voxels: int
    intersection_voxels: int
    union_voxels: int
    rotation_error_deg: float
    translation_error_m: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkEvaluationEngine:
    """Calculates research-grade reconstruction metrics against ground-truth benchmarks."""

    def evaluate_depth(self, pred_depth: np.ndarray, gt_depth: np.ndarray) -> Tuple[float, float, float]:
        """Calculates AbsRel, RMSE, and Scale Error."""
        valid_mask = (gt_depth > 0.05) & (pred_depth > 0.05)
        p, g = pred_depth[valid_mask], gt_depth[valid_mask]

        if len(g) == 0:
            return 0.0, 0.0, 0.0

        abs_rel = float(np.mean(np.abs(p - g) / g))
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))
        scale_error = float(np.mean(p) / np.mean(g))

        return abs_rel, rmse, scale_error

    def evaluate_pose_error(self, R_pred: np.ndarray, t_pred: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray) -> Tuple[float, float]:
        """Calculates Geodesic Rotation Error (degrees) and Translation Error (meters)."""
        R_diff = R_pred.T @ R_gt
        trace_val = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
        rot_err_deg = float(np.degrees(np.arccos(trace_val)))
        trans_err_m = float(np.linalg.norm(t_pred.flatten() - t_gt.flatten()))
        return rot_err_deg, trans_err_m

    def evaluate_occupancy_iou(
        self,
        pred_mesh: trimesh.Trimesh,
        gt_mesh: trimesh.Trimesh,
        grid_res: int = 64,
        distance_threshold: float = 0.05
    ) -> Tuple[float, int, int, int, int]:
        """Calculates 3D Voxelized Surface/Occupancy IoU over a unified 3D evaluation grid."""
        if len(pred_mesh.vertices) < 10 or len(gt_mesh.vertices) < 10:
            return 0.0, 0, 0, 0, 0

        # Compute unified bounding envelope
        min_bound = np.minimum(pred_mesh.bounds[0], gt_mesh.bounds[0]) - 0.1
        max_bound = np.maximum(pred_mesh.bounds[1], gt_mesh.bounds[1]) + 0.1

        # Build 3D spatial evaluation grid
        x = np.linspace(min_bound[0], max_bound[0], grid_res)
        y = np.linspace(min_bound[1], max_bound[1], grid_res)
        z = np.linspace(min_bound[2], max_bound[2], grid_res)
        grid_pts = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)

        # Sample surface point clouds
        p_samples = pred_mesh.sample(30000)
        g_samples = gt_mesh.sample(30000)

        # Query KD-Tree distance to grid points
        tree_p = cKDTree(p_samples)
        tree_g = cKDTree(g_samples)

        dist_p, _ = tree_p.query(grid_pts)
        dist_g, _ = tree_g.query(grid_pts)

        occ_p = dist_p <= distance_threshold
        occ_g = dist_g <= distance_threshold

        intersection = int(np.sum(occ_p & occ_g))
        union = int(np.sum(occ_p | occ_g))
        pred_occ = int(np.sum(occ_p))
        gt_occ = int(np.sum(occ_g))

        iou = float(intersection / union) if union > 0 else 0.0

        return iou, pred_occ, gt_occ, intersection, union

    def evaluate_mesh_geometry(
        self,
        pred_mesh: trimesh.Trimesh,
        gt_mesh: trimesh.Trimesh,
        threshold_meters: float = 0.05,
        num_sample_points: int = 20000
    ) -> Tuple[float, float, float, float, float, int, int, int, int]:
        """Calculates Chamfer, Hausdorff, F-Score@5cm, Normal Consistency, and Voxelized Occupancy IoU."""
        if len(pred_mesh.vertices) < 10 or len(gt_mesh.vertices) < 10:
            return 1.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0

        p_points, p_face_idx = pred_mesh.sample(num_sample_points, return_index=True)
        g_points, g_face_idx = gt_mesh.sample(num_sample_points, return_index=True)

        p_normals = pred_mesh.face_normals[p_face_idx]
        g_normals = gt_mesh.face_normals[g_face_idx]

        kdtree_pred = cKDTree(p_points)
        kdtree_gt = cKDTree(g_points)

        dist_p2g, idx_p2g = kdtree_gt.query(p_points)
        dist_g2p, _ = kdtree_pred.query(g_points)

        chamfer = float(np.mean(dist_p2g) + np.mean(dist_g2p)) / 2.0
        hausdorff = float(max(np.max(dist_p2g), np.max(dist_g2p)))

        precision = np.mean(dist_p2g < threshold_meters)
        recall = np.mean(dist_g2p < threshold_meters)
        f_score = float(2 * (precision * recall) / (precision + recall + 1e-8))

        matched_gt_normals = g_normals[idx_p2g]
        cos_sim = np.abs(np.sum(p_normals * matched_gt_normals, axis=-1))
        normal_consistency = float(np.mean(cos_sim))

        iou, pred_occ, gt_occ, inter_vox, union_vox = self.evaluate_occupancy_iou(
            pred_mesh, gt_mesh, grid_res=64, distance_threshold=threshold_meters
        )

        return chamfer, hausdorff, f_score, normal_consistency, iou, pred_occ, gt_occ, inter_vox, union_vox

    def run_full_benchmark(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        pred_mesh: trimesh.Trimesh,
        gt_mesh: trimesh.Trimesh,
        R_pred: Optional[np.ndarray] = None,
        t_pred: Optional[np.ndarray] = None,
        R_gt: Optional[np.ndarray] = None,
        t_gt: Optional[np.ndarray] = None
    ) -> ExtendedEvaluationReport:
        """Executes full diagnostic suite."""
        abs_rel, rmse, scale_err = self.evaluate_depth(pred_depth, gt_depth)
        chamfer, hausdorff, f_score, norm_cons, iou, pred_occ, gt_occ, inter_vox, union_vox = self.evaluate_mesh_geometry(pred_mesh, gt_mesh)

        rot_err, trans_err = 0.0, 0.0
        if R_pred is not None and R_gt is not None:
            rot_err, trans_err = self.evaluate_pose_error(R_pred, t_pred, R_gt, t_gt)

        return ExtendedEvaluationReport(
            abs_rel_error=abs_rel,
            rmse=rmse,
            scale_error=scale_err,
            chamfer_distance=chamfer,
            hausdorff_distance=hausdorff,
            f_score=f_score,
            normal_consistency=norm_cons,
            volumetric_iou=iou,
            pred_occupied_voxels=pred_occ,
            gt_occupied_voxels=gt_occ,
            intersection_voxels=inter_vox,
            union_voxels=union_vox,
            rotation_error_deg=rot_err,
            translation_error_m=trans_err,
            metadata={"protocol": "Voxelized_3D_Occupancy_ScanNet++_Protocol"}
        )
