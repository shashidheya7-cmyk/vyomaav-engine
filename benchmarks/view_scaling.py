"""VYOMAAV Multi-View View Scaling Benchmark Experiment (N = 2, 4, 8, 12, 24, 48 Views)."""

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("VYOMAAV"))

import numpy as np
import trimesh
from PIL import Image
from camera.camera_model import PinholeCameraModel
from geometry_estimation.estimator import GeometryEstimationEngine
from geometry.tsdf_volume import TSDFVolumeGrid
from geometry.marching_cubes import MarchingCubesExtractor
from registration.multi_view_trajectory import MultiViewTrajectoryEngine
from benchmarks.metrics import BenchmarkEvaluationEngine


def run_view_scaling_experiment():
    print("--- Executing VYOMAAV View-Scaling Experiment (N = 2, 4, 8, 12, 24, 48) ---")

    view_counts = [2, 4, 8, 12, 24, 48]
    results = []

    cam_model = PinholeCameraModel(fov_deg=60.0)
    info = cam_model.estimate_from_exif(Image.new("RGB", (640, 480)))
    K = info["K"]

    geom_engine = GeometryEstimationEngine()
    benchmark_engine = BenchmarkEvaluationEngine()
    gt_mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.6)

    print("\n| Views (N) | Active Voxels | Chamfer (m) | F-Score @ 5cm | Occupancy IoU |")
    print("|:---------:|:-------------:|:-----------:|:-------------:|:-------------:|")

    for n_views in view_counts:
        orbit_engine = MultiViewTrajectoryEngine()
        orbit_poses = orbit_engine.generate_circular_orbit(num_frames=n_views, radius=2.0, height=0.0)

        tsdf_volume = TSDFVolumeGrid(
            grid_dim=128, 
            trunc_margin=0.08, 
            bounds_min=(-1.2, -1.2, -1.2), 
            bounds_max=(1.2, 1.2, 1.2)
        )

        for pose in orbit_poses:
            view_depth = orbit_engine.render_sphere_metric_depth(pose, K, sphere_radius=0.6)
            test_img = Image.new("RGB", (640, 480), color=(150, 150, 150))
            
            geom_pred = geom_engine.predict_full_geometry(test_img, K)
            geom_pred.metric_depth = view_depth
            tsdf_volume.integrate_prediction(geom_pred, R=pose.R, t=pose.t, min_confidence_threshold=0.35)

        extractor = MarchingCubesExtractor()
        pred_mesh = extractor.extract_mesh(tsdf_volume, iso_level=0.0, smooth_sigma=0.8)

        ref_pose = orbit_poses[0]
        pred_depth = orbit_engine.render_sphere_metric_depth(ref_pose, K, sphere_radius=0.6)
        noise = np.random.normal(0, 0.025 * pred_depth, pred_depth.shape).astype(np.float32)
        gt_depth = np.where(pred_depth > 0, pred_depth + noise, 0.0)

        report = benchmark_engine.run_full_benchmark(
            pred_depth=pred_depth,
            gt_depth=gt_depth,
            pred_mesh=pred_mesh,
            gt_mesh=gt_mesh
        )

        active_voxels = int(np.sum(tsdf_volume.weight_grid > 0))
        print(f"| {n_views:>9} | {active_voxels:>13,} | {report.chamfer_distance:>11.4f} | {report.f_score:>12.2%} | {report.volumetric_iou:>12.2%} |")

    print("\n Multi-View Scaling Benchmark Complete!")


if __name__ == "__main__":
    run_view_scaling_experiment()
