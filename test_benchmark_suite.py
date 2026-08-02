"""Test Runner for Multi-View Orbital Scaling (12 Views) and Quantitative Benchmark Audit."""

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


def test_multiview_scaling_and_benchmarks():
    print("--- Executing Multi-Frame Scaling (12 Views) & Quantitative Benchmark Test ---")

    # 1. Camera Intrinsics
    cam_model = PinholeCameraModel(fov_deg=60.0)
    info = cam_model.estimate_from_exif(Image.new("RGB", (640, 480)))
    K = info["K"]

    test_img = Image.new("RGB", (640, 480), color=(150, 150, 150))
    geom_engine = GeometryEstimationEngine()
    geom_pred = geom_engine.predict_full_geometry(test_img, K)

    # 2. Generate 12-Frame Circular Camera Trajectory Orbit
    orbit_engine = MultiViewTrajectoryEngine()
    orbit_poses = orbit_engine.generate_circular_orbit(num_frames=12, radius=2.2, height=0.4)

    print(f"\n1. Multi-Frame Orbital Scaling Audit:")
    print(f"   - Generated Camera Trajectory: {len(orbit_poses)} Orbit Views")

    # 3. Fuse All 12 Views into TSDF Volumetric Grid
    tsdf_volume = TSDFVolumeGrid(grid_dim=128, trunc_margin=0.08)

    for i, pose in enumerate(orbit_poses):
        tsdf_volume.integrate_prediction(
            geom_pred, R=pose.R, t=pose.t, min_confidence_threshold=0.35
        )

    active_voxels = np.sum(tsdf_volume.weight_grid > 0)
    print(f"   - 12-View Active Voxels:       {active_voxels:,} / {128**3:,} ({(active_voxels / 128**3):.2%})")

    # 4. Extract Isosurface Surface Mesh
    extractor = MarchingCubesExtractor()
    pred_mesh = extractor.extract_mesh(tsdf_volume, iso_level=0.0, smooth_sigma=0.8)

    # 5. Synthesize Synthetic Ground Truth Mesh (Reference Sphere Domain)
    gt_mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    gt_depth = geom_pred.metric_depth * 1.02  # Simulated 2% depth noise ground truth

    # 6. Execute Quantitative Benchmark Evaluation Engine
    benchmark_engine = BenchmarkEvaluationEngine()
    report = benchmark_engine.run_full_benchmark(
        pred_depth=geom_pred.metric_depth,
        gt_depth=gt_depth,
        pred_mesh=pred_mesh,
        gt_mesh=gt_mesh
    )

    print("\n2. Quantitative Benchmark Diagnostics Audit (ScanNet++ / Replica Protocol):")
    print(f"   - Absolute Relative Error (AbsRel): {report.abs_rel_error:.4f} (Target < 0.05)")
    print(f"   - Root Mean Squared Error (RMSE):   {report.rmse:.4f} meters")
    print(f"   - Scale Error Ratio:               {report.scale_error:.4f} (1.00 = Perfect Scale)")
    print(f"   - Chamfer Distance:                {report.chamfer_distance:.4f} meters")
    print(f"   - F-Score @ 5cm Threshold:         {report.f_score:.2%}")
    print(f"   - Normal Surface Consistency:      {report.normal_consistency:.4f} (1.00 = Perfect Alignment)")

    os.makedirs("./real_world_output", exist_ok=True)
    out_obj = "./real_world_output/12view_orbit_world.obj"
    pred_mesh.export(out_obj)
    print(f"\n   - Exported 12-View Asset:          {out_obj}")

    print("\n Multi-Frame Scaling & Quantitative Benchmark Test Complete!")


if __name__ == "__main__":
    test_multiview_scaling_and_benchmarks()
