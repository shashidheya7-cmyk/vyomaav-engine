"""Test Runner for 12-View Pipeline with Enriched Metadata Logging & Performance Profiling."""

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
from neural_completion.trellis import TrellisCompletionAdapter
from benchmarks.metrics import BenchmarkEvaluationEngine
from benchmarks.profiler import PipelineProfiler
from benchmarks.logger import BenchmarkLogger


def test_multiview_scaling_and_benchmarks():
    print("--- Executing 12-View Pipeline with Enriched Metadata Logging ---")

    profiler = PipelineProfiler()
    logger = BenchmarkLogger(output_dir="./benchmarks")

    # 1. Camera Orbit Generation
    with profiler.profile_stage("1_Camera_Trajectory_Generation"):
        cam_model = PinholeCameraModel(fov_deg=60.0)
        info = cam_model.estimate_from_exif(Image.new("RGB", (640, 480)))
        K = info["K"]

        orbit_engine = MultiViewTrajectoryEngine()
        orbit_poses = orbit_engine.generate_circular_orbit(num_frames=12, radius=2.0, height=0.0)

    # 2. TSDF Volumetric Grid Integration
    tsdf_volume = TSDFVolumeGrid(grid_dim=128, trunc_margin=0.08, bounds_min=(-1.2, -1.2, -1.2), bounds_max=(1.2, 1.2, 1.2))
    geom_engine = GeometryEstimationEngine()

    with profiler.profile_stage("2_Angle_Weighted_TSDF_Integration"):
        for pose in orbit_poses:
            view_depth = orbit_engine.render_sphere_metric_depth(pose, K, sphere_radius=0.6)
            test_img = Image.new("RGB", (640, 480), color=(150, 150, 150))
            
            geom_pred = geom_engine.predict_full_geometry(test_img, K)
            geom_pred.metric_depth = view_depth
            tsdf_volume.integrate_prediction(geom_pred, R=pose.R, t=pose.t, min_confidence_threshold=0.35)

    health = tsdf_volume.audit_grid_health()
    print("\n1. TSDF Volumetric Grid Health Audit:")
    print(f"   - Observed Voxels:            {health['observed_voxels']:,} / {health['total_voxels']:,} ({health['observation_ratio']:.2%})")
    print(f"   - Active Surface Voxels:     {health['surface_voxels']:,}")

    # 3. Isosurface Extraction
    with profiler.profile_stage("3_Marching_Cubes_Extraction"):
        extractor = MarchingCubesExtractor()
        partial_mesh = extractor.extract_mesh(tsdf_volume, iso_level=0.0, smooth_sigma=0.8)

    # 4. Neural Completion Execution
    with profiler.profile_stage("4_TRELLIS_Neural_Completion"):
        trellis_adapter = TrellisCompletionAdapter(device="cuda")
        completion_res = trellis_adapter.complete_mesh(partial_mesh, geom_pred)
        fused_mesh = completion_res["fused_mesh"]

    # 5. Quantitative Benchmark Evaluation
    with profiler.profile_stage("5_Quantitative_Benchmark_Evaluation"):
        gt_mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.6)
        ref_pose = orbit_poses[0]
        pred_depth = orbit_engine.render_sphere_metric_depth(ref_pose, K, sphere_radius=0.6)
        
        noise = np.random.normal(0, 0.025 * pred_depth, pred_depth.shape).astype(np.float32)
        gt_depth = np.where(pred_depth > 0, pred_depth + noise, 0.0)

        benchmark_engine = BenchmarkEvaluationEngine()
        report = benchmark_engine.run_full_benchmark(
            pred_depth=pred_depth,
            gt_depth=gt_depth,
            pred_mesh=fused_mesh,
            gt_mesh=gt_mesh,
            R_pred=orbit_poses[0].R,
            t_pred=orbit_poses[0].t,
            R_gt=orbit_poses[0].R,
            t_gt=orbit_poses[0].t
        )

    print("\n2. Quantitative Benchmark Diagnostics Audit:")
    print(f"   - Absolute Relative Error:    {report.abs_rel_error:.4f}")
    print(f"   - Root Mean Squared Error:   {report.rmse:.4f} meters")
    print(f"   - Chamfer Distance:           {report.chamfer_distance:.4f} meters")
    print(f"   - Hausdorff Maximum Error:   {report.hausdorff_distance:.4f} meters")
    print(f"   - F-Score @ 5cm Threshold:    {report.f_score:.2%}")
    print(f"   - Normal Surface Consistency: {report.normal_consistency:.4f}")
    print(f"   - Volumetric Occupancy IoU:   {report.volumetric_iou:.2%}")

    # 6. Pipeline Latency Breakdown
    print("\n3. Pipeline Stage Execution Profiling Breakdown:")
    perf_report = profiler.report()
    print(f"   - Total Execution Time:       {perf_report['total_latency_sec']:.4f} seconds")
    print(f"   - CPU Memory RSS:             {perf_report['cpu_ram_rss_mb']:.2f} MB")
    for stage, metrics in perf_report["stages"].items():
        print(f"     * {stage:<35}: {metrics['duration_sec']:.4f}s ({metrics['percentage']:>5.1f}%) [VRAM: {metrics['peak_vram_mb']:.1f} MB]")

    # 7. Log Enriched Benchmark Metadata
    metrics_dict = {
        "abs_rel_error": report.abs_rel_error,
        "rmse": report.rmse,
        "chamfer_distance": report.chamfer_distance,
        "hausdorff_distance": report.hausdorff_distance,
        "f_score": report.f_score,
        "normal_consistency": report.normal_consistency,
        "volumetric_iou": report.volumetric_iou
    }
    
    run_config = {
        "num_views": 12,
        "resolution": "640x480",
        "depth_model": "DepthAnythingV2",
        "completion_backend": completion_res["backend"]
    }
    
    logger.log_run("12_view_orbit_enriched_benchmark", metrics_dict, perf_report, config=run_config)

    print("\n Enriched Benchmark Test Complete!")


if __name__ == "__main__":
    test_multiview_scaling_and_benchmarks()
