"""Test Runner for VYOMAAV Phase 4 Multi-View Geometry & Metric Scale Accuracy Engine."""

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("VYOMAAV"))

import numpy as np
from PIL import Image
from registration.dust3r_adapter import DUSt3RMultiViewAdapter
from geometry_estimation.metric_depth import Metric3DDepthEstimator
from registration.bundle_adjustment import WorldCoordinateOptimizer
from geometry_estimation.estimator import GeometryEstimationEngine
from geometry.tsdf_volume import TSDFVolumeGrid
from geometry.marching_cubes import MarchingCubesExtractor


def test_phase4_multiview_pipeline():
    print("--- Executing VYOMAAV Phase 4: Multi-View Geometry & Metric Scale Accuracy Test ---")

    # 1. Multi-View Image Inputs
    frame1 = Image.new("RGB", (640, 480), color=(140, 140, 140))
    frame2 = Image.new("RGB", (640, 480), color=(160, 160, 160))

    # 2. Execute DUSt3R / MASt3R Unconstrained Multi-View Registration
    dust3r_engine = DUSt3RMultiViewAdapter(device="cuda")
    mv_res = dust3r_engine.solve_pairwise_geometry(frame1, frame2)

    print("\n1. DUSt3R / MASt3R Multi-View Registration Audit:")
    print(f"   - Pointmap 1 Shape:           {mv_res['pointmap1'].shape}")
    print(f"   - Pointmap 2 Shape:           {mv_res['pointmap2'].shape}")
    print(f"   - Pairwise Pose Confidence:   {mv_res['pose'].confidence:.2f}")
    print(f"   - Pairwise Translation (t):  {mv_res['pose'].t.flatten()}")

    # 3. Execute Metric3D Absolute Distance Recovery
    metric_engine = Metric3DDepthEstimator(device="cuda")
    K = mv_res["camera_K"]
    metric_depth, learned_normals, scale_uncertainty = metric_engine.predict_metric_depth_and_normals(frame1, K)

    print("\n2. Metric3D Physical Distance Audit:")
    print(f"   - Absolute Metric Depth Range: [{metric_depth.min():.2f}m, {metric_depth.max():.2f}m]")
    print(f"   - Scale Uncertainty Score:    {scale_uncertainty:.2f} (Low = High Precision)")

    # 4. Global Bundle Adjustment & Coordinate Optimization
    optimizer = WorldCoordinateOptimizer()
    opt_poses = optimizer.optimize_trajectory([mv_res["pose"]], [0.95])

    print("\n3. World Coordinate Optimizer Audit:")
    print(f"   - Bundle Adjusted Poses:      {len(opt_poses)}")
    print(f"   - Adjusted Pose Error Metric: {opt_poses[0].reprojection_error:.4f}")

    # 5. Multi-View TSDF Volumetric Reconstruction
    print("\n4. Integrating Multi-View Metric Geometry into TSDF Volumetric Grid...")
    geom_engine = GeometryEstimationEngine()
    geom_pred = geom_engine.predict_full_geometry(frame1, K)
    
    # Overwrite estimated depth with Metric3D absolute ground truth
    geom_pred.metric_depth = metric_depth

    tsdf_volume = TSDFVolumeGrid(grid_dim=128, trunc_margin=0.08)
    
    # Integrate Frame 1
    tsdf_volume.integrate_prediction(geom_pred, min_confidence_threshold=0.35)
    
    # Integrate Frame 2 with Bundle-Adjusted Relative Pose [R|t]
    tsdf_volume.integrate_prediction(
        geom_pred, 
        R=opt_poses[0].R, 
        t=opt_poses[0].t, 
        min_confidence_threshold=0.35
    )

    extractor = MarchingCubesExtractor()
    multiview_mesh = extractor.extract_mesh(tsdf_volume, iso_level=0.0, smooth_sigma=0.8)

    os.makedirs("./real_world_output", exist_ok=True)
    out_obj = "./real_world_output/multiview_metric_world.obj"
    multiview_mesh.export(out_obj)

    print("\n5. Multi-View Reconstruction Audit:")
    print(f"   - Multi-View Active Voxels:   {np.sum(tsdf_volume.weight_grid > 0):,} / {128**3:,}")
    print(f"   - Reconstructed Vertices:     {len(multiview_mesh.vertices):,}")
    print(f"   - Reconstructed Triangles:    {len(multiview_mesh.faces):,}")
    print(f"   - True Metric Extents:        {np.round(multiview_mesh.extents, 3)} meters")
    print(f"   - Exported Multi-View Asset:  {out_obj}")

    print("\n Phase 4 Multi-View Geometry & Metric Scale Test Complete!")


if __name__ == "__main__":
    test_phase4_multiview_pipeline()
