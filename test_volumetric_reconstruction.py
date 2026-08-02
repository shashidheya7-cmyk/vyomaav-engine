"""Test Runner for VYOMAAV Phase 3B Volumetric Reconstruction Engine."""

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("VYOMAAV"))

import numpy as np
from PIL import Image
from camera.camera_model import PinholeCameraModel
from geometry_estimation.estimator import GeometryEstimationEngine
from geometry.tsdf_volume import TSDFVolumeGrid
from geometry.marching_cubes import MarchingCubesExtractor


def test_volumetric_reconstruction():
    print("--- Executing VYOMAAV Volumetric Reconstruction Engine (Phase 3B) ---")

    # 1. Initialize Camera Model & Geometry Engine
    cam_model = PinholeCameraModel(fov_deg=60.0)
    info = cam_model.estimate_from_exif(Image.new("RGB", (640, 480)))
    K = info["K"]

    test_img = Image.new("RGB", (640, 480), color=(160, 160, 160))

    geom_engine = GeometryEstimationEngine()
    geom_pred = geom_engine.predict_full_geometry(test_img, K)

    print("\n1. Geometry Prediction Contract Delivered:")
    print(f"   - Metric Depth Range: [{geom_pred.metric_depth.min():.2f}m, {geom_pred.metric_depth.max():.2f}m]")
    print(f"   - Mean Confidence:  {geom_pred.geometry_confidence.mean():.4f}")

    # 2. Allocate 128^3 TSDF Volume & Integrate Prediction
    print("\n2. Integrating Confidence-Weighted Geometry into TSDF Voxel Grid...")
    tsdf_volume = TSDFVolumeGrid(grid_dim=128, trunc_margin=0.08)
    tsdf_volume.integrate_prediction(geom_pred, min_confidence_threshold=0.35)

    observed_voxels = np.sum(tsdf_volume.weight_grid > 0)
    print(f"   - Active Integrated Voxels: {observed_voxels:,} / {128**3:,}")

    # 3. Extract Isosurface Surface via Marching Cubes
    print("\n3. Extracting Continuous 3D Isosurface Mesh via Marching Cubes...")
    extractor = MarchingCubesExtractor()
    mesh = extractor.extract_mesh(tsdf_volume, iso_level=0.0)

    # 4. Save and Verify 3D Mesh Output
    os.makedirs("./real_world_output", exist_ok=True)
    out_obj = "./real_world_output/volumetric_surface.obj"
    mesh.export(out_obj)

    print("\n4. Mesh Geometry Diagnostics Audit:")
    print(f"   - Reconstructed Vertices: {len(mesh.vertices):,}")
    print(f"   - Reconstructed Triangles: {len(mesh.faces):,}")
    print(f"   - Is Mesh Watertight:      {mesh.is_watertight}")
    print(f"   - Mesh Bounding Box Extents: {np.round(mesh.extents, 3)}")
    print(f"   - Saved Volumetric Mesh:   {out_obj}")

    print("\n Phase 3B Volumetric Reconstruction Engine Test Complete!")


if __name__ == "__main__":
    test_volumetric_reconstruction()
