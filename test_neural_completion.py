"""Test Runner for Integrated Phase 3B TSDF + Phase 3C Neural Geometry Completion Engine."""

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
from neural_completion.completion_engine import NeuralGeometryCompletionEngine


def test_full_reconstruction_pipeline():
    print("--- Executing VYOMAAV Phase 3B (TSDF) + Phase 3C (Neural Completion) Pipeline ---")

    # 1. Camera Intrinsics & Geometry Prediction
    cam_model = PinholeCameraModel(fov_deg=60.0)
    info = cam_model.estimate_from_exif(Image.new("RGB", (640, 480)))
    K = info["K"]

    test_img = Image.new("RGB", (640, 480), color=(170, 170, 170))

    geom_engine = GeometryEstimationEngine()
    geom_pred = geom_engine.predict_full_geometry(test_img, K)

    # 2. Phase 3B: TSDF Integration with Variance Tracking & Gaussian Smoothing
    tsdf_volume = TSDFVolumeGrid(grid_dim=128, trunc_margin=0.08)
    tsdf_volume.integrate_prediction(geom_pred, min_confidence_threshold=0.35)

    extractor = MarchingCubesExtractor()
    partial_mesh = extractor.extract_mesh(tsdf_volume, iso_level=0.0, smooth_sigma=0.8)

    print(f"\n1. Phase 3B Partial Front Shell Extracted:")
    print(f"   - Partial Vertices:  {len(partial_mesh.vertices):,}")
    print(f"   - Is Watertight:      {partial_mesh.is_watertight}")

    # 3. Phase 3C: Neural Geometry Completion
    completer = NeuralGeometryCompletionEngine(backend="trellis_prior")
    watertight_mesh = completer.complete_unseen_geometry(partial_mesh, geom_pred)

    os.makedirs("./real_world_output", exist_ok=True)
    out_watertight = "./real_world_output/watertight_world_mesh.obj"
    watertight_mesh.export(out_watertight)

    print(f"\n2. Phase 3C Neural Completion Audit:")
    print(f"   - Final Vertices:    {len(watertight_mesh.vertices):,}")
    print(f"   - Final Triangles:   {len(watertight_mesh.faces):,}")
    print(f"   - Is Mesh Watertight:{watertight_mesh.is_watertight}")
    print(f"   - Extents (Meters):  {np.round(watertight_mesh.extents, 3)}")
    print(f"   - Output Export:     {out_watertight}")

    print("\n Phase 3B + Phase 3C Pipeline Test Complete!")


if __name__ == "__main__":
    test_full_reconstruction_pipeline()
