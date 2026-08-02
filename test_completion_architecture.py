"""Test Runner for VYOMAAV Refactored Neural Completion Architecture & Provenance Engine."""

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
from neural_completion.engine import MasterNeuralCompletionEngine


def test_completion_provenance():
    print("--- Executing VYOMAAV Neural Completion Architecture Verification ---")

    # 1. Camera & Geometry Foundation
    cam_model = PinholeCameraModel(fov_deg=60.0)
    info = cam_model.estimate_from_exif(Image.new("RGB", (640, 480)))
    K = info["K"]

    test_img = Image.new("RGB", (640, 480), color=(160, 160, 160))

    geom_engine = GeometryEstimationEngine()
    geom_pred = geom_engine.predict_full_geometry(test_img, K)

    # 2. Phase 3B: TSDF Integration
    tsdf_volume = TSDFVolumeGrid(grid_dim=128, trunc_margin=0.08)
    tsdf_volume.integrate_prediction(geom_pred, min_confidence_threshold=0.35)

    extractor = MarchingCubesExtractor()
    partial_mesh = extractor.extract_mesh(tsdf_volume, iso_level=0.0, smooth_sigma=0.8)

    # 3. Phase 3C: Refactored Master Completion Engine
    completion_engine = MasterNeuralCompletionEngine(backend="trellis")
    completion_res = completion_engine.complete_geometry(partial_mesh, geom_pred)

    # 4. Provenance & Diagnostics Audit
    obs_count = int(np.sum(~completion_res.vertex_provenance_mask))
    inf_count = int(np.sum(completion_res.vertex_provenance_mask))

    print("\n1. Neural Completion Diagnostics & Provenance Audit:")
    print(f"   - Selected Primary Backend:      {completion_res.metadata.get('selected_primary_backend')}")
    print(f"   - Model Confidence Score:        {completion_res.completion_confidence:.2f}")
    print(f"   - Total Fused Mesh Vertices:     {len(completion_res.fused_mesh.vertices):,}")
    print(f"   - Camera-Observed Vertices (GT): {obs_count:,} ({(obs_count/len(completion_res.fused_mesh.vertices)):.2%})")
    print(f"   - AI-Inferred Vertices (Prior):  {inf_count:,} ({(inf_count/len(completion_res.fused_mesh.vertices)):.2%})")
    print(f"   - Provenance Mask Tensor Length: {len(completion_res.vertex_provenance_mask):,}")

    os.makedirs("./real_world_output", exist_ok=True)
    completion_res.fused_mesh.export("./real_world_output/provenance_tracked_world.obj")
    print(f"   - Exported Provenance Asset:     ./real_world_output/provenance_tracked_world.obj")

    print("\n Refactored Neural Completion Architecture Test Complete!")


if __name__ == "__main__":
    test_completion_provenance()
