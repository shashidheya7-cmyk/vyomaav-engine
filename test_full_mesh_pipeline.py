"""Test Runner for VYOMAAV Topology Validation, Repair, and Mesh Optimization Pipeline."""

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
from meshing.topology_validator import TopologyValidatorEngine
from meshing.retopology import MeshRetopologyEngine
from materials.pbr_baker import ProvenanceAwarePBRBaker


def test_topology_and_optimization():
    print("--- Executing VYOMAAV Topology Validation & Mesh Optimization Pipeline ---")

    # 1. Geometry Foundation & TSDF Integration
    cam_model = PinholeCameraModel(fov_deg=60.0)
    info = cam_model.estimate_from_exif(Image.new("RGB", (640, 480)))
    K = info["K"]

    test_img = Image.new("RGB", (640, 480), color=(150, 150, 150))

    geom_engine = GeometryEstimationEngine()
    geom_pred = geom_engine.predict_full_geometry(test_img, K)

    tsdf_volume = TSDFVolumeGrid(grid_dim=128, trunc_margin=0.08)
    tsdf_volume.integrate_prediction(geom_pred, min_confidence_threshold=0.35)

    extractor = MarchingCubesExtractor()
    partial_mesh = extractor.extract_mesh(tsdf_volume, iso_level=0.0, smooth_sigma=0.8)

    # 2. Neural Completion Framework
    completion_engine = MasterNeuralCompletionEngine(backend="trellis")
    completion_res = completion_engine.complete_geometry(partial_mesh, geom_pred)

    # 3. Topology Audit & Automated Repair
    print("\n1. Running Topology Audit & Repair Engine...")
    validator = TopologyValidatorEngine()
    
    pre_audit = validator.audit_topology(completion_res.fused_mesh)
    print(f"   - Raw Vertices / Faces:        {pre_audit.vertex_count:,} / {pre_audit.face_count:,}")
    print(f"   - Raw Degenerate Faces:       {pre_audit.degenerate_face_count}")
    print(f"   - Raw Is Watertight:          {pre_audit.is_watertight}")

    repaired_mesh = validator.repair_and_seal_topology(completion_res.fused_mesh)
    post_audit = validator.audit_topology(repaired_mesh)

    print(f"\n2. Post-Repair Topology Health:")
    print(f"   - Repaired Vertices / Faces:   {post_audit.vertex_count:,} / {post_audit.face_count:,}")
    print(f"   - Degenerate Faces Remaining: {post_audit.degenerate_face_count}")
    print(f"   - Euler Characteristic (V-E+F): {post_audit.euler_number}")

    # 4. QEM Retopology & Decimation
    print("\n3. Executing QEM Retopology & Polygon Decimation...")
    retop_engine = MeshRetopologyEngine()
    decimated_mesh = retop_engine.decimate(repaired_mesh, target_face_count=3500)

    # 5. Provenance-Aware PBR Material Baking
    baker = ProvenanceAwarePBRBaker()
    pbr_maps = baker.bake_textures(
        test_img, geom_pred.normals, completion_res.vertex_provenance_mask
    )

    # 6. Asset Export (.OBJ, .GLB)
    os.makedirs("./real_world_output", exist_ok=True)
    out_obj = "./real_world_output/optimized_world_mesh.obj"
    out_glb = "./real_world_output/optimized_world_mesh.glb"

    decimated_mesh.export(out_obj)
    decimated_mesh.export(out_glb)

    print(f"\n4. Final Production Asset Export Audit:")
    print(f"   - Final Vertices:              {len(decimated_mesh.vertices):,}")
    print(f"   - Final Triangles:             {len(decimated_mesh.faces):,}")
    print(f"   - PBR Albedo Atlas Resolution: {pbr_maps['albedo'].size}")
    print(f"   - Exported Production OBJ:     {out_obj}")
    print(f"   - Exported Production GLB:     {out_glb}")

    print("\n Topology Validation & Mesh Optimization Test Complete!")


if __name__ == "__main__":
    test_topology_and_optimization()
