import pytest
import numpy as np
from reconstruction.trellis import TRELLISReconstructionBackend
from reconstruction.hunyuan3d import Hunyuan3DReconstructionBackend
from reconstruction.fusion import GeometryFusionEngine
from somg.scene import SceneState
from somg.entity import SOMGEntity

def test_geometry_fusion_trellis_hunyuan3d(tmp_path):
    trellis_engine = TRELLISReconstructionBackend()
    hunyuan_engine = Hunyuan3DReconstructionBackend()
    fusion_engine = GeometryFusionEngine()

    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    out_dir = str(tmp_path / "fusion_out")

    t_res = trellis_engine.reconstruct(dummy_img)
    h_res = hunyuan_engine.reconstruct(dummy_img)

    fused_res = fusion_engine.fuse_reconstruction_results(t_res, h_res, output_dir=out_dir)

    assert fused_res["status"] == "fused"
    assert "TRELLIS" in fused_res["source_backends"]
    assert "Hunyuan3D" in fused_res["source_backends"]
    assert fused_res["vertex_count"] >= 256
    assert fused_res["topology"]["fusion_quality_score"] > 0.90

def test_fused_mesh_somg_integration():
    trellis_engine = TRELLISReconstructionBackend()
    hunyuan_engine = Hunyuan3DReconstructionBackend()
    fusion_engine = GeometryFusionEngine()

    scene = SceneState("FusedScene")
    entity = SOMGEntity("sofa_node_0")
    scene.base_graph.add_node(entity)

    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    t_res = trellis_engine.reconstruct(dummy_img)
    h_res = hunyuan_engine.reconstruct(dummy_img)

    fused_res = fusion_engine.fuse_reconstruction_results(t_res, h_res)
    scene = fusion_engine.integrate_fused_mesh_into_somg(scene, "sofa_node_0", fused_res)

    target_node = scene.base_graph.nodes["sofa_node_0"]
    assert hasattr(target_node, "mesh_geometry")
    assert target_node.mesh_geometry["backend"] == "UNIFIED_FUSED_ENGINE"
    assert len(target_node.mesh_geometry["vertices"]) >= 256
