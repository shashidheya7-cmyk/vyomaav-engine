import os
import numpy as np
import pytest
from reconstruction.hunyuan3d import Hunyuan3DReconstructionBackend
from somg.scene import SceneState
from somg.entity import SOMGEntity

def test_hunyuan3d_reconstruction_pipeline(tmp_path):
    backend = Hunyuan3DReconstructionBackend()
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    out_dir = str(tmp_path / "hunyuan_out")

    res = backend.reconstruct(dummy_img, output_dir=out_dir)

    assert res["status"] == "reconstructed"
    assert res["backend"] == "Hunyuan3D"
    assert len(res["vertices"]) == 256
    assert len(res["faces"]) == 480
    assert "textures" in res

def test_hunyuan3d_somg_adapter():
    backend = Hunyuan3DReconstructionBackend()
    scene = SceneState(scene_id="HunyuanScene")
    entity = SOMGEntity("table_node_0")
    scene.base_graph.add_node(entity)

    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = backend.reconstruct(dummy_img)

    scene = backend.integrate_into_somg(scene, "table_node_0", res)
    target_node = scene.base_graph.nodes["table_node_0"]

    assert hasattr(target_node, "mesh_geometry")
    assert target_node.mesh_geometry["backend"] == "Hunyuan3D"
    assert len(target_node.mesh_geometry["vertices"]) == 256
