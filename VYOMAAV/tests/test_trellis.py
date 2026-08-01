import os
import numpy as np
import pytest
from reconstruction.trellis import TRELLISReconstructionBackend
from reconstruction.exporter import MeshExporter
from somg.scene import SceneState
from somg.entity import SOMGEntity

def test_trellis_reconstruction_pipeline(tmp_path):
    backend = TRELLISReconstructionBackend()
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    out_dir = str(tmp_path / "trellis_out")

    res = backend.reconstruct(dummy_img, output_dir=out_dir)

    assert res["status"] == "reconstructed"
    assert len(res["vertices"]) == 128
    assert len(res["normals"]) == 128
    assert len(res["uvs"]) == 128
    assert len(res["faces"]) == 240
    assert res["topology"]["is_watertight"] is True

def test_trellis_somg_adapter_and_obj_exporter(tmp_path):
    backend = TRELLISReconstructionBackend()
    scene = SceneState(scene_id="ReconstructionScene")
    entity = SOMGEntity("chair_node_0")
    scene.base_graph.add_node(entity)

    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = backend.reconstruct(dummy_img)

    scene = backend.integrate_into_somg(scene, "chair_node_0", res)
    target_node = scene.base_graph.nodes["chair_node_0"]

    assert hasattr(target_node, "mesh_geometry")
    assert target_node.mesh_geometry["backend"] == "TRELLIS"

    # Export to .obj file
    obj_path = str(tmp_path / "mesh.obj")
    exported_file = MeshExporter.export_to_obj(target_node.mesh_geometry, obj_path)
    assert os.path.exists(exported_file)
