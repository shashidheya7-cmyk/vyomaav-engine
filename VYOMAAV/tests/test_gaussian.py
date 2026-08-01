import pytest
import numpy as np
from reconstruction.trellis import TRELLISReconstructionBackend
from reconstruction.gaussian import GaussianSplatEngine
from somg.scene import SceneState
from somg.entity import SOMGEntity

def test_gaussian_splat_initialization():
    trellis_engine = TRELLISReconstructionBackend()
    gaussian_engine = GaussianSplatEngine(sh_degree=3)

    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mesh_data = trellis_engine.reconstruct(dummy_img)

    splat_res = gaussian_engine.initialize_gaussians_from_mesh(mesh_data)

    assert splat_res["status"] == "splat_initialized"
    assert splat_res["num_gaussians"] == 128
    assert len(splat_res["positions"]) == 128
    assert len(splat_res["rotations"]) == 128
    assert splat_res["sh_degree"] == 3

def test_gaussian_splat_somg_integration():
    trellis_engine = TRELLISReconstructionBackend()
    gaussian_engine = GaussianSplatEngine(sh_degree=3)

    scene = SceneState("GaussianScene")
    entity = SOMGEntity("vase_node_0")
    scene.base_graph.add_node(entity)

    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mesh_data = trellis_engine.reconstruct(dummy_img)
    splat_res = gaussian_engine.initialize_gaussians_from_mesh(mesh_data)

    scene = gaussian_engine.integrate_gaussians_into_somg(scene, "vase_node_0", splat_res)
    target_node = scene.base_graph.nodes["vase_node_0"]

    assert "gaussian_splat" in target_node.attributes
    assert target_node.attributes["gaussian_splat"]["status"] == "ready_for_view_synthesis"
