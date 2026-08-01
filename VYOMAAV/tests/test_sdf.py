import pytest
import numpy as np
from reconstruction.sdf import NeuralSDFEngine
from somg.scene import SceneState
from somg.entity import SOMGEntity

def test_neural_sdf_implicit_mesh_extraction():
    sdf_engine = NeuralSDFEngine(grid_resolution=16)
    dummy_input = {"vertices": np.random.uniform(-0.5, 0.5, size=(64, 3)).tolist()}

    res = sdf_engine.compute_implicit_sdf_and_watertight_mesh(dummy_input)

    assert res["status"] == "sdf_surface_extracted"
    assert res["is_watertight"] is True
    assert res["physics_ready"] is True
    assert len(res["vertices"]) == 120
    assert len(res["faces"]) == 236

def test_sdf_collision_mesh_somg_integration():
    sdf_engine = NeuralSDFEngine(grid_resolution=16)
    scene = SceneState("SDFScene")
    entity = SOMGEntity("obstacle_0")
    scene.base_graph.add_node(entity)

    dummy_input = {"vertices": np.random.uniform(-0.5, 0.5, size=(64, 3)).tolist()}
    res = sdf_engine.compute_implicit_sdf_and_watertight_mesh(dummy_input)

    scene = sdf_engine.integrate_sdf_into_somg(scene, "obstacle_0", res)
    target = scene.base_graph.nodes["obstacle_0"]

    assert "collision_mesh" in target.attributes
    assert target.attributes["collision_mesh"]["physics_ready"] is True
