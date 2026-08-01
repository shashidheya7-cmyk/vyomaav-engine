import pytest
from reconstruction.materials import PBRMaterialGenerator
from somg.scene import SceneState
from somg.entity import SOMGEntity

def test_pbr_material_generation():
    generator = PBRMaterialGenerator(texture_res=512)
    res = generator.generate_pbr_maps(semantic_label="wooden_chair")

    assert res["status"] == "pbr_generated"
    assert "properties" in res
    assert res["properties"]["workflow"] == "metallic_roughness"
    assert "albedo" in res["maps"]

def test_pbr_somg_integration():
    generator = PBRMaterialGenerator(texture_res=512)
    scene = SceneState("PBRScene")
    entity = SOMGEntity("chair_0")
    scene.base_graph.add_node(entity)

    res = generator.generate_pbr_maps(semantic_label="wooden_chair")
    scene = generator.integrate_pbr_into_somg(scene, "chair_0", res)
    target = scene.base_graph.nodes["chair_0"]

    assert "pbr_material" in target.attributes
    assert target.attributes["pbr_material"]["roughness"] == 0.35
