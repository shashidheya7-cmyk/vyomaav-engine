from somg.entity import SOMGEntity, SpatialComponent, PhysicsComponent
from somg.scene import SceneState

def test_somg_entity_and_spatial_dimensions():
    spatial = SpatialComponent(bbox_min=[0.0, 0.0, 0.0], bbox_max=[2.0, 4.0, 6.0])
    entity = SOMGEntity("robot_base", spatial=spatial)
    assert entity.entity_id == "robot_base"
    assert entity.spatial.dimensions == [2.0, 4.0, 6.0]
    assert entity.spatial.center == [1.0, 2.0, 3.0]

def test_somg_scene_graph_node_addition():
    scene = SceneState("TestScene")
    entity = SOMGEntity("item_0")
    scene.base_graph.add_node(entity)
    nodes = scene.resolve_active_graph().nodes
    assert "item_0" in nodes
    assert nodes["item_0"].semantic_label == "object"
