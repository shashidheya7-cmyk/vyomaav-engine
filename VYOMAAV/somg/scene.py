"""VYOMAAV Spatial Object Mesh Graph (SOMG) Scene Definitions."""
from typing import Dict, Any, Optional
from somg.entity import SOMGEntity

class SOMGGraph:
    def __init__(self):
        self.nodes: Dict[str, SOMGEntity] = {}
        self.edges: list = []

    def add_node(self, node: SOMGEntity):
        self.nodes[node.entity_id] = node

class SceneState:
    def __init__(self, scene_id: str = "default_scene"):
        self.scene_id = scene_id
        self.base_graph = SOMGGraph()
        self.metadata: Dict[str, Any] = {}

    def resolve_active_graph(self) -> SOMGGraph:
        return self.base_graph
