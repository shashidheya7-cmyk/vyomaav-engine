"""
VYOMAAV Base Model Engine
Module: somg.query

Query engine providing geometric spatial searching and graph traversals over SOMG scenes.
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import deque
from somg.scene import SceneState
from somg.graph import SpatialGraph, RelationType
from somg.entity import SOMGEntity
from somg.spatial_index import AABBSpatialIndex


class SOMGQueryEngine:
    """Executes geometric and relational queries over an active SceneState."""

    def __init__(self, scene: SceneState):
        self.scene = scene
        self.spatial_index = AABBSpatialIndex(cell_size=2.0)
        self.rebuild_spatial_index()

    def rebuild_spatial_index(self):
        """Builds spatial index from resolved scene graph."""
        graph = self.scene.resolve_active_graph()
        for entity in graph.nodes.values():
            if entity.spatial:
                self.spatial_index.insert_or_update(
                    entity.entity_id, entity.spatial.bbox_min, entity.spatial.bbox_max
                )

    def find_in_volume(self, bbox_min: List[float], bbox_max: List[float]) -> List[SOMGEntity]:
        """Finds all entities physically located within a 3D bounding box."""
        matching_ids = self.spatial_index.query_aabb_overlap(bbox_min, bbox_max)
        graph = self.scene.resolve_active_graph()
        return [graph.nodes[eid] for eid in matching_ids if eid in graph.nodes]

    def find_by_semantic_label(self, label: str) -> List[SOMGEntity]:
        """Filters entities by semantic label."""
        graph = self.scene.resolve_active_graph()
        return [e for e in graph.nodes.values() if e.semantic.label == label]

    def bfs_traversal(
        self, start_entity_id: str, max_depth: int = 3, relation_filter: Optional[RelationType] = None
    ) -> List[Tuple[SOMGEntity, int]]:
        """Executes Breadth-First Search graph traversal from a starting node."""
        graph = self.scene.resolve_active_graph()
        if start_entity_id not in graph.nodes:
            return []

        visited: Set[str] = {start_entity_id}
        queue: deque = deque([(start_entity_id, 0)])
        traversal_results: List[Tuple[SOMGEntity, int]] = []

        while queue:
            curr_id, depth = queue.popleft()
            if depth > max_depth:
                break

            if curr_id != start_entity_id:
                traversal_results.append((graph.nodes[curr_id], depth))

            # Traverse outgoing edges
            if curr_id in graph.outgoing_edges:
                for edge in graph.outgoing_edges[curr_id]:
                    if relation_filter is None or edge.relation_type == relation_filter:
                        if edge.target_id not in visited and edge.target_id in graph.nodes:
                            visited.add(edge.target_id)
                            queue.append((edge.target_id, depth + 1))

        return traversal_results