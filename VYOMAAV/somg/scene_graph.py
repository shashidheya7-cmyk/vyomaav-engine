"""Spatial Object Mesh Graph (SOMG) Module for Multi-Object Room Tracking."""

import os
import json
from typing import Dict, Any, List, Optional


class SpatialObjectMeshGraph:
    """Manages spatial relationships, positions, and mesh nodes across a 3D scene graph."""

    def __init__(self, scene_id: str = "UniversalScene"):
        self.scene_id = scene_id
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {
            "scene_id": scene_id,
            "status": "initializing",
            "global_coordinate_frame": "camera_relative_floor_aligned"
        }

    def add_entity_node(
        self,
        entity_id: str,
        label: str,
        position: List[float],
        scale: List[float],
        mesh_status: str = "completed"
    ) -> Dict[str, Any]:
        """Registers a 3D spatial entity node into the SOMG graph."""
        node_data = {
            "entity_id": entity_id,
            "label": label,
            "spatial_box": {
                "position_xyz": position,
                "scale_extents": scale
            },
            "status": mesh_status
        }
        self.nodes[entity_id] = node_data
        return node_data

    def export_summary(self, filepath: str) -> str:
        """Exports the SOMG scene graph structure as a JSON summary file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        self.metadata["status"] = "world_reconstruction_complete"
        self.metadata["total_entities"] = len(self.nodes)

        export_payload = {
            "metadata": self.metadata,
            "nodes": self.nodes
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)

        return filepath


class ActiveGraphResolver:
    """Helper wrapper for node resolution in legacy calls."""
    def __init__(self, graph: SpatialObjectMeshGraph):
        self.graph = graph

    def get_node(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.graph.nodes.get(entity_id)
