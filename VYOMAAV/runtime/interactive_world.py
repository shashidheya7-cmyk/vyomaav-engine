"""VYOMAAV Sprint 38: Interactive World Layer (Locomotion, Collision, Navigation, Character Controls)."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class CharacterController:
    """Handles avatar movement state: walking, jumping, flying, and collision resolution."""

    def __init__(self, start_pos: Optional[List[float]] = None):
        self.position = np.array(start_pos or [0.0, 0.0, 1.0], dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.mode = "walk"  # 'walk', 'jump', 'fly'
        self.is_grounded = True

    def move(self, direction: List[float], speed: float = 2.0, delta_time: float = 0.016) -> List[float]:
        """Updates character position based on locomotion mode."""
        move_dir = np.array(direction, dtype=np.float32)
        if np.linalg.norm(move_dir) > 0:
            move_dir = move_dir / np.linalg.norm(move_dir)

        if self.mode == "walk":
            move_dir[2] = 0.0  # Lock vertical movement on ground
            self.position += move_dir * speed * delta_time
        elif self.mode == "fly":
            self.position += move_dir * speed * delta_time
        elif self.mode == "jump":
            if self.is_grounded:
                self.velocity[2] = 5.0
                self.is_grounded = False
            self.velocity[2] -= 9.8 * delta_time  # Gravity
            self.position += self.velocity * delta_time
            if self.position[2] <= 0.0:
                self.position[2] = 0.0
                self.is_grounded = True
                self.mode = "walk"

        return self.position.tolist()


class InteractiveWorldManager:
    """Manages full scene interactivity, character state, and navigation mesh bounds."""

    def __init__(self, scene: Any):
        self.scene = scene
        self.character = CharacterController()

    def generate_navmesh_nodes(self) -> Dict[str, Any]:
        """Extracts walkable floor surfaces from SOMG entities."""
        walkable_nodes = []
        if hasattr(self.scene, "base_graph") and hasattr(self.scene.base_graph, "nodes"):
            for node_id, entity in self.scene.base_graph.nodes.items():
                if hasattr(entity, "spatial") and hasattr(entity.spatial, "center"):
                    walkable_nodes.append({
                        "node_id": node_id,
                        "center": entity.spatial.center,
                        "walkable": True
                    })

            return {
                "navmesh_status": "generated",
                "walkable_nodes_count": len(walkable_nodes),
                "nodes": walkable_nodes
            }
