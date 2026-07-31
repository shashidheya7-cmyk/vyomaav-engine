"""
VYOMAAV Base Model Engine
Module: engine.pathfinding

A* Navigation Pathfinding & Waypoint Trajectory Engine (Sprint 15).
Executes A* graph search over Recast Navigation Meshes, evaluates Euclidean 3D
heuristics, accounts for jump-link and flight cost penalties, and generates
smoothed 3D waypoint trajectories for Autonomous Agent Locomotion.
"""

from dataclasses import dataclass, field
import heapq
import math
from typing import List, Dict, Set, Tuple, Optional
import torch

from engine.physics import NavigationMesh, NavMeshPolygon


@dataclass
class WayPoint:
    """3D spatial waypoint along a navigation trajectory."""
    position: List[float]       # [x, y, z] in meters
    poly_id: int                # Corresponding NavMesh polygon ID
    is_jump: bool = False       # True if waypoint represents a jump action
    is_fly: bool = False        # True if waypoint represents a flight action


@dataclass
class PathfindingResult:
    """Output container for pathfinding query results."""
    found: bool
    total_cost: float
    path_poly_ids: List[int] = field(default_factory=list)
    waypoints: List[WayPoint] = field(default_factory=list)


@dataclass(order=True)
class PriorityNode:
    """Priority queue item for A* search."""
    f_score: float
    poly_id: int = field(compare=False)


class AStarNavMeshPathfinder:
    """A* Graph Search Engine over Recast Navigation Meshes."""

    def __init__(
        self,
        navmesh: NavigationMesh,
        jump_cost_penalty: float = 2.5,
        fly_cost_penalty: float = 1.0
    ):
        self.navmesh = navmesh
        self.jump_penalty = jump_cost_penalty
        self.fly_penalty = fly_cost_penalty
        self.poly_map: Dict[int, NavMeshPolygon] = {
            p.poly_id: p for p in navmesh.polygons
        }

    def _euclidean_distance(self, p1: torch.Tensor, p2: torch.Tensor) -> float:
        """Calculates 3D Euclidean distance heuristic between two points."""
        return torch.norm(p1 - p2).item()

    def find_nearest_polygon(self, point: List[float], device: torch.device = torch.device("cpu")) -> Optional[int]:
        """Finds the nearest NavMesh polygon centroid to a given 3D point."""
        if not self.navmesh.polygons:
            return None

        pt_tensor = torch.tensor(point, dtype=torch.float32, device=device)
        best_poly_id = None
        min_dist = float("inf")

        for poly in self.navmesh.polygons:
            dist = torch.norm(poly.centroid.to(device) - pt_tensor).item()
            if dist < min_dist:
                min_dist = dist
                best_poly_id = poly.poly_id

        return best_poly_id

    def find_path(
        self,
        start_pos: List[float],
        target_pos: List[float],
        device: torch.device = torch.device("cpu")
    ) -> PathfindingResult:
        """Executes A* search from start_pos to target_pos over the Navigation Mesh."""
        start_poly_id = self.find_nearest_polygon(start_pos, device=device)
        target_poly_id = self.find_nearest_polygon(target_pos, device=device)

        if start_poly_id is None or target_poly_id is None:
            return PathfindingResult(found=False, total_cost=float("inf"))

        target_pt = torch.tensor(target_pos, dtype=torch.float32, device=device)

        # Priority Queue for A*
        open_set: List[PriorityNode] = []
        heapq.heappush(open_set, PriorityNode(f_score=0.0, poly_id=start_poly_id))

        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {p.poly_id: float("inf") for p in self.navmesh.polygons}
        g_score[start_poly_id] = 0.0

        f_score: Dict[int, float] = {p.poly_id: float("inf") for p in self.navmesh.polygons}
        start_centroid = self.poly_map[start_poly_id].centroid.to(device)
        f_score[start_poly_id] = self._euclidean_distance(start_centroid, target_pt)

        open_set_ids: Set[int] = {start_poly_id}

        while open_set:
            current_node = heapq.heappop(open_set)
            current_id = current_node.poly_id
            open_set_ids.discard(current_id)

            if current_id == target_poly_id:
                # Target reached: Reconstruct path
                return self._reconstruct_path(
                    came_from, current_id, start_pos, target_pos, g_score[current_id]
                )

            current_poly = self.poly_map[current_id]
            neighbors = self.navmesh.adjacency.get(current_id, set())

            for neighbor_id in neighbors:
                if neighbor_id not in self.poly_map:
                    continue

                neighbor_poly = self.poly_map[neighbor_id]

                # Base edge cost = Euclidean distance between centroids
                dist = self._euclidean_distance(
                    current_poly.centroid.to(device), neighbor_poly.centroid.to(device)
                )

                # Apply action cost penalties
                if neighbor_poly.is_jump_link:
                    edge_cost = dist * self.jump_penalty
                elif neighbor_poly.is_flyable:
                    edge_cost = dist * self.fly_penalty
                else:
                    edge_cost = dist

                tentative_g = g_score[current_id] + edge_cost

                if tentative_g < g_score[neighbor_id]:
                    came_from[neighbor_id] = current_id
                    g_score[neighbor_id] = tentative_g
                    
                    h = self._euclidean_distance(neighbor_poly.centroid.to(device), target_pt)
                    f = tentative_g + h
                    f_score[neighbor_id] = f

                    if neighbor_id not in open_set_ids:
                        heapq.heappush(open_set, PriorityNode(f_score=f, poly_id=neighbor_id))
                        open_set_ids.add(neighbor_id)

        # Path not found
        return PathfindingResult(found=False, total_cost=float("inf"))

    def _reconstruct_path(
        self,
        came_from: Dict[int, int],
        current_id: int,
        start_pos: List[float],
        target_pos: List[float],
        total_cost: float
    ) -> PathfindingResult:
        """Reconstructs polynomial path ID sequence and extracts 3D Waypoints."""
        path_ids = [current_id]
        while current_id in came_from:
            current_id = came_from[current_id]
            path_ids.append(current_id)

        path_ids.reverse()

        # Build Waypoints
        waypoints: List[WayPoint] = []
        # Add exact start position waypoint
        waypoints.append(WayPoint(position=start_pos, poly_id=path_ids[0]))

        # Intermediate centroids
        for poly_id in path_ids[1:-1]:
            poly = self.poly_map[poly_id]
            pos = poly.centroid.tolist()
            waypoints.append(
                WayPoint(
                    position=pos,
                    poly_id=poly_id,
                    is_jump=poly.is_jump_link,
                    is_fly=poly.is_flyable
                )
            )

        # Add exact target position waypoint
        if len(path_ids) > 1:
            waypoints.append(WayPoint(position=target_pos, poly_id=path_ids[-1]))

        return PathfindingResult(
            found=True,
            total_cost=total_cost,
            path_poly_ids=path_ids,
            waypoints=waypoints
        )