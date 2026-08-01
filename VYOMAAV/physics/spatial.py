"""VYOMAAV Spatial Bounding & Collision Verification."""
from typing import List

def check_bbox_intersection(min1: List[float], max1: List[float], min2: List[float], max2: List[float]) -> bool:
    return all(max1[i] >= min2[i] and max2[i] >= min1[i] for i in range(3))
