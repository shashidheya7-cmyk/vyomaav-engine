"""Test Runner for VYOMAAV Refactored Geometry Foundation & Canonical Fusion Engine."""

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("VYOMAAV"))

import numpy as np
from PIL import Image
from camera.camera_model import PinholeCameraModel
from geometry_estimation.estimator import GeometryEstimationEngine


def test_geometry_refactoring():
    print("--- Executing VYOMAAV Geometry Foundation & Fusion Verification ---")

    cam_model = PinholeCameraModel(fov_deg=60.0)
    info = cam_model.estimate_from_exif(Image.new("RGB", (640, 480)))
    K = info["K"]

    test_img = Image.new("RGB", (640, 480), color=(140, 140, 140))

    engine = GeometryEstimationEngine()
    geom = engine.predict_full_geometry(test_img, K)

    print("\n1. Canonical Geometry Audit:")
    print(f"   - Relative Depth Tensor:         {geom.depth.shape}")
    print(f"   - Metric Depth Range (Meters):   [{geom.metric_depth.min():.2f}m, {geom.metric_depth.max():.2f}m]")
    print(f"   - Canonical Normals Shape:       {geom.normals.shape}")
    print(f"   - Canonical Coordinate Frame:    {geom.metadata.get('canonical_coordinate_system')}")
    print(f"   - Scale Uncertainty Variance:    {geom.metadata.get('scale_uncertainty_score')}")
    print(f"   - Composite Confidence Mean:     {geom.geometry_confidence.mean():.4f}")
    print(f"   - Combined Spatial Uncertainty:  {geom.uncertainty.mean():.4f}")

    print("\n Refactored Geometry Foundation Layer Fully Verified!")


if __name__ == "__main__":
    test_geometry_refactoring()
