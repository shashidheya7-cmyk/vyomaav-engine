"""Test Runner for VYOMAAV Phase 3.4 Geometric Registration Engine using Synthetic Motion."""

import cv2
import numpy as np
from PIL import Image
from camera.camera_model import PinholeCameraModel
from registration.matcher import FeatureRegistrationMatcher
from registration.pose_solver import RelativePoseSolver


def generate_textured_scene(width: int = 640, height: int = 480) -> np.ndarray:
    """Generates a high-contrast geometric scene with corners and textures for feature matching."""
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    
    # Draw geometric features (grid, circles, text, shapes)
    for i in range(0, width, 40):
        cv2.line(img, (i, 0), (i, height), (50, 50, 50), 2)
    for j in range(0, height, 40):
        cv2.line(img, (0, j), (width, j), (50, 50, 50), 2)

    cv2.circle(img, (200, 200), 80, (220, 50, 50), -1)
    cv2.rectangle(img, (350, 150), (550, 350), (50, 200, 50), -1)
    cv2.putText(img, "VYOMAAV 3D", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 220), 4)

    return img


def test_registration():
    print("--- Executing VYOMAAV Geometric Registration Engine (Phase 3.4) ---")

    # 1. Create Camera Model & Intrinsics K
    cam_model = PinholeCameraModel(fov_deg=60.0)
    info = cam_model.estimate_from_exif(Image.new("RGB", (640, 480)))
    K = info["K"]

    # 2. Generate Camera View 1 and View 2 (simulating camera pan/rotation)
    img1 = generate_textured_scene(640, 480)

    # Perspective shift matrix simulating camera motion
    H = np.array([
        [0.98, -0.02, 15.0],
        [0.01,  0.99, 10.0],
        [0.0001, 0.00,  1.0]
    ], dtype=np.float32)
    img2 = cv2.warpPerspective(img1, H, (640, 480))

    # 3. Match Features across View 1 and View 2
    matcher = FeatureRegistrationMatcher(method="sift_flann_fallback")
    match_pred = matcher.match_pair(img1, img2)

    print(f"\n1. Feature Correspondence Analysis:")
    print(f"   - Total Raw Matches: {match_pred.metadata.get('total_raw_matches', 0)}")
    print(f"   - RANSAC Inliers Count: {match_pred.inlier_count}")
    print(f"   - Inlier Ratio: {match_pred.inlier_ratio:.2%}")

    # 4. Solve Relative Extrinsics [R|t]
    solver = RelativePoseSolver()
    pose_pred = solver.solve_pose(match_pred, K, K)

    print(f"\n2. Camera Pose Solver Diagnostics:")
    print(f"   - Pose Confidence Score: {pose_pred.confidence:.4f}")
    print(f"   - Mean Reprojection Error: {pose_pred.reprojection_error:.4f} pixels")
    print(f"   - Rotation Matrix R:\n{np.round(pose_pred.R, 4)}")
    print(f"   - Translation Vector t:\n{np.round(pose_pred.t.T, 4)}")

    print("\n Phase 3.4 Geometric Registration Engine Test Complete!")


if __name__ == "__main__":
    test_registration()
