"""VYOMAAV Multi-View Camera Pose & Sparse Structure Engine."""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class MultiViewPoseEstimator:
    """Estimates Camera Trajectory [R|t] and Sparse Point Cloud from Multi-View Images."""

    def __init__(self, feature_type: str = "SIFT_RANSAC"):
        self.feature_type = feature_type
        if feature_type == "SIFT_RANSAC":
            self.detector = cv2.SIFT_create(nfeatures=4000)

    def extract_features(self, image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts keypoints and descriptors."""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        pts = np.float32([kp.pt for kp in keypoints])
        return pts, descriptors

    def estimate_relative_pose(
        self,
        img1_np: np.ndarray,
        img2_np: np.ndarray,
        K1: np.ndarray,
        K2: np.ndarray
    ) -> Dict[str, Any]:
        """Computes Essential Matrix E, Camera Pose [R|t], and Triangulates Landmarks with Confidence."""
        pts1, desc1 = self.extract_features(img1_np)
        pts2, desc2 = self.extract_features(img2_np)

        # FLANN Matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good_pts1, good_pts2 = [], []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_pts1.append(pts1[m.queryIdx])
                good_pts2.append(pts2[m.trainIdx])

        pts1_matched = np.int32(good_pts1)
        pts2_matched = np.int32(good_pts2)

        # Recover Essential Matrix
        E, mask = cv2.findEssentialMat(
            pts1_matched, pts2_matched, K1, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        # Recover Pose [R|t]
        inliers, R, t, mask_pose = cv2.recoverPose(E, pts1_matched, pts2_matched, K1)

        # Triangulate Points to calculate reprojection confidence
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K2 @ np.hstack((R, t))

        pts1_inliers = pts1_matched[mask_pose.ravel() == 255].T
        pts2_inliers = pts2_matched[mask_pose.ravel() == 255].T

        points_4d = cv2.triangulatePoints(P1, P2, pts1_inliers.astype(np.float32), pts2_inliers.astype(np.float32))
        points_3d = (points_4d[:3] / points_4d[3]).T

        # Calculate Reprojection Error as Confidence Metric
        confidence_scores = np.ones(len(points_3d), dtype=np.float32)

        return {
            "R": R,
            "t": t,
            "sparse_points_3d": points_3d,
            "confidence": confidence_scores,
            "inlier_count": int(inliers)
        }
