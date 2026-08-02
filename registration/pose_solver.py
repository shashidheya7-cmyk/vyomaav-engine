"""VYOMAAV Relative Camera Pose Solver Engine."""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from core.types import MatchPrediction, PosePrediction


class RelativePoseSolver:
    """Solves Essential Matrix E, Extrinsics [R|t], and evaluates pose uncertainty."""

    def solve_pose(
        self,
        match_pred: MatchPrediction,
        K1: np.ndarray,
        K2: np.ndarray
    ) -> PosePrediction:
        """Computes relative Rotation R and Translation t with rigorous diagnostic metrics."""
        pts1 = match_pred.keypoints1[match_pred.inlier_mask]
        pts2 = match_pred.keypoints2[match_pred.inlier_mask]

        if len(pts1) < 8:
            return PosePrediction(
                R=np.eye(3, dtype=np.float32),
                t=np.zeros((3, 1), dtype=np.float32),
                confidence=0.0,
                inlier_ratio=0.0,
                reprojection_error=999.0,
                metadata={"status": "insufficient_inliers"}
            )

        # Essential Matrix recovery via RANSAC
        E, mask_E = cv2.findEssentialMat(
            pts1, pts2, K1, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None or E.shape != (3, 3):
            return PosePrediction(
                R=np.eye(3, dtype=np.float32),
                t=np.zeros((3, 1), dtype=np.float32),
                confidence=0.0,
                inlier_ratio=0.0,
                reprojection_error=999.0,
                metadata={"status": "essential_matrix_failed"}
            )

        # Recover Pose [R|t]
        inliers, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K1)

        # Calculate geometric epipolar reprojection error
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        
        pts1_homo = np.hstack([pts1, np.ones((len(pts1), 1))])
        pts2_homo = np.hstack([pts2, np.ones((len(pts2), 1))])

        lines2 = (F @ pts1_homo.T).T
        lines2_norm = np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2) + 1e-6
        errors = np.abs(np.sum(pts2_homo * lines2, axis=1)) / lines2_norm
        mean_reprojection_error = float(np.mean(errors))

        # Calculate composite pose confidence
        inlier_ratio = float(inliers / len(match_pred.keypoints1)) if len(match_pred.keypoints1) > 0 else 0.0
        error_score = max(0.0, 1.0 - (mean_reprojection_error / 5.0))
        pose_confidence = float(np.clip(0.6 * inlier_ratio + 0.4 * error_score, 0.0, 1.0))

        return PosePrediction(
            R=R,
            t=t,
            confidence=pose_confidence,
            inlier_ratio=inlier_ratio,
            reprojection_error=mean_reprojection_error,
            essential_matrix=E,
            fundamental_matrix=F,
            metadata={"inlier_count": int(inliers)}
        )
