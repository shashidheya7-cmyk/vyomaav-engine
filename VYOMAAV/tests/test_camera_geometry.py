import numpy as np
import pytest
from reconstruction.camera import CameraPoseEstimator

def test_camera_estimation_and_calibration():
    estimator = CameraPoseEstimator()
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    res = estimator.estimate_camera_and_sparse_cloud([dummy_img])

    assert res["status"] == "calibrated"
    assert "intrinsics" in res
    assert np.array(res["intrinsics"]).shape == (3, 3)
    assert len(res["camera_trajectories"]) == 1
    assert len(res["sparse_point_cloud"]) == 256
