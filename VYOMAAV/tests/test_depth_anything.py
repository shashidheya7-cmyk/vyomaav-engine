import numpy as np
from perception.depth_anything import DepthAnythingPredictor

def test_depth_anything_inference():
    predictor = DepthAnythingPredictor()
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = predictor.infer(dummy_img)
    assert "depth" in res
    assert res["depth"].shape == (256, 256)
