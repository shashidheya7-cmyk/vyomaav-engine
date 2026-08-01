import numpy as np
from perception.sam2 import SAM2Predictor

def test_sam2_inference():
    predictor = SAM2Predictor()
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = predictor.infer(dummy_img)
    assert "masks" in res
    assert len(res["masks"]) > 0
