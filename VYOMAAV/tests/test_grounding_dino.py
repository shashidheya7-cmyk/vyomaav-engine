import numpy as np
from perception.grounding_dino import GroundingDINOPredictor

def test_grounding_dino_inference():
    predictor = GroundingDINOPredictor()
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = predictor.infer(dummy_img, text_prompt="robot . chair")
    assert "bboxes_2d" in res
    assert len(res["bboxes_2d"]) > 0
