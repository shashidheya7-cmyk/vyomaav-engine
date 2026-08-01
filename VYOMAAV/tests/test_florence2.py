import numpy as np
from perception.florence2 import Florence2Predictor

def test_florence2_inference():
    predictor = Florence2Predictor()
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = predictor.infer(dummy_img, task_prompt="<MORE_DETAILED_CAPTION>")
    assert "caption" in res
    assert isinstance(res["caption"], str)
