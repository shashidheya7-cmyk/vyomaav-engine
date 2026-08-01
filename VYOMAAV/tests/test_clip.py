import numpy as np
from perception.clip import CLIPPredictor

def test_clip_inference():
    predictor = CLIPPredictor()
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    emb = predictor.get_image_embedding(dummy_img)
    assert emb.shape == (512,)
