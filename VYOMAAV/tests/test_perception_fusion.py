import os, torch, numpy as np, pytest
from perception.depth_anything import DepthAnythingPredictor
from perception.sam2 import SAM2Predictor
from perception.grounding_dino import GroundingDINOPredictor
from perception.florence2 import Florence2Predictor
from perception.clip import CLIPPredictor
from somg.scene import SceneState

def test_end_to_end_multimodal_perception_fusion(tmp_path):
    scene = SceneState(scene_id="UnifiedPerceptionScene")
    depth_p = DepthAnythingPredictor()
    sam2_p = SAM2Predictor()
    dino_p = GroundingDINOPredictor()
    florence_p = Florence2Predictor()
    clip_p = CLIPPredictor()

    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    out_dir = str(tmp_path / "fusion_output")

    depth_res = depth_p.infer(dummy_img, output_dir=out_dir)
    sam2_res = sam2_p.infer(dummy_img, output_dir=out_dir)
    dino_res = dino_p.infer(dummy_img, text_prompt="robot . obstacle", output_dir=out_dir)
    florence_res = florence_p.infer(dummy_img, task_prompt="<MORE_DETAILED_CAPTION>", output_dir=out_dir)
    clip_emb = clip_p.get_image_embedding(dummy_img)

    scene = dino_p.integrate_into_somg(scene, dino_res, depth_result=depth_res)
    scene = sam2_p.integrate_into_somg(scene, sam2_res, depth_result=depth_res)
    scene = florence_p.integrate_into_somg(scene, florence_res)

    active_nodes = scene.resolve_active_graph().nodes
    assert len(active_nodes) > 0
    first_node_id = list(active_nodes.keys())[0]
    scene = clip_p.integrate_into_somg(scene, first_node_id, clip_emb)

    target_node = active_nodes[first_node_id]
    assert hasattr(target_node, "spatial")
    assert hasattr(target_node, "clip_embedding")
    assert len(target_node.clip_embedding) == 512
