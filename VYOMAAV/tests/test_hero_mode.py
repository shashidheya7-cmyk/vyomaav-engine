import pytest
import numpy as np
from reconstruction.hero_pipeline import WorldReconstructionPipeline

def test_end_to_end_image_to_3d_world_reconstruction(tmp_path):
    pipeline = WorldReconstructionPipeline()
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    out_dir = str(tmp_path / "hero_world_output")

    summary = pipeline.process_media_to_3d_world(
        media_input=dummy_image,
        scene_id="HeroTestScene",
        output_dir=out_dir
    )

    assert summary["status"] == "3d_world_generated"
    assert summary["perception_nodes_count"] > 0
    assert summary["is_physics_ready"] is True
    assert summary["is_photorealistic"] is True
    assert summary["navmesh_status"] == "generated"