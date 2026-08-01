"""VYOMAAV CLI Demo: Transform any Image or Video into a 3D World."""

import sys
import os
import numpy as np
from reconstruction.hero_pipeline import WorldReconstructionPipeline

def main():
    media_path = sys.argv[1] if len(sys.argv) > 1 else None

    if media_path and os.path.exists(media_path):
        print(f"Loading input media: {media_path}")
        input_data = media_path
    else:
        print("No image/video provided. Creating square synthetic test frame (512x512)...")
        input_data = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    pipeline = WorldReconstructionPipeline()
    summary = pipeline.process_media_to_3d_world(
        media_input=input_data,
        scene_id="LiveDemoWorld",
        output_dir="./reconstructed_world_output"
    )

    print("\n" + "=" * 60)
    print("VYOMAAV 3D WORLD RECONSTRUCTION COMPLETE!")
    print("=" * 60)
    for k, v in summary.items():
        print(f" - {k}: {v}")
    print("=" * 60)

if __name__ == "__main__":
    main()
