"""VYOMAAV Real-World Image End-to-End Validation Script."""

import os
import urllib.request
import numpy as np
from PIL import Image
from reconstruction.hero_pipeline import WorldReconstructionPipeline
from reconstruction.exporter import MeshExporter

def download_sample_image(url: str, save_path: str) -> str:
    print(f"Downloading real-world test image from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
        out_file.write(response.read())
    return save_path

def main():
    img_url = "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=600&q=80" # Chair/Interior photo
    local_img_path = "real_chair_test.jpg"
    out_dir = "./real_world_output"

    if not os.path.exists(local_img_path):
        download_sample_image(img_url, local_img_path)

    print("\n--- Executing VYOMAAV Hero Pipeline on Real RGB Photo ---")
    pipeline = WorldReconstructionPipeline()
    summary = pipeline.process_media_to_3d_world(
        media_input=local_img_path,
        scene_id="RealWorldChairScene",
        output_dir=out_dir
    )

    print("\n" + "=" * 60)
    print("REAL IMAGE RECONSTRUCTION AUDIT COMPLETE!")
    print("=" * 60)
    for k, v in summary.items():
        print(f" - {k}: {v}")
    print("=" * 60)

    # Check generated files on disk
    print("\nGenerated Production Assets:")
    for root, _, files in os.walk(out_dir):
        for f in files:
            print(f" 📄 {os.path.join(root, f)}")

if __name__ == "__main__":
    main()
