"""VYOMAAV Universal Photo-to-3D World Reconstruction Engine (SOMG Architecture)."""

import os
import torch
import trimesh
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Union

# Perception & Geometry Modules
from perception.florence2 import Florence2Predictor
from perception.grounding_dino import GroundingDINOPredictor
from perception.sam2 import SAM2Predictor
from perception.depth_anything import DepthAnythingV2Predictor
from reconstruction.trellis import TRELLISReconstructionBackend
from somg.scene_graph import SpatialObjectMeshGraph


class WorldReconstructionPipeline:
    """Universal pipeline that reconstructs full 3D multi-object room environments from ANY input photo."""

    def __init__(self, device: Optional[str] = None):
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Initializing Universal VYOMAAV World Engine on [{self.device.upper()}]...")

        self.depth_engine = DepthAnythingV2Predictor(device=self.device)
        self.sam2 = SAM2Predictor(device=self.device)
        self.dino = GroundingDINOPredictor(device=self.device)
        self.florence = Florence2Predictor(device=self.device)
        self.trellis = TRELLISReconstructionBackend(device=self.device)

    def _estimate_3d_spatial_box(
        self, mask: np.ndarray, depth_map: np.ndarray, img_shape: tuple
    ) -> Dict[str, np.ndarray]:
        """Maps 2D pixel coordinates and depth values into 3D world positions [X, Y, Z, S_x, S_y, S_z]."""
        h, w = img_shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return {"position": np.zeros(3), "scale": np.ones(3)}

        entity_depths = depth_map[ys, xs]
        z_center = float(np.median(entity_depths))
        
        x_center = (np.mean(xs) - w / 2.0) / w * z_center
        y_center = -(np.mean(ys) - h / 2.0) / h * z_center

        x_span = max((np.max(xs) - np.min(xs)) / w * z_center, 0.2)
        y_span = max((np.max(ys) - np.min(ys)) / h * z_center, 0.2)
        z_span = max(float(np.percentile(entity_depths, 90) - np.percentile(entity_depths, 10)), 0.2)

        return {
            "position": np.array([x_center, y_center, z_center], dtype=np.float32),
            "scale": np.array([x_span, y_span, z_span], dtype=np.float32)
        }

    def _export_clean_gaussians_ply(self, vertices: np.ndarray, colors: np.ndarray, output_path: str) -> None:
        """Exports room vertices with RGB colors as a clean PLY point cloud."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        point_cloud = trimesh.points.PointCloud(vertices, colors=colors)
        point_cloud.export(output_path)

    def reconstruct_world_from_photo(
        self, image: Image.Image, scene_id: str = "UniversalScene", output_dir: str = "./real_world_output"
    ) -> Dict[str, Any]:
        """Full zero-shot reconstruction of any input scene photo."""
        os.makedirs(output_dir, exist_ok=True)
        img_np = np.array(image.convert("RGB"))
        h, w = img_np.shape[:2]

        print(f"\n1. Estimating Global Room Depth & Layout...")
        depth_map = self.depth_engine.predict_depth(image)

        print(f"2. Parsing Open-World Objects via Vision-Language Perception...")
        detection_prompt = "furniture, table, chair, decor, lamp, wall art, appliance, object, floor"
        detections = self.dino.predict(image, text_prompt=detection_prompt, box_threshold=0.20)

        if not detections or len(detections.get("boxes", [])) == 0:
            boxes = [[0, 0, w, h]]
            labels = ["scene_object"]
        else:
            boxes = detections["boxes"]
            labels = detections.get("labels", [f"entity_{i}" for i in range(len(boxes))])

        print(f"   Detected {len(boxes)} dynamic spatial entities.")

        print(f"3. Generating Segmentations & 3D Spatial Transforms...")
        masks = self.sam2.segment_boxes(image, boxes)

        somg = SpatialObjectMeshGraph(scene_id=scene_id)
        combined_meshes = []

        print(f"4. Reconstructing High-Clarity 3D Meshes & Placing in World Coordinates...")
        for idx, (box, label, mask) in enumerate(zip(boxes, labels, masks)):
            entity_id = f"{label}_{idx}"
            print(f"   Processing entity [{entity_id}]...")

            x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            cropped_img = image.crop((x1, y1, x2, y2)) if (x2 > x1 and y2 > y1) else image

            spatial_box = self._estimate_3d_spatial_box(mask, depth_map, (h, w))
            pos = spatial_box["position"]
            scale = spatial_box["scale"]

            mesh_data = self.trellis.reconstruct(cropped_img)
            verts = np.array(mesh_data["vertices"], dtype=np.float32)
            faces = np.array(mesh_data["faces"], dtype=np.int32)
            colors = np.array(mesh_data.get("colors", []), dtype=np.uint8)

            if len(verts) > 0 and len(faces) > 0:
                entity_mesh = trimesh.Trimesh(
                    vertices=verts,
                    faces=faces,
                    vertex_colors=colors if len(colors) == len(verts) else None,
                    process=True
                )

                entity_mesh.vertices -= entity_mesh.center_mass
                max_extent = np.max(entity_mesh.extents)
                if max_extent > 0:
                    entity_mesh.vertices /= max_extent

                entity_mesh.vertices *= scale
                entity_mesh.vertices += pos

                combined_meshes.append(entity_mesh)

            somg.add_entity_node(
                entity_id=entity_id,
                label=label,
                position=pos.tolist(),
                scale=scale.tolist(),
                mesh_status=mesh_data.get("status", "completed")
            )

        print(f"5. Assembling Full Composite 3D World Scene...")
        if combined_meshes:
            world_mesh = trimesh.util.concatenate(combined_meshes)
        else:
            world_mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

        world_mesh.update_faces(world_mesh.nondegenerate_faces())
        world_mesh.update_faces(world_mesh.unique_faces())
        world_mesh.remove_infinite_values()
        world_mesh.remove_unreferenced_vertices()

        obj_out = os.path.join(output_dir, "fused_mesh.obj")
        glb_out = os.path.join(output_dir, "fused_world.glb")
        ply_out = os.path.join(output_dir, "gaussians.ply")
        
        world_mesh.export(obj_out)
        world_mesh.export(glb_out)
        self._export_clean_gaussians_ply(world_mesh.vertices, world_mesh.visual.vertex_colors, ply_out)

        summary_path = os.path.join(output_dir, "world_reconstruction_summary.json")
        somg.export_summary(summary_path)

        print(f"\n World Reconstruction Complete!")
        print(f" 📄 Exported High-Clarity 3D Mesh (OBJ): {obj_out}")
        print(f" 📄 Exported High-Clarity 3D Asset (GLB): {glb_out}")
        print(f" 📄 Exported Room Gaussians (PLY):     {ply_out}")
        print(f" 📄 Exported SOMG Scene Summary:       {summary_path}")

        return {
            "scene_id": scene_id,
            "entity_count": len(combined_meshes),
            "total_vertices": len(world_mesh.vertices),
            "total_faces": len(world_mesh.faces),
            "exported_obj": obj_out,
            "exported_glb": glb_out,
            "exported_ply": ply_out,
            "summary_path": summary_path
        }

    def process_media_to_3d_world(
        self,
        media_input: Optional[Union[str, Image.Image]] = None,
        image_input: Optional[Union[str, Image.Image]] = None,
        scene_id: str = "RealWorldScene",
        output_dir: str = "./real_world_output",
        **kwargs
    ) -> Dict[str, Any]:
        raw_input = media_input if media_input is not None else image_input
        if raw_input is None:
            raw_input = kwargs.get("image") or kwargs.get("photo")

        if raw_input is None:
            raise ValueError("No valid image/media input passed to process_media_to_3d_world.")

        if isinstance(raw_input, str):
            if not os.path.exists(raw_input):
                raise FileNotFoundError(f"Input image file not found: {raw_input}")
            image = Image.open(raw_input).convert("RGB")
        else:
            image = raw_input.convert("RGB")

        return self.reconstruct_world_from_photo(image=image, scene_id=scene_id, output_dir=output_dir)
