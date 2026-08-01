"""VYOMAAV Sprint 40: Hero Mode — Master Image & Video to Interactive 3D World Engine."""

import os
import json
import cv2
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from PIL import Image

from somg.scene import SceneState
from somg.entity import SOMGEntity
from perception.depth_anything import DepthAnythingPredictor
from perception.sam2 import SAM2Predictor
from perception.grounding_dino import GroundingDINOPredictor
from perception.florence2 import Florence2Predictor
from perception.clip import CLIPPredictor
from reconstruction.camera import CameraPoseEstimator
from reconstruction.trellis import TRELLISReconstructionBackend
from reconstruction.hunyuan3d import Hunyuan3DReconstructionBackend
from reconstruction.fusion import GeometryFusionEngine
from reconstruction.gaussian import GaussianSplatEngine
from reconstruction.sdf import NeuralSDFEngine
from reconstruction.materials import PBRMaterialGenerator
from reconstruction.completion import WorldCompletionEngine
from runtime.interactive_world import InteractiveWorldManager


class WorldReconstructionPipeline:
    """Master pipeline: Accepts an RGB image or video clip and outputs a fully interactive 3D scene."""

    def __init__(self, device: Optional[str] = None):
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Initializing VYOMAAV Master Hero Pipeline on [{self.device.upper()}]...")

        # Initialize Phase 2 Perception Stack
        self.depth_p = DepthAnythingPredictor(device=self.device)
        self.sam2_p = SAM2Predictor(device=self.device)
        self.dino_p = GroundingDINOPredictor(device=self.device)
        self.florence_p = Florence2Predictor(device=self.device)
        self.clip_p = CLIPPredictor(device=self.device)

        # Initialize Phase 3 Reconstruction Engine
        self.camera_estimator = CameraPoseEstimator(device=self.device)
        self.trellis_backend = TRELLISReconstructionBackend(device=self.device)
        self.hunyuan_backend = Hunyuan3DReconstructionBackend(device=self.device)
        self.fusion_engine = GeometryFusionEngine()
        self.gaussian_engine = GaussianSplatEngine(device=self.device)
        self.sdf_engine = NeuralSDFEngine(device=self.device)
        self.pbr_generator = PBRMaterialGenerator(device=self.device)
        self.completion_engine = WorldCompletionEngine()

    def process_media_to_3d_world(
        self,
        media_input: Union[str, np.ndarray, Image.Image],
        scene_id: str = "ReconstructedWorld_0",
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes full end-to-end transformation from raw 2D input to an interactive 3D world."""
        scene = SceneState(scene_id=scene_id)
        if not hasattr(scene, "metadata"):
            scene.metadata = {}

        # 1. Video Frame Extraction or Image Prep
        if isinstance(media_input, str) and media_input.endswith((".mp4", ".avi", ".mov")):
            cap = cv2.VideoCapture(media_input)
            ret, frame = cap.read()
            cap.release()
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret else Image.new("RGB", (640, 480))
        elif isinstance(media_input, str) and os.path.exists(media_input):
            pil_img = Image.open(media_input).convert("RGB")
        elif isinstance(media_input, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(media_input, cv2.COLOR_BGR2RGB) if media_input.shape[2] == 3 else media_input)
        elif isinstance(media_input, Image.Image):
            pil_img = media_input.convert("RGB")
        else:
            pil_img = Image.new("RGB", (640, 480))

        img_np = np.array(pil_img)

        # 2. Camera Geometry Calibration (Sprint 31)
        cam_res = self.camera_estimator.estimate_camera_and_sparse_cloud([img_np])
        scene.metadata["camera_geometry"] = cam_res

        # 3. Phase 2 Neural Perception
        depth_res = self.depth_p.infer(img_np)
        sam2_res = self.sam2_p.infer(img_np)
        dino_res = self.dino_p.infer(img_np, text_prompt="object . chair . table")
        florence_res = self.florence_p.infer(img_np)
        clip_emb = self.clip_p.get_image_embedding(img_np)

        scene = self.dino_p.integrate_into_somg(scene, dino_res, depth_result=depth_res)
        scene = self.sam2_p.integrate_into_somg(scene, sam2_res, depth_result=depth_res)
        scene = self.florence_p.integrate_into_somg(scene, florence_res)

        # Get primary SOMG Node ID
        nodes = scene.resolve_active_graph().nodes
        target_entity_id = list(nodes.keys())[0] if len(nodes) > 0 else "default_entity"
        if target_entity_id not in nodes:
            nodes[target_entity_id] = SOMGEntity(target_entity_id)

        scene = self.clip_p.integrate_into_somg(scene, target_entity_id, clip_emb)

        # 4. Phase 3 Reconstruction Suite (TRELLIS + Hunyuan3D + Fusion)
        t_mesh = self.trellis_backend.reconstruct(pil_img)
        h_mesh = self.hunyuan_backend.reconstruct(pil_img)
        fused_mesh = self.fusion_engine.fuse_reconstruction_results(t_mesh, h_mesh)

        scene = self.fusion_engine.integrate_fused_mesh_into_somg(scene, target_entity_id, fused_mesh)

        # 5. Photorealistic Gaussians & Physics SDF Mesh
        splat_res = self.gaussian_engine.initialize_gaussians_from_mesh(fused_mesh)
        scene = self.gaussian_engine.integrate_gaussians_into_somg(scene, target_entity_id, splat_res)

        sdf_res = self.sdf_engine.compute_implicit_sdf_and_watertight_mesh(fused_mesh)
        scene = self.sdf_engine.integrate_sdf_into_somg(scene, target_entity_id, sdf_res)

        # 6. PBR Textures & World Completion
        pbr_res = self.pbr_generator.generate_pbr_maps(semantic_label="reconstructed_object")
        scene = self.pbr_generator.integrate_pbr_into_somg(scene, target_entity_id, pbr_res)

        completion_res = self.completion_engine.infer_unseen_geometry(fused_mesh)
        scene = self.completion_engine.integrate_completed_mesh_into_somg(scene, target_entity_id, completion_res)

        # 7. Interactive World Engine Initialization (Sprint 38)
        world_mgr = InteractiveWorldManager(scene)
        navmesh = world_mgr.generate_navmesh_nodes()

        pipeline_summary = {
            "scene_id": scene_id,
            "status": "3d_world_generated",
            "active_entity_id": target_entity_id,
            "perception_nodes_count": len(nodes),
            "camera_status": cam_res["status"],
            "fusion_backend": fused_mesh["fusion_type"],
            "is_physics_ready": sdf_res["physics_ready"],
            "is_photorealistic": splat_res["status"] == "splat_initialized",
            "navmesh_status": navmesh["navmesh_status"]
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            summary_path = os.path.join(output_dir, "world_reconstruction_summary.json")
            with open(summary_path, "w") as f:
                json.dump(pipeline_summary, f, indent=2)
            pipeline_summary["summary_path"] = summary_path

        return pipeline_summary
