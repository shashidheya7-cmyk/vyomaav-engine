"""VYOMAAV Perception: Florence-2 Dense Vision-Language Predictor."""

import os
import json
import torch
import numpy as np
import cv2
from typing import Dict, Any, Optional, Union
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel

# Patch 1: PretrainedConfig forced_bos_token_id fallback
if not hasattr(PretrainedConfig, "forced_bos_token_id"):
    PretrainedConfig.forced_bos_token_id = None

# Patch 2: PreTrainedTokenizerBase additional_special_tokens property descriptor
def _get_additional_special_tokens(self):
    if hasattr(self, "_additional_special_tokens") and isinstance(self._additional_special_tokens, (list, tuple)):
        return [str(t) for t in self._additional_special_tokens]
    special_map = getattr(self, "special_tokens_map", {})
    if isinstance(special_map, dict):
        tokens = special_map.get("additional_special_tokens", [])
        if isinstance(tokens, list):
            return [str(t) for t in tokens]
    return []

PreTrainedTokenizerBase.additional_special_tokens = property(_get_additional_special_tokens)

# Patch 3: PreTrainedModel _supports_sdpa fallback for custom remote code models
if not hasattr(PreTrainedModel, "_supports_sdpa"):
    PreTrainedModel._supports_sdpa = False


class Florence2Predictor:
    """Florence-2 Predictor for detailed captioning, object detection, and region tagging."""

    def __init__(self, model_id: str = "microsoft/Florence-2-base", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_id = model_id
        print(f"Loading Florence-2 ({self.model_id}) onto {self.device}...")

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device).eval()

    def _ensure_square(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Pads rectangular images to a square aspect ratio to prevent Florence-2 feature map assertion errors."""
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image)
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            return Image.new("RGB", (512, 512))

        w, h = pil_img.size
        if w == h:
            return pil_img

        max_dim = max(w, h)
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(pil_img, ((max_dim - w) // 2, (max_dim - h) // 2))
        return square_img

    @torch.no_grad()
    def infer(
        self,
        image: Union[np.ndarray, Image.Image],
        task_prompt: str = "<MORE_DETAILED_CAPTION>",
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs Florence-2 vision-language inference with automatic square padding."""
        pil_img = self._ensure_square(image)

        inputs = self.processor(text=task_prompt, images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if "pixel_values" in inputs and self.device.type == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            use_cache=False
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(pil_img.width, pil_img.height)
        )

        caption = parsed_answer.get(task_prompt, generated_text) if isinstance(parsed_answer, dict) else str(parsed_answer)

        res = {
            "task_prompt": task_prompt,
            "caption": caption if isinstance(caption, str) else str(caption),
            "raw_output": str(parsed_answer),
            "status": "success"
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_file = os.path.join(output_dir, "florence2_caption.json")
            with open(out_file, "w") as f:
                json.dump({"caption": str(caption)}, f, indent=2)
            res["json_path"] = out_file

        return res

    def integrate_into_somg(self, scene: Any, florence_result: Dict[str, Any]) -> Any:
        """Attaches Florence-2 textual caption metadata to the active SOMG scene graph."""
        if hasattr(scene, "metadata"):
            scene.metadata["dense_caption"] = florence_result.get("caption", "")
        return scene
