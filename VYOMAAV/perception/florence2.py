"""VYOMAAV Perception: Florence-2 Vision-Language Model Integration."""
import os, glob, json, torch, numpy as np, cv2
from typing import Dict, Any, Optional, Union
from PIL import Image

import transformers.configuration_utils
_orig_config_getattr = transformers.configuration_utils.PretrainedConfig.__getattribute__
def _safe_config_getattr(self, key):
    try: return _orig_config_getattr(self, key)
    except AttributeError:
        if key in ("forced_bos_token_id", "forced_eos_token_id") or key in getattr(self, "attribute_map", {}): return None
        raise
transformers.configuration_utils.PretrainedConfig.__getattribute__ = _safe_config_getattr

import transformers.tokenization_utils_base
_orig_tok_getattr = transformers.tokenization_utils_base.PreTrainedTokenizerBase.__getattr__
def _safe_tok_getattr(self, key):
    if key == "additional_special_tokens": return getattr(self, "_additional_special_tokens", getattr(self, "extra_special_tokens", []))
    try: return _orig_tok_getattr(self, key)
    except AttributeError:
        if key in ("forced_bos_token_id", "forced_eos_token_id"): return None
        raise
transformers.tokenization_utils_base.PreTrainedTokenizerBase.__getattr__ = _safe_tok_getattr

import transformers.modeling_utils
if not hasattr(transformers.modeling_utils.PreTrainedModel, "_supports_sdpa"):
    transformers.modeling_utils.PreTrainedModel._supports_sdpa = False
if not hasattr(transformers.modeling_utils.PreTrainedModel, "_supports_flash_attn_2"):
    transformers.modeling_utils.PreTrainedModel._supports_flash_attn_2 = False
if not hasattr(transformers.modeling_utils.PreTrainedModel, "_supports_flex_attn"):
    transformers.modeling_utils.PreTrainedModel._supports_flex_attn = False

try:
    import transformers.cache_utils
    if hasattr(transformers.cache_utils, "EncoderDecoderCache"):
        if not hasattr(transformers.cache_utils.EncoderDecoderCache, "__getitem__"):
            transformers.cache_utils.EncoderDecoderCache.__getitem__ = lambda self, idx: self.self_attention_cache[idx]
    if hasattr(transformers.cache_utils, "DynamicCache"):
        if not hasattr(transformers.cache_utils.DynamicCache, "__getitem__"):
            transformers.cache_utils.DynamicCache.__getitem__ = lambda self, idx: (self.key_cache[idx], self.value_cache[idx])
except Exception:
    pass

from transformers import AutoProcessor, AutoModelForCausalLM

class Florence2Predictor:
    def __init__(self, model_id: str = "microsoft/Florence-2-base", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading Florence-2 ({model_id}) onto {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, attn_implementation="eager", torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, image: Union[str, np.ndarray, torch.Tensor], task_prompt: str = "<MORE_DETAILED_CAPTION>", output_dir: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(image, str) and os.path.exists(image): pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray): pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image)
        else: pil_img = Image.new("RGB", (640, 480))
        inputs = self.processor(text=task_prompt, images=pil_img, return_tensors="pt").to(self.device)
        if self.device.type == "cuda": inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3, do_sample=False, use_cache=False)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_output = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=pil_img.size)
        caption = parsed_output.get(task_prompt, str(parsed_output)) if isinstance(parsed_output, dict) else str(parsed_output)
        return {"caption": caption, "task_prompt": task_prompt, "parsed_output": parsed_output, "resolution": pil_img.size}

    def integrate_into_somg(self, scene: Any, florence_result: Dict[str, Any]) -> Any:
        if hasattr(scene, "scene_caption"):
            scene.scene_caption = florence_result.get("caption", "")
        return scene
