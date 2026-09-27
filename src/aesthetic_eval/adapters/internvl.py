from typing import Any, List, Tuple

import torch
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample, load_and_resize_rgb


def _patch_transformers_tied_weight_compat() -> None:
    """Allow older remote-code models to load under transformers 5.x."""
    import transformers.modeling_utils as modeling_utils

    original = modeling_utils.get_total_byte_count
    if getattr(original, "_aesthetic_eval_tied_weight_compat", False):
        return

    def compat_get_total_byte_count(model, accelerator_device_map, hf_quantizer=None):
        if not hasattr(model, "all_tied_weights_keys"):
            try:
                model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(all_submodels=True)
            except Exception:
                model.all_tied_weights_keys = {}
        return original(model, accelerator_device_map, hf_quantizer)

    compat_get_total_byte_count._aesthetic_eval_tied_weight_compat = True
    modeling_utils.get_total_byte_count = compat_get_total_byte_count


class InternVLAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.tokenizer = None
        self.to_tensor = transforms.ToTensor()
        self.infer_dtype = torch.float16

    def load(self) -> None:
        model_path = self.model_cfg["model_path"]
        dtype_name = str(self.model_cfg.get("torch_dtype", "float16"))
        torch_dtype = getattr(torch, dtype_name)
        self.infer_dtype = torch_dtype
        device_map = self.model_cfg.get("device_map", {"": 0})

        trust_remote_code = bool(self.model_cfg.get("trust_remote_code", True))
        use_fast = bool(self.model_cfg.get("use_fast_tokenizer", True))

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
            fix_mistral_regex=True,
        )

        _patch_transformers_tied_weight_compat()
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=bool(self.model_cfg.get("trust_remote_code", True)),
        ).eval()
        if not hasattr(self.model, "all_tied_weights_keys"):
            self.model.all_tied_weights_keys = {}

    def build_prompt(self, sample: EvalSample) -> str:
        prompt = self.format_prompt(sample, "Please describe the aesthetic quality of this image.")
        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt
        return prompt

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        image_size = int(
            self.model_cfg.get(
                "image_size", self.base_cfg.get("preprocess", {}).get("image_size", 448)
            )
        )
        required_image_size = int(
            getattr(self.model.config, "force_image_size", None)
            or self.model.config.vision_config.image_size
        )
        if image_size != required_image_size:
            raise RuntimeError(
                f"InternVL image_size={image_size} does not match checkpoint-required "
                f"image_size={required_image_size}; refusing a visual-token mismatch"
            )
        valid: List[EvalSample] = []
        tensors = []
        prompts = []

        for s in batch_samples:
            try:
                img = load_and_resize_rgb(s.image_resolved, image_size)
            except Exception as exc:
                raise RuntimeError(
                    f"InternVL image preprocessing failed for sample_id={s.sample_id!r} "
                    f"image={s.image_resolved!r}"
                ) from exc
            t = self.to_tensor(img)
            tensors.append(t)
            valid.append(s)
            prompts.append(self.build_prompt(s))

        if not valid:
            return None, [], []

        stacked = torch.stack(tensors)
        return stacked, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        stacked, prompts = prepared
        device = next(self.model.parameters()).device
        stacked = stacked.to(device=device, dtype=self.infer_dtype, non_blocking=True)

        gen_cfg = self.base_cfg.get("generation", {})
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
        do_sample = bool(gen_cfg.get("do_sample", False))
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if not hasattr(self.model, "batch_chat"):
            raise RuntimeError("InternVL checkpoint does not expose the required batch_chat API")

        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=self.infer_dtype):
                batch_outs = self.model.batch_chat(
                    self.tokenizer,
                    pixel_values=stacked,
                    questions=prompts,
                    generation_config=generation_config,
                    num_patches_list=[1] * len(prompts),
                )

        cleaned = [str(x).strip() for x in batch_outs]
        if len(cleaned) != len(prompts):
            raise RuntimeError(
                f"InternVL output coverage mismatch: outputs={len(cleaned)} prompts={len(prompts)}"
            )
        empty = [i for i, text in enumerate(cleaned) if not text]
        if empty:
            raise RuntimeError(f"InternVL returned empty outputs at batch positions {empty}")
        return cleaned
