from typing import Dict, List

import torch
from PIL import Image, ImageFile

from aesthetic_score_eval.adapters.base import BaseScoreAdapter
from aesthetic_score_eval.adapters.score_parse import extract_numeric_score
from aesthetic_score_eval.data import ScoreSample

ImageFile.LOAD_TRUNCATED_IMAGES = True


class InternVLPromptScoreAdapter(BaseScoreAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = str(model_cfg.get("device", "cuda:0"))
        self.prompt_text = str(
            base_cfg.get("prompt", {}).get(
                "score_prompt",
                "Rate the aesthetic quality of this image from 1 to 10. Only output a single float number.",
            )
        )
        self.min_score = float(base_cfg.get("prompt", {}).get("min_score", 1.0))
        self.max_score = float(base_cfg.get("prompt", {}).get("max_score", 10.0))

    def _resolve_dtype(self):
        dtype_name = str(self.model_cfg.get("torch_dtype", "bfloat16")).lower().strip()
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "float32":
            return torch.float32
        return torch.bfloat16

    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
        try:
            import torch.nn as nn

            if not hasattr(nn.Module, "all_tied_weights_keys"):
                def _get_all_tied_weights_keys(module):
                    return getattr(module, "_all_tied_weights_keys_compat", {})

                def _set_all_tied_weights_keys(module, value):
                    module.__dict__["_all_tied_weights_keys_compat"] = value

                nn.Module.all_tied_weights_keys = property(_get_all_tied_weights_keys, _set_all_tied_weights_keys)
        except Exception:
            pass

        model_path = str(self.model_cfg["model_path"])
        dtype = self._resolve_dtype()
        trust_remote_code = bool(self.model_cfg.get("trust_remote_code", True))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            fix_mistral_regex=True,
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)
        kwargs = {
            "torch_dtype": dtype,
            "device_map": self.model_cfg.get("device_map", {"": 0}),
            "trust_remote_code": trust_remote_code,
        }
        attn_impl = self.model_cfg.get("attn_implementation")
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        self.model = AutoModel.from_pretrained(model_path, **kwargs).eval()

    def score_batch(self, batch_samples: List[ScoreSample]) -> List[Dict]:
        outputs: List[Dict] = []
        dtype = self._resolve_dtype()
        gen_cfg = {
            "max_new_tokens": int(self.model_cfg.get("generation", {}).get("max_new_tokens", 16)),
            "do_sample": bool(self.model_cfg.get("generation", {}).get("do_sample", False)),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        query = "<image>\n" + self.prompt_text

        for sample in batch_samples:
            try:
                image = Image.open(sample.image_resolved).convert("RGB")
                pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].to(
                    self.device, dtype=dtype
                )
                with torch.inference_mode():
                    response = self.model.chat(self.tokenizer, pixel_values, query, dict(gen_cfg), history=[])
                score = extract_numeric_score(response, min_score=self.min_score, max_score=self.max_score)
                if score is None:
                    outputs.append({"raw_score": None, "raw_response": response, "parse_status": "parse_failed", "error": ""})
                else:
                    outputs.append({"raw_score": float(score), "raw_response": response, "parse_status": "ok", "error": ""})
            except Exception as e:
                outputs.append({"raw_score": None, "raw_response": "", "parse_status": "error", "error": str(e)})

        return outputs
