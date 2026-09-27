from typing import Dict, List

import torch
from PIL import Image, ImageFile

from aesthetic_score_eval.adapters.base import BaseScoreAdapter
from aesthetic_score_eval.adapters.score_parse import extract_numeric_score
from aesthetic_score_eval.data import ScoreSample

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LLaVAOneVisionPromptScoreAdapter(BaseScoreAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = str(model_cfg.get("device", "cuda"))
        self.prompt_text = str(
            base_cfg.get("prompt", {}).get(
                "score_prompt",
                "Rate the aesthetic quality of this image from 1 to 10. Only output a single float number.",
            )
        )
        self.min_score = float(base_cfg.get("prompt", {}).get("min_score", 1.0))
        self.max_score = float(base_cfg.get("prompt", {}).get("max_score", 10.0))

    def _resolve_dtype(self):
        dtype_name = str(self.model_cfg.get("torch_dtype", "float16")).lower().strip()
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float32":
            return torch.float32
        return torch.float16

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        self._require_modern_transformers_for_onevision()
        self._patch_transformers_compat()

        model_path = str(self.model_cfg["model_path"])
        dtype = self._resolve_dtype()
        trust_remote_code = bool(self.model_cfg.get("trust_remote_code", True))
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        device_map = self.model_cfg.get("device_map", "auto")
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if device_map is None:
            self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=trust_remote_code,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            tokenizer=self.tokenizer,
            trust_remote_code=trust_remote_code,
            use_fast=bool(self.model_cfg.get("use_fast", True)),
        )

    def _require_modern_transformers_for_onevision(self) -> None:
        import transformers

        def _version_tuple(version_text: str):
            parts = []
            for item in version_text.split(".")[:3]:
                digits = "".join(ch for ch in item if ch.isdigit())
                parts.append(int(digits or 0))
            while len(parts) < 3:
                parts.append(0)
            return tuple(parts)

        current = _version_tuple(getattr(transformers, "__version__", "0.0.0"))
        if current < (4, 57, 0):
            raise RuntimeError(
                "LLaVA-OneVision-1.5 requires a newer Transformers runtime than this environment provides "
                f"(found transformers {transformers.__version__}). Use "
                "a Transformers >=4.57 runtime for this adapter. "
                "The older llava env fails in remote-code imports such as SlidingWindowCache/modeling_rope_utils."
            )

    def _patch_transformers_compat(self) -> None:
        try:
            import sys
            import types
            import transformers.configuration_utils as configuration_utils

            if not hasattr(configuration_utils, "layer_type_validation"):
                configuration_utils.layer_type_validation = lambda *_args, **_kwargs: None
            try:
                import transformers.modeling_rope_utils  # noqa: F401
            except ImportError:
                rope_utils = types.ModuleType("transformers.modeling_rope_utils")
                rope_utils.rope_config_validation = lambda *_args, **_kwargs: None
                sys.modules["transformers.modeling_rope_utils"] = rope_utils
        except Exception:
            pass

    def score_batch(self, batch_samples: List[ScoreSample]) -> List[Dict]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None

        max_new_tokens = int(self.model_cfg.get("generation", {}).get("max_new_tokens", 8))
        do_sample = bool(self.model_cfg.get("generation", {}).get("do_sample", False))
        messages = []
        valid_indices = []
        outputs: List[Dict] = [
            {"raw_score": None, "raw_response": "", "parse_status": "error", "error": "not_processed"}
            for _ in batch_samples
        ]

        for idx, sample in enumerate(batch_samples):
            try:
                image = Image.open(sample.image_resolved).convert("RGB")
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": self.prompt_text},
                        ],
                    }
                )
                valid_indices.append(idx)
            except Exception as e:
                outputs[idx] = {"raw_score": None, "raw_response": "", "parse_status": "error", "error": str(e)}

        if not valid_indices:
            return outputs

        try:
            texts = [self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages]
            if process_vision_info is None:
                image_inputs = [msg["content"][0]["image"] for msg in messages]
                video_inputs = None
            else:
                image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            decoded = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for out_idx, text in zip(valid_indices, decoded):
                score = extract_numeric_score(text, min_score=self.min_score, max_score=self.max_score)
                if score is None:
                    outputs[out_idx] = {"raw_score": None, "raw_response": text, "parse_status": "parse_failed", "error": ""}
                else:
                    outputs[out_idx] = {"raw_score": float(score), "raw_response": text, "parse_status": "ok", "error": ""}
        except Exception as e:
            for out_idx in valid_indices:
                outputs[out_idx] = {"raw_score": None, "raw_response": "", "parse_status": "error", "error": str(e)}

        return outputs
