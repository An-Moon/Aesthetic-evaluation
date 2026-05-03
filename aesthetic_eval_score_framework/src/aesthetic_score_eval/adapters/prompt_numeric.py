from typing import Dict, List, Optional

import torch
from PIL import Image, ImageFile
from transformers import AutoModelForCausalLM, AutoProcessor

from aesthetic_score_eval.adapters.base import BaseScoreAdapter
from aesthetic_score_eval.data import ScoreSample
from aesthetic_score_eval.adapters.score_parse import extract_numeric_score

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _extract_score(text: str, min_score: float, max_score: float) -> Optional[float]:
    return extract_numeric_score(text, min_score=min_score, max_score=max_score)


class PromptNumericScoreAdapter(BaseScoreAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.processor = None
        self.min_score = float(base_cfg.get("prompt", {}).get("min_score", 1.0))
        self.max_score = float(base_cfg.get("prompt", {}).get("max_score", 10.0))
        self.prompt_text = str(
            base_cfg.get("prompt", {}).get(
                "score_prompt",
                "Rate the aesthetic quality of this image from 1 to 10. Only output a single float number.",
            )
        )

    def _resolve_dtype(self):
        dtype_name = str(self.model_cfg.get("torch_dtype", "bfloat16")).lower().strip()
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "float32":
            return torch.float32
        return torch.bfloat16

    def load(self) -> None:
        model_path = str(self.model_cfg["model_path"])
        trust_remote_code = bool(self.model_cfg.get("trust_remote_code", True))
        device_map = self.model_cfg.get("device_map", "auto")
        dtype = self._resolve_dtype()

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

    def score_batch(self, batch_samples: List[ScoreSample]) -> List[Dict]:
        outputs: List[Dict] = []
        max_new_tokens = int(self.model_cfg.get("generation", {}).get("max_new_tokens", 8))
        do_sample = bool(self.model_cfg.get("generation", {}).get("do_sample", False))

        for sample in batch_samples:
            try:
                image = Image.open(sample.image_resolved).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": self.prompt_text},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                    )

                generated_ids_trimmed = [generated_ids[0][len(inputs["input_ids"][0]):]]
                decoded = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                raw = _extract_score(decoded[0], self.min_score, self.max_score)
                if raw is None:
                    outputs.append({"raw_score": None, "raw_response": decoded[0], "parse_status": "parse_failed", "error": ""})
                else:
                    outputs.append({"raw_score": float(raw), "raw_response": decoded[0], "parse_status": "ok", "error": ""})
            except Exception as e:
                outputs.append({"raw_score": None, "raw_response": "", "parse_status": "error", "error": str(e)})

        return outputs
