from typing import Dict, List

import torch
from PIL import Image, ImageFile

from aesthetic_score_eval.adapters.base import BaseScoreAdapter
from aesthetic_score_eval.adapters.score_parse import extract_numeric_score
from aesthetic_score_eval.data import ScoreSample

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Qwen3VLPromptScoreAdapter(BaseScoreAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.processor = None
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
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_path = str(self.model_cfg["model_path"])
        dtype = self._resolve_dtype()
        trust_remote_code = bool(self.model_cfg.get("trust_remote_code", True))
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=self.model_cfg.get("device_map", {"": 0}),
            trust_remote_code=trust_remote_code,
        ).eval()

    def _device(self):
        return next(self.model.parameters()).device

    def score_batch(self, batch_samples: List[ScoreSample]) -> List[Dict]:
        max_new_tokens = int(self.model_cfg.get("generation", {}).get("max_new_tokens", 8))
        do_sample = bool(self.model_cfg.get("generation", {}).get("do_sample", False))
        texts = []
        images = []
        valid_indices = []
        outputs: List[Dict] = [
            {"raw_score": None, "raw_response": "", "parse_status": "error", "error": "not_processed"}
            for _ in batch_samples
        ]

        for idx, sample in enumerate(batch_samples):
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
                texts.append(text)
                images.append(image)
                valid_indices.append(idx)
            except Exception as e:
                outputs[idx] = {"raw_score": None, "raw_response": "", "parse_status": "error", "error": str(e)}

        if not valid_indices:
            return outputs

        try:
            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self._device())
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
            input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            trimmed = [out_ids[int(in_len):] for out_ids, in_len in zip(generated_ids, input_lens)]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
