import os
import re
import sys
from typing import Dict, List, Optional

import torch
from PIL import Image, ImageFile

from aesthetic_score_eval.adapters.base import BaseScoreAdapter
from aesthetic_score_eval.data import ScoreSample

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _extract_score(text: str, min_score: float, max_score: float) -> Optional[float]:
    lines = text.strip().split("\n")
    for line in reversed(lines):
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if matches:
            score = float(matches[-1])
            return float(max(min_score, min(max_score, score)))

    if "assistant" in text.lower():
        parts = re.split(r"assistant", text, flags=re.IGNORECASE)
        if parts:
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", parts[-1])
            if matches:
                score = float(matches[-1])
                return float(max(min_score, min(max_score, score)))

    return None


class AesExpertScoreAdapter(BaseScoreAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.prompt_text = str(
            base_cfg.get("prompt", {}).get(
                "score_prompt",
                "Rate the aesthetic quality of this image from 1 to 10. Only output a single float number.",
            )
        )
        self.min_score = float(base_cfg.get("prompt", {}).get("min_score", 1.0))
        self.max_score = float(base_cfg.get("prompt", {}).get("max_score", 10.0))
        self.score_mode = str(model_cfg.get("score_mode", "logits_wa5")).lower().strip()
        self.score_tokens = list(model_cfg.get("score_tokens", ["excellent", "good", "fair", "poor", "bad"]))
        self.score_weights = torch.tensor(
            [float(x) for x in model_cfg.get("score_weights", [10.0, 7.5, 5.0, 2.5, 0.0])],
            dtype=torch.float32,
        )
        self.logits_prompt = str(
            model_cfg.get(
                "logits_prompt",
                "How would you rate the aesthetics of this image?",
            )
        )
        self.logits_answer_prefix = str(
            model_cfg.get(
                "logits_answer_prefix",
                " The aesthetics of the image is",
            )
        )

    def _resolve_conv_mode(self, model_name: str) -> str:
        if self.model_cfg.get("conv_mode"):
            return str(self.model_cfg.get("conv_mode"))
        name = model_name.lower()
        if "llama-2" in name:
            return "llava_llama_2"
        if "mistral" in name:
            return "mistral_instruct"
        if "v1.6-34b" in name:
            return "chatml_direct"
        if "v1" in name:
            return "llava_v1"
        if "mpt" in name:
            return "mpt"
        return "llava_v0"

    def _build_query(self, query: str) -> str:
        image_token_se = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
        if self.IMAGE_PLACEHOLDER in query:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(self.IMAGE_PLACEHOLDER, image_token_se, query)
            else:
                qs = re.sub(self.IMAGE_PLACEHOLDER, self.DEFAULT_IMAGE_TOKEN, query)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + query
            else:
                qs = self.DEFAULT_IMAGE_TOKEN + "\n" + query
        return qs

    def load(self) -> None:
        llava_root = str(self.model_cfg.get("llava_repo_root"))
        if llava_root and llava_root not in sys.path:
            sys.path.insert(0, llava_root)

        from llava.utils import disable_torch_init
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            IMAGE_PLACEHOLDER,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.model.builder import load_pretrained_model

        disable_torch_init()

        model_path = str(self.model_cfg["model_path"])
        model_base = self.model_cfg.get("model_base")
        model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name
        )
        self.model.eval()

        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.conv_templates = conv_templates
        self.conv_mode = self._resolve_conv_mode(model_name)
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_PLACEHOLDER = IMAGE_PLACEHOLDER
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        tokenized = self.tokenizer(self.score_tokens)
        self.score_token_ids = [ids[1] if len(ids) > 1 else ids[0] for ids in tokenized["input_ids"]]

    def _make_prompt(self, query: str, add_answer_prefix: bool = False) -> str:
        qs = self._build_query(query)
        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if add_answer_prefix:
            prompt += self.logits_answer_prefix
        return prompt

    def _score_logits_wa5(self, image: Image.Image) -> Dict:
        prompt = self._make_prompt(self.logits_prompt, add_answer_prefix=True)
        image_tensors = self.process_images([image], self.image_processor, self.model.config)
        image_tensors = image_tensors.to(device=self.model.device, dtype=self.model.dtype)
        input_ids = (
            self.tokenizer_image_token(
                prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        with torch.inference_mode():
            output = self.model(input_ids=input_ids, images=image_tensors)

        logits = output["logits"][:, -1, self.score_token_ids][0].float().detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        raw = float((probs * self.score_weights).sum().item())
        response = {
            tok: {
                "logit": float(logit),
                "prob": float(prob),
            }
            for tok, logit, prob in zip(self.score_tokens, logits.tolist(), probs.tolist())
        }
        return {
            "raw_score": raw,
            "raw_response": response,
            "parse_status": "ok",
            "score_method": "fallback_logits_wa5",
            "official_alignment": "fallback_no_official_regression_script_found",
            "error": "",
        }

    def _score_prompt_numeric(self, image: Image.Image) -> Dict:
        prompt = self._make_prompt(self.prompt_text, add_answer_prefix=False)
        image_sizes = [image.size]
        image_tensors = self.process_images([image], self.image_processor, self.model.config)
        image_tensors = image_tensors.to(device=self.model.device, dtype=self.model.dtype)
        input_ids = (
            self.tokenizer_image_token(
                prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        gen_cfg = self.model_cfg.get("generation", {})
        temperature = float(gen_cfg.get("temperature", 0.2))
        top_p = gen_cfg.get("top_p")
        num_beams = int(gen_cfg.get("num_beams", 1))
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 16))

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True if temperature and temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        raw = _extract_score(decoded, self.min_score, self.max_score)
        if raw is None:
            return {"raw_score": None, "raw_response": decoded, "parse_status": "parse_failed", "error": ""}
        return {"raw_score": float(raw), "raw_response": decoded, "parse_status": "ok", "error": ""}

    def score_batch(self, batch_samples: List[ScoreSample]) -> List[Dict]:
        outputs: List[Dict] = []
        for sample in batch_samples:
            try:
                image = Image.open(sample.image_resolved).convert("RGB")
                if self.score_mode == "prompt_numeric":
                    outputs.append(self._score_prompt_numeric(image))
                else:
                    outputs.append(self._score_logits_wa5(image))
            except Exception as e:
                outputs.append({"raw_score": None, "raw_response": "", "parse_status": "error", "error": str(e)})

        return outputs
