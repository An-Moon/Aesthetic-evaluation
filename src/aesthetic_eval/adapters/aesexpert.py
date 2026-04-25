from typing import Any, List, Tuple

import os
import sys

import torch
from PIL import Image

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample


class AesExpertAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = str(self.model_cfg.get("device", "cuda:0"))
        self.conv_mode = str(self.model_cfg.get("conv_mode", "llava_v1"))
        self.max_new_tokens = int(self.base_cfg.get("generation", {}).get("max_new_tokens", 256))
        self.do_sample = bool(self.base_cfg.get("generation", {}).get("do_sample", False))

    def load(self) -> None:
        repo_root = str(self.model_cfg.get("llava_repo_root", "/home/Hu_xuanwei/LLaVA"))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model

        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._conv_templates = conv_templates
        self._SeparatorStyle = SeparatorStyle
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token

        model_path = str(self.model_cfg["model_path"])
        model_base = self.model_cfg.get("model_base")
        model_name = str(self.model_cfg.get("model_name_hint", get_model_name_from_path(model_path)))

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            load_8bit=bool(self.model_cfg.get("load_8bit", False)),
            load_4bit=bool(self.model_cfg.get("load_4bit", False)),
            device=self.device,
            device_map=self.model_cfg.get("device_map", {"": 0}),
        )
        self.tokenizer.padding_side = "left"
        self.model.eval()

    def build_prompt(self, sample: EvalSample) -> str:
        q = str(sample.question or "").strip()
        if q:
            return q
        return "Please describe the aesthetic experience of this image in detail."

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        valid: List[EvalSample] = []
        prompts: List[str] = []
        records = []

        for s in batch_samples:
            try:
                image = Image.open(s.image_resolved).convert("RGB")

                prompt_text = self.build_prompt(s)
                image_token = self._DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, "mm_use_im_start_end", False):
                    image_token = (
                        self._DEFAULT_IM_START_TOKEN
                        + self._DEFAULT_IMAGE_TOKEN
                        + self._DEFAULT_IM_END_TOKEN
                    )

                conv = self._conv_templates[self.conv_mode].copy()
                conv.append_message(conv.roles[0], image_token + "\n" + prompt_text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = self._tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    self._IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0)

                stop_str = conv.sep if conv.sep_style != self._SeparatorStyle.TWO else conv.sep2
                records.append(
                    {
                        "input_ids": input_ids,
                        "image": image,
                        "image_size": image.size,
                        "stop_str": stop_str,
                        "prompt_text": prompt_text,
                    }
                )
                valid.append(s)
                prompts.append(prompt_text)
            except Exception:
                continue

        if not valid:
            return None, [], []
        return {"records": records}, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        records = prepared["records"]
        if not records:
            return []
        outs: List[str] = []

        for rec in records:
            input_ids = rec["input_ids"].to(self.model.device)
            image_size = [rec["image_size"]]
            image_tensor = self._process_images(
                [rec["image"]],
                self.image_processor,
                self.model.config,
            ).to(self.model.device, dtype=torch.float16)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_size,
                    do_sample=self.do_sample,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                )

            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            text = text.strip()
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:")[-1].strip()
            stop_str = rec["stop_str"]
            if stop_str and stop_str in text:
                text = text.split(stop_str, 1)[0].strip()
            if stop_str and text.endswith(stop_str):
                text = text[: -len(stop_str)].strip()
            if text.startswith("."):
                text = text[1:].strip()
            if self._looks_garbled(text):
                text = ""
            outs.append(text)
        return outs

    @staticmethod
    def _looks_garbled(text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return True
        if len(s) <= 2:
            return True
        allowed = sum(ch.isalnum() or ch.isspace() or ch in ",.;:!?'-\"()[]{}" for ch in s)
        return (allowed / max(1, len(s))) < 0.35