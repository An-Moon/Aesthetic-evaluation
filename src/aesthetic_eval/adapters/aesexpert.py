from typing import Any, List, Tuple

import os
import re
import sys

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from aesthetic_eval.adapters.base import BaseAdapter, require_directory
from aesthetic_eval.data import EvalSample


def _patch_official_clip_vision_tower(hf_name: str, local_path: str) -> None:
    """Resolve the checkpoint-declared CLIP tower to one pinned local snapshot."""
    if not local_path or not os.path.isdir(local_path):
        raise FileNotFoundError(
            f"AesExpert official vision tower snapshot is missing: {local_path!r}"
        )
    required = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
    missing = [name for name in required if not os.path.isfile(os.path.join(local_path, name))]
    if missing:
        raise FileNotFoundError(
            f"AesExpert official vision tower snapshot is incomplete at {local_path}: missing={missing}"
        )

    for cls in (CLIPVisionConfig, CLIPImageProcessor, CLIPVisionModel):
        if getattr(cls, "_aesexpert_local_patch", False):
            continue
        original = cls.from_pretrained.__func__

        def wrapped(inner_cls, pretrained_model_name_or_path, *args, _original=original, **kwargs):
            if str(pretrained_model_name_or_path) == hf_name:
                pretrained_model_name_or_path = local_path
                kwargs["local_files_only"] = True
            return _original(inner_cls, pretrained_model_name_or_path, *args, **kwargs)

        cls.from_pretrained = classmethod(wrapped)
        cls._aesexpert_local_patch = True


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
        self.temperature = self.base_cfg.get("generation", {}).get("temperature")
        self.top_p = self.base_cfg.get("generation", {}).get("top_p")
        self.num_beams = int(self.base_cfg.get("generation", {}).get("num_beams", 1))

    def load(self) -> None:
        repo_root = require_directory(self.model_cfg, "llava_repo_root", component="AesExpert")
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
        vision_tower_name = str(
            self.model_cfg.get("vision_tower_name", "openai/clip-vit-large-patch14-336")
        )
        vision_tower_local_path = str(self.model_cfg.get("vision_tower_local_path", ""))
        _patch_official_clip_vision_tower(vision_tower_name, vision_tower_local_path)

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            load_8bit=bool(self.model_cfg.get("load_8bit", False)),
            load_4bit=bool(self.model_cfg.get("load_4bit", False)),
            device=self.device,
            device_map=self.model_cfg.get("device_map", {"": 0}),
            use_flash_attn=bool(self.model_cfg.get("use_flash_attn", False)),
        )
        self.context_len = int(
            getattr(
                self.model.config,
                "max_sequence_length",
                getattr(self.model.config, "max_position_embeddings", 2048),
            )
        )
        self.tokenizer.model_max_length = self.context_len
        vision_tower = self.model.get_vision_tower()
        self.image_token_count = int(getattr(vision_tower, "num_patches", 576))
        self.tokenizer.padding_side = "left"
        self.model.eval()

    def build_prompt(self, sample: EvalSample) -> str:
        return self.format_prompt(sample, "Please describe the aesthetic experience of this image in detail.")

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
                effective_tokens = int(input_ids.shape[1]) + self.image_token_count - 1 + self.max_new_tokens
                if effective_tokens > self.context_len:
                    raise RuntimeError(
                        f"AesExpert context overflow for sample_id={s.sample_id}: "
                        f"text_tokens={input_ids.shape[1]} image_tokens={self.image_token_count} "
                        f"max_new_tokens={self.max_new_tokens} limit={self.context_len}"
                    )

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
            except Exception as exc:
                if bool(self.base_cfg.get("runtime", {}).get("strict_inference", False)):
                    raise RuntimeError(
                        f"AesExpert preprocessing failed for sample_id={s.sample_id} image={s.image_resolved}"
                    ) from exc
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
                generation_kwargs = {
                    "images": image_tensor,
                    "image_sizes": image_size,
                    "do_sample": self.do_sample,
                    "num_beams": self.num_beams,
                    "max_new_tokens": self.max_new_tokens,
                    "use_cache": True,
                }
                if self.do_sample:
                    if self.temperature is not None:
                        generation_kwargs["temperature"] = float(self.temperature)
                    if self.top_p is not None:
                        generation_kwargs["top_p"] = float(self.top_p)
                output_ids = self.model.generate(input_ids, **generation_kwargs)

            if (
                output_ids.shape[1] >= input_ids.shape[1]
                and torch.equal(output_ids[:, : input_ids.shape[1]], input_ids)
            ):
                decode_ids = output_ids[:, input_ids.shape[1] :]
            else:
                decode_ids = output_ids

            text = self.tokenizer.decode(decode_ids[0], skip_special_tokens=True)
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
            text = re.sub(r"^(?:reference\s+answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
            text = self._normalize_choice_answer(text, rec.get("prompt_text", ""))
            if self._looks_garbled(text):
                raise RuntimeError("AesExpert generated empty or garbled text")
            outs.append(text)
        return outs

    @staticmethod
    def _normalize_choice_answer(text: str, prompt: str) -> str:
        s = (text or "").strip()
        if not s:
            return s

        choices = {}
        for label, value in re.findall(r"(?m)^\s*([A-Z])\.\s*(.+?)\s*$", prompt or ""):
            choices[label.upper()] = value.strip()

        if not choices:
            return s

        compact = s.strip().strip(" .,:;!?()[]{}").upper()
        if compact in choices:
            return choices[compact]

        match = re.match(r"^\s*([A-Z])[\.\):：、\s]", s)
        if match and match.group(1).upper() in choices:
            return choices[match.group(1).upper()]

        return s

    @staticmethod
    def _looks_garbled(text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return True
        allowed = sum(ch.isalnum() or ch.isspace() or ch in ",.;:!?'-\"()[]{}" for ch in s)
        return (allowed / max(1, len(s))) < 0.35
