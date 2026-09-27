import gc
import glob
import hashlib
import os
import sys
from typing import Any, List, Tuple

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
from transformers.utils import logging as hf_logging

from aesthetic_eval.adapters.base import BaseAdapter, require_directory
from aesthetic_eval.data import EvalSample


def _patch_clip_vision_tower(hf_name: str, local_path: str) -> None:
    if not local_path or not os.path.isdir(local_path):
        raise FileNotFoundError(
            f"UNIAA pinned CLIP vision tower snapshot is missing: {local_path!r}"
        )

    for cls in (CLIPVisionConfig, CLIPImageProcessor, CLIPVisionModel):
        if getattr(cls, "_uniaa_local_patch", False):
            continue
        original = cls.from_pretrained.__func__

        def wrapped(inner_cls, pretrained_model_name_or_path, *args, _original=original, **kwargs):
            if str(pretrained_model_name_or_path) == hf_name:
                pretrained_model_name_or_path = local_path
                kwargs["local_files_only"] = True
            return _original(inner_cls, pretrained_model_name_or_path, *args, **kwargs)

        cls.from_pretrained = classmethod(wrapped)
        cls._uniaa_local_patch = True


class UniaaAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.conv_templates = None
        self.separator_style = None
        self.tokenizer_image_token = None
        self.keywords_stopping_criteria = None
        self.image_token_index = None
        self.default_image_token = None
        self.default_im_start_token = None
        self.default_im_end_token = None

    def load(self) -> None:
        llava_repo_path = require_directory(
            self.model_cfg, "llava_repo_path", component="UNIAA-LLaVA"
        )
        if llava_repo_path and llava_repo_path not in sys.path:
            sys.path.insert(0, llava_repo_path)

        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        self.conv_templates = conv_templates
        self.separator_style = SeparatorStyle
        self.tokenizer_image_token = tokenizer_image_token
        self.keywords_stopping_criteria = KeywordsStoppingCriteria
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.default_image_token = DEFAULT_IMAGE_TOKEN
        self.default_im_start_token = DEFAULT_IM_START_TOKEN
        self.default_im_end_token = DEFAULT_IM_END_TOKEN

        model_path = str(self.model_cfg["model_path"])
        model_base = self.model_cfg.get("model_base")
        model_name = str(self.model_cfg.get("llava_model_name", "llava-v1.5"))
        device_map = self.model_cfg.get("device_map", "auto")
        if device_map in (None, "", "null"):
            device_map = "auto"
        if bool(self.model_cfg.get("require_cuda", True)) and not torch.cuda.is_available():
            raise RuntimeError(
                "UNIAA-LLaVA requires CUDA for fp16 inference, but torch.cuda.is_available() is False. "
                "Run from a GPU-visible shell with the documented LLaVA environment."
            )
        vision_tower_name = str(self.model_cfg.get("vision_tower_name", "openai/clip-vit-large-patch14-336"))
        vision_tower_local_path = str(self.model_cfg.get("vision_tower_local_path", ""))
        _patch_clip_vision_tower(vision_tower_name, vision_tower_local_path)

        disable_torch_init()
        old_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path,
                model_base,
                model_name,
                device_map=device_map,
                use_flash_attn=bool(self.model_cfg.get("use_flash_attn", False)),
            )
        finally:
            hf_logging.set_verbosity(old_verbosity)
        self.model.eval()
        self._load_uniaa_vision_tower(model_path)

    def _load_uniaa_vision_tower(self, model_path: str) -> None:
        weight_files = sorted(glob.glob(os.path.join(model_path, "pytorch_model-*.bin")))
        if not weight_files:
            raise FileNotFoundError(f"No UNIAA checkpoint shards found under {model_path}")

        prefix = "model.vision_tower.vision_tower.vision_model."
        encoder_dict = {}
        for weight_file in weight_files:
            state_dict = torch.load(weight_file, map_location="cpu")
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    encoder_dict[key.replace(prefix, "vision_model.")] = value
            del state_dict
            gc.collect()

        if not encoder_dict:
            raise RuntimeError(f"No vision tower weights found in UNIAA checkpoint shards under {model_path}")

        vision_tower = self.model.get_vision_tower()
        missing, unexpected = vision_tower.vision_tower.load_state_dict(encoder_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                "Failed to load UNIAA vision tower weights exactly: "
                f"missing={missing}, unexpected={unexpected}"
            )
        del encoder_dict
        gc.collect()

    def build_prompt(self, sample: EvalSample) -> str:
        return self.format_prompt(sample, "Please describe the aesthetic quality of this image.")

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        images = []
        valid: List[EvalSample] = []
        prompts: List[str] = []

        for sample in batch_samples:
            try:
                image = Image.open(sample.image_resolved).convert("RGB")
            except Exception:
                continue
            images.append(image)
            valid.append(sample)
            prompts.append(self.build_prompt(sample))

        if not valid:
            return None, [], []
        sample_ids = [sample.sample_id for sample in valid]
        return (images, prompts, sample_ids), valid, prompts

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _format_prompt(self, prompt: str) -> Tuple[str, Any, str]:
        if getattr(self.model.config, "mm_use_im_start_end", False):
            question = (
                self.default_im_start_token
                + self.default_image_token
                + self.default_im_end_token
                + "\n"
                + prompt
            )
        else:
            question = self.default_image_token + "\n" + prompt

        conv_mode = str(self.model_cfg.get("conv_mode", "llava_v1"))
        conv = self.conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != self.separator_style.TWO else conv.sep2
        return full_prompt, conv, stop_str

    def generate_batch(self, prepared: Any) -> List[str]:
        device = self._model_device()
        gen_cfg = self.base_cfg.get("generation", {})
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
        do_sample = bool(gen_cfg.get("do_sample", False))
        temperature = gen_cfg.get("temperature")
        top_p = gen_cfg.get("top_p", self.model_cfg.get("top_p", None))

        if len(prepared) == 3:
            images, prompts, sample_ids = prepared
        else:
            images, prompts = prepared
            sample_ids = [str(index) for index in range(len(images))]

        per_sample_seed = bool(gen_cfg.get("per_sample_seed", False))
        base_seed = int(self.base_cfg.get("seed", 42))

        outputs = []
        for image, prompt, sample_id in zip(images, prompts, sample_ids):
            if do_sample and per_sample_seed:
                seed_material = f"{base_seed}:{sample_id}".encode("utf-8")
                sample_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big") % (2**63 - 1)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
            full_prompt, _conv, stop_str = self._format_prompt(prompt)
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            image_tensor = image_tensor.to(device=device, dtype=torch.float16)
            input_ids = self.tokenizer_image_token(
                full_prompt,
                self.tokenizer,
                self.image_token_index,
                return_tensors="pt",
            ).unsqueeze(0).to(device)
            stopping_criteria = self.keywords_stopping_criteria([stop_str], self.tokenizer, input_ids)

            generate_kwargs = {
                "inputs": input_ids,
                "images": image_tensor,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "use_cache": True,
                "stopping_criteria": [stopping_criteria],
            }
            if do_sample and temperature is not None:
                generate_kwargs["temperature"] = float(temperature)
            if do_sample and top_p is not None:
                generate_kwargs["top_p"] = float(top_p)
            if not do_sample:
                generate_kwargs["num_beams"] = int(gen_cfg.get("num_beams", 1))

            with torch.inference_mode():
                output_ids = self.model.generate(**generate_kwargs)

            # LLaVA releases differ here: some return prompt+generation ids, while
            # newer Transformers paths with inputs_embeds may return only generated ids.
            if (
                output_ids.shape[1] >= input_ids.shape[1]
                and torch.equal(output_ids[:, : input_ids.shape[1]], input_ids)
            ):
                decode_ids = output_ids[:, input_ids.shape[1] :]
            else:
                decode_ids = output_ids
            text = self.tokenizer.batch_decode(
                decode_ids,
                skip_special_tokens=True,
            )[0].strip()
            if text.endswith(stop_str):
                text = text[: -len(stop_str)].strip()
            outputs.append(text)

        return outputs
