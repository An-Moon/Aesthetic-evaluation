from typing import Any, List, Tuple

import inspect
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from aesthetic_eval.adapters.base import BaseAdapter, require_directory
from aesthetic_eval.data import EvalSample


class UniPerceptAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.tokenizer = None
        self.device = str(self.model_cfg.get("device", "cuda:0"))
        self.infer_mode = str(self.model_cfg.get("infer_mode", "generate")).lower().strip()
        self.chat_batch_size = int(self.model_cfg.get("chat_batch_size", 16))
        self.infer_dtype = getattr(torch, str(self.model_cfg.get("torch_dtype", "bfloat16")))
        image_size = int(
            self.model_cfg.get(
                "image_size", self.base_cfg.get("preprocess", {}).get("image_size", 448)
            )
        )
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def load(self) -> None:
        repo_root = require_directory(
            self.model_cfg, "unipercept_repo_root", component="UniPercept"
        )
        src_root = os.path.join(repo_root, "src")
        if src_root not in sys.path:
            sys.path.insert(0, src_root)

        from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig

        InternVLChatConfig.has_no_defaults_at_init = True

        from internvl.model.internvl_chat.modeling_unipercept import InternVLChatModel
        from transformers import AutoTokenizer

        model_path = str(self.model_cfg["model_path"])
        trust_remote_code = bool(self.model_cfg.get("trust_remote_code", True))

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        self.model = InternVLChatModel.from_pretrained(
            model_path,
            torch_dtype=self.infer_dtype,
            low_cpu_mem_usage=bool(self.model_cfg.get("low_cpu_mem_usage", True)),
            use_flash_attn=bool(self.model_cfg.get("use_flash_attn", False)),
        ).to(self.device).eval()

        required_image_size = int(
            getattr(self.model.config, "force_image_size", None)
            or self.model.config.vision_config.image_size
        )
        if self.image_size != required_image_size:
            raise RuntimeError(
                f"UniPercept image_size={self.image_size} does not match checkpoint-required "
                f"image_size={required_image_size}; refusing a visual-token mismatch"
            )

    def build_prompt(self, sample: EvalSample) -> str:
        return self.format_prompt(sample, "Please describe the aesthetic quality of this image.")

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        valid: List[EvalSample] = []
        prompts: List[str] = []
        pixel_values = []

        for s in batch_samples:
            img = Image.open(s.image_resolved).convert("RGB")
            pv = self.transform(img).unsqueeze(0)
            pixel_values.append(pv)
            valid.append(s)
            prompts.append(self.build_prompt(s))

        if not valid:
            return None, [], []

        stacked = torch.cat(pixel_values, dim=0)
        return {"pixel_values": stacked, "prompts": prompts}, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        pixel_values = prepared["pixel_values"].to(self.device, dtype=self.infer_dtype)
        prompts = prepared.get("prompts", [])

        gen_cfg = {
            "max_new_tokens": int(self.base_cfg.get("generation", {}).get("max_new_tokens", 128)),
            "do_sample": bool(self.base_cfg.get("generation", {}).get("do_sample", False)),
            "num_beams": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        outs: List[str] = []
        if self.infer_mode == "reward":
            if not hasattr(self.model, "score"):
                raise RuntimeError("UniPercept reward mode requires the official model.score API")
            for pv in pixel_values:
                iaa = float(self._score_once(pv.unsqueeze(0), dict(gen_cfg), "aesthetics"))
                iqa = float(self._score_once(pv.unsqueeze(0), dict(gen_cfg), "quality"))
                ista = float(self._score_once(pv.unsqueeze(0), dict(gen_cfg), "structure and texture"))
                outs.append(f'{{"iaa": {iaa:.4f}, "iqa": {iqa:.4f}, "ista": {ista:.4f}}}')
            return outs

        default_prompt = "Please analyze this image from an aesthetic perspective."
        all_prompts = [str(prompts[i]).strip() if i < len(prompts) else default_prompt for i in range(len(pixel_values))]

        use_batch_chat = hasattr(self.model, "batch_chat")
        if use_batch_chat:
            for start in range(0, len(pixel_values), self.chat_batch_size):
                end = min(start + self.chat_batch_size, len(pixel_values))
                pv_chunk = pixel_values[start:end]
                q_chunk = all_prompts[start:end]
                chunk_outs = self._batch_chat(pv_chunk, q_chunk, dict(gen_cfg))
                outs.extend([str(t).strip() for t in chunk_outs])
            return outs

        for i, pv in enumerate(pixel_values):
            text = self._chat_once(pv.unsqueeze(0), all_prompts[i], dict(gen_cfg))
            outs.append(str(text).strip())
        return outs

    def _batch_chat(self, pixel_values: torch.Tensor, questions: List[str], gen_cfg: dict) -> List[str]:
        sig = inspect.signature(self.model.batch_chat)
        params = list(sig.parameters.keys())
        num_patches_list = [1] * len(questions)
        if params and params[0] == "device":
            return self.model.batch_chat(self.device, self.tokenizer, pixel_values, questions, gen_cfg, num_patches_list=num_patches_list)
        return self.model.batch_chat(self.tokenizer, pixel_values, questions, gen_cfg, num_patches_list=num_patches_list)

    def _chat_once(self, pixel_values: torch.Tensor, prompt: str, gen_cfg: dict) -> str:
        sig = inspect.signature(self.model.chat)
        params = list(sig.parameters.keys())
        if len(params) >= 2 and params[1] == "device":
            return self.model.chat(self.device, self.tokenizer, pixel_values, prompt, gen_cfg)
        return self.model.chat(self.tokenizer, pixel_values, prompt, gen_cfg)

    def _score_once(self, pixel_values: torch.Tensor, gen_cfg: dict, desc: str) -> float:
        sig = inspect.signature(self.model.score)
        params = list(sig.parameters.keys())
        if len(params) >= 2 and params[1] == "device":
            return float(self.model.score(self.device, self.tokenizer, pixel_values, gen_cfg, desc=desc))
        return float(self.model.score(self.tokenizer, pixel_values, gen_cfg, desc=desc))
