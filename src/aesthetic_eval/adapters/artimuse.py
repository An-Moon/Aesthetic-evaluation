from typing import Any, List, Tuple

import os
import sys
import types

import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

from aesthetic_eval.adapters.base import BaseAdapter, require_directory
from aesthetic_eval.data import EvalSample

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _patch_peft_inference_shim() -> None:
    try:
        import peft  # noqa: F401
        return
    except Exception:
        pass

    shim = types.ModuleType("peft")

    class _DummyLoraConfig:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def _dummy_get_peft_model(model, _cfg):
        return model

    shim.LoraConfig = _DummyLoraConfig
    shim.get_peft_model = _dummy_get_peft_model
    sys.modules["peft"] = shim


class ArtiMuseAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.tokenizer = None
        self.device = str(self.model_cfg.get("device", "cuda:0"))
        self.infer_dtype = self._resolve_dtype()
        self.chat_batch_size = int(self.model_cfg.get("chat_batch_size", 16))
        image_size = int(
            self.model_cfg.get(
                "image_size", self.base_cfg.get("preprocess", {}).get("image_size", 448)
            )
        )
        self.image_size = image_size
        self.transform = _build_transform(image_size)

    def _resolve_dtype(self):
        dtype_name = str(self.model_cfg.get("torch_dtype", "bfloat16")).lower().strip()
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "float32":
            return torch.float32
        return torch.bfloat16

    def load(self) -> None:
        repo_root = require_directory(self.model_cfg, "repo_root", component="ArtiMuse")
        src_root = os.path.join(repo_root, "src")
        artimuse_root = os.path.join(repo_root, "src", "artimuse")
        if src_root not in sys.path:
            sys.path.insert(0, src_root)
        if artimuse_root not in sys.path:
            sys.path.insert(0, artimuse_root)

        try:
            from artimuse.internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig

            InternVLChatConfig.has_no_defaults_at_init = True
        except Exception:
            pass

        _patch_peft_inference_shim()

        from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel

        model_path = str(self.model_cfg["model_path"])
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
                f"ArtiMuse image_size={self.image_size} does not match checkpoint-required "
                f"image_size={required_image_size}; refusing a visual-token mismatch"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=bool(self.model_cfg.get("trust_remote_code", True)),
            use_fast=False,
        )

    def build_prompt(self, sample: EvalSample) -> str:
        return self.format_prompt(sample, "Please describe the aesthetic quality of this image.")

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        valid: List[EvalSample] = []
        prompts: List[str] = []
        pixel_values = []

        for s in batch_samples:
            try:
                img = Image.open(s.image_resolved).convert("RGB")
                pixel_values.append(self.transform(img).unsqueeze(0))
                valid.append(s)
                prompts.append(self.build_prompt(s))
            except Exception as exc:
                raise RuntimeError(
                    f"ArtiMuse preprocessing failed for sample_id={s.sample_id} image={s.image_resolved}"
                ) from exc

        if not valid:
            return None, [], []
        return {"pixel_values": torch.cat(pixel_values, dim=0), "prompts": prompts}, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        pixel_values = prepared["pixel_values"].to(self.device, dtype=self.infer_dtype)
        prompts = [str(x).strip() for x in prepared.get("prompts", [])]
        gen_cfg = {
            "max_new_tokens": int(self.base_cfg.get("generation", {}).get("max_new_tokens", 128)),
            "do_sample": bool(self.base_cfg.get("generation", {}).get("do_sample", False)),
            "num_beams": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        outs: List[str] = []
        for start in range(0, len(pixel_values), self.chat_batch_size):
            end = min(start + self.chat_batch_size, len(pixel_values))
            pv_chunk = pixel_values[start:end]
            q_chunk = prompts[start:end]
            chunk_outs = self.model.batch_chat(
                self.device,
                self.tokenizer,
                pv_chunk,
                q_chunk,
                dict(gen_cfg),
                num_patches_list=[1] * len(q_chunk),
            )
            outs.extend([str(t).strip() for t in chunk_outs])
        return outs
