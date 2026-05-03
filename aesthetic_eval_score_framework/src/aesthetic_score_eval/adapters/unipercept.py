import os
import sys
import types
from typing import Dict, List

import torch
from PIL import Image, ImageFile
from transformers import AutoTokenizer

from aesthetic_score_eval.adapters.base import BaseScoreAdapter
from aesthetic_score_eval.data import ScoreSample

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class UniPerceptScoreAdapter(BaseScoreAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.tokenizer = None
        self.device = str(model_cfg.get("device", "cuda:0"))
        self.score_desc = str(model_cfg.get("score_desc", "aesthetics"))
        self.image_size = int(model_cfg.get("image_size", 448))
        self.transform = None
        self.generation_config = {
            "max_new_tokens": int(model_cfg.get("generation", {}).get("max_new_tokens", 1024)),
            "do_sample": bool(model_cfg.get("generation", {}).get("do_sample", False)),
        }

    def _resolve_dtype(self):
        dtype_name = str(self.model_cfg.get("torch_dtype", "bfloat16")).lower().strip()
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "float32":
            return torch.float32
        return torch.bfloat16

    def load(self) -> None:
        self.transform = build_transform(self.image_size)
        repo_root = str(self.model_cfg.get("repo_root", self.model_cfg.get("unipercept_repo_root", "")))
        src_root = os.path.join(repo_root, "src")
        if src_root not in sys.path:
            sys.path.insert(0, src_root)

        try:
            from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig

            InternVLChatConfig.has_no_defaults_at_init = True
        except Exception:
            pass

        try:
            import peft  # noqa: F401
        except Exception:
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

        from internvl.model.internvl_chat.modeling_unipercept import InternVLChatModel

        model_path = str(self.model_cfg["model_path"])
        dtype = self._resolve_dtype()
        self.model = InternVLChatModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=bool(self.model_cfg.get("low_cpu_mem_usage", True)),
            use_flash_attn=bool(self.model_cfg.get("use_flash_attn", False)),
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def _load_image(self, image_file: str):
        image = Image.open(image_file).convert("RGB")
        return self.transform(image).unsqueeze(0)

    def score_batch(self, batch_samples: List[ScoreSample]) -> List[Dict]:
        outputs: List[Dict] = []
        dtype = self._resolve_dtype()

        for sample in batch_samples:
            try:
                pixel_values = self._load_image(sample.image_resolved).to(dtype).to(self.device)
                score = self.model.score(
                    self.device,
                    self.tokenizer,
                    pixel_values,
                    dict(self.generation_config),
                    desc=self.score_desc,
                )
                outputs.append(
                    {
                        "raw_score": float(score),
                        "raw_response": {self.score_desc: float(score)},
                        "parse_status": "ok",
                        "error": "",
                    }
                )
            except Exception as e:
                outputs.append({"raw_score": None, "raw_response": "", "parse_status": "error", "error": str(e)})

        return outputs
