from typing import Any, List, Tuple

import os
import sys
import types

from PIL import Image

from aesthetic_eval.adapters.base import BaseAdapter, require_directory
from aesthetic_eval.data import EvalSample


class OneAlignAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.scorer = None
        self.device = str(self.model_cfg.get("device", "cuda:0"))

    def load(self) -> None:
        repo_root = require_directory(self.model_cfg, "onealign_repo_root", component="OneAlign")
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        sys.modules.setdefault("icecream", types.SimpleNamespace(ic=lambda *a, **k: None))
        try:
            import accelerate.utils.memory as _acc_mem

            if not hasattr(_acc_mem, "clear_device_cache"):
                def _clear_device_cache():
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                _acc_mem.clear_device_cache = _clear_device_cache
        except Exception:
            pass

        from q_align.evaluate.scorer import QAlignAestheticScorer

        model_path = str(self.model_cfg.get("model_path", "q-future/one-align"))
        self._patch_attention_mask_compat()
        self._patch_hardcoded_model_path(model_path)
        self.scorer = QAlignAestheticScorer(pretrained=model_path, device=self.device).eval()

    def _patch_attention_mask_compat(self) -> None:
        try:
            import q_align.model.modeling_llama2 as modeling_llama2
        except Exception:
            return
        if hasattr(modeling_llama2, "_prepare_4d_causal_attention_mask_for_sdpa"):
            return
        if hasattr(modeling_llama2, "_prepare_4d_causal_attention_mask"):
            modeling_llama2._prepare_4d_causal_attention_mask_for_sdpa = (
                modeling_llama2._prepare_4d_causal_attention_mask
            )

    def _patch_hardcoded_model_path(self, model_path: str) -> None:
        try:
            import q_align.model.modeling_mplug_owl2 as modeling_mplug_owl2
        except Exception:
            return

        hf_name = "q-future/one-align"
        original_tokenizer = modeling_mplug_owl2.AutoTokenizer.from_pretrained
        original_processor = modeling_mplug_owl2.CLIPImageProcessor.from_pretrained

        def _redirect(path, *args, **kwargs):
            return model_path if str(path) == hf_name else path

        def _tokenizer_from_pretrained(path, *args, **kwargs):
            return original_tokenizer(_redirect(path), *args, **kwargs)

        def _processor_from_pretrained(path, *args, **kwargs):
            return original_processor(_redirect(path), *args, **kwargs)

        modeling_mplug_owl2.AutoTokenizer.from_pretrained = _tokenizer_from_pretrained
        modeling_mplug_owl2.CLIPImageProcessor.from_pretrained = _processor_from_pretrained

    def build_prompt(self, sample: EvalSample) -> str:
        q = str(sample.question or "").strip()
        if q:
            return q
        return "How would you rate the aesthetics of this image?"

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        valid: List[EvalSample] = []
        images: List[Image.Image] = []
        prompts: List[str] = []

        for s in batch_samples:
            try:
                img = Image.open(s.image_resolved).convert("RGB")
                valid.append(s)
                images.append(img)
                prompts.append(self.build_prompt(s))
            except Exception as exc:
                raise RuntimeError(
                    f"OneAlign preprocessing failed for sample_id={s.sample_id} image={s.image_resolved}"
                ) from exc

        if not valid:
            return None, [], []
        return {"images": images}, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        images: List[Image.Image] = prepared["images"]
        scores = self.scorer(images)
        return [f"{float(x):.6f}" for x in scores]
