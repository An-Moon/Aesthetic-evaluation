import sys
import types
from typing import Dict, List

from PIL import Image, ImageFile

from aesthetic_score_eval.adapters.base import BaseScoreAdapter
from aesthetic_score_eval.data import ScoreSample

ImageFile.LOAD_TRUNCATED_IMAGES = True


class QAlignScoreAdapter(BaseScoreAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.scorer = None
        self.device = str(model_cfg.get("device", "cuda:0"))

    def load(self) -> None:
        repo_root = str(self.model_cfg.get("repo_root", self.model_cfg.get("qalign_repo_root", "")))
        if repo_root and repo_root not in sys.path:
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
            modeling_llama2._prepare_4d_causal_attention_mask_for_sdpa = modeling_llama2._prepare_4d_causal_attention_mask

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

    def score_batch(self, batch_samples: List[ScoreSample]) -> List[Dict]:
        outputs: List[Dict] = []
        valid_images: List[Image.Image] = []
        valid_indices: List[int] = []

        for idx, sample in enumerate(batch_samples):
            try:
                valid_images.append(Image.open(sample.image_resolved).convert("RGB"))
                valid_indices.append(idx)
                outputs.append({})
            except Exception as e:
                outputs.append({"raw_score": None, "raw_response": "", "parse_status": "error", "error": str(e)})

        if valid_images:
            try:
                scores = self.scorer(valid_images)
                for out_idx, score in zip(valid_indices, scores):
                    # Official Q-Align scorer returns 0-1 WA5. The model config maps it to 0-10.
                    outputs[out_idx] = {
                        "raw_score": float(score),
                        "raw_response": "",
                        "parse_status": "ok",
                        "error": "",
                    }
            except Exception as e:
                for out_idx in valid_indices:
                    outputs[out_idx] = {
                        "raw_score": None,
                        "raw_response": "",
                        "parse_status": "error",
                        "error": str(e),
                    }

        return outputs
