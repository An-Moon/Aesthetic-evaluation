from aesthetic_score_eval.adapters.base import BaseScoreAdapter
from aesthetic_score_eval.io_utils import write_json
import os
import re
import time
import torch
import numpy as np
import importlib
import sys
import types

class ArtQuantAdapter(BaseScoreAdapter):
    """Adapter for ArtQuant official scoring flow.

    This adapter attempts to reuse the official ArtQuant evaluation logic where
    possible. It expects model weights at `model_weights` and base model path
    in the model config. The adapter provides `load` and `generate_batch`.
    """

    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.model_config = model_cfg
        self.base_config = base_cfg

    def load(self, device='cuda'):
        # Try to prepare ArtQuant model loader
        artquant_root = self.model_config.get("artquant_root")
        if artquant_root and os.path.isdir(artquant_root):
            if artquant_root not in sys.path:
                sys.path.insert(0, artquant_root)

        self.model_weights = self.model_config.get("weights")
        self.base_model = self.model_config.get("base_model")

        # level names and score weights (defaults mirror ArtQuant eval.sh)
        self.level_names = self.model_config.get("level_names", ["excellent", "good", "fair", "poor", "bad"])
        self.score_weight = self.model_config.get("score_weight", [10.0, 7.5, 5.0, 3.5, 1.0])

        # Lazy load heavy model on first score_batch call
        self._loaded = False
        return self

    def build_prompt(self, sample):
        # ArtQuant uses logits-based official scoring; no prompt required.
        return None

    def prepare_batch(self, batch_samples):
        # return as-is; generator will either call official eval or fallback
        return batch_samples

    def score_batch(self, batch_samples):
        """Return list[dict] with keys 'raw_score' and 'error' per sample."""
        results = []

        # Load model components if needed
        if not getattr(self, '_loaded', False):
            try:
                # Inject lightweight shims for optional dependencies used by ArtQuant.
                sys.modules.setdefault('icecream', types.SimpleNamespace(ic=lambda *a, **k: None))
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
                try:
                    import transformers as _tfm

                    if not hasattr(_tfm, "Cache"):
                        class Cache:  # pragma: no cover
                            pass

                        _tfm.Cache = Cache
                    if not hasattr(_tfm, "DynamicCache"):
                        class DynamicCache:  # pragma: no cover
                            pass

                        _tfm.DynamicCache = DynamicCache
                    if not hasattr(_tfm, "EncoderDecoderCache"):
                        class EncoderDecoderCache:  # pragma: no cover
                            pass

                        _tfm.EncoderDecoderCache = EncoderDecoderCache
                except Exception:
                    pass

                # Build a synthetic `src` package that points only at ArtQuant's src tree.
                # This avoids executing ArtQuant's top-level src/__init__.py, which imports
                # missing modules and breaks normal inference.
                artquant_src_dir = os.path.join(self.model_config.get("artquant_root"), "src")
                src_pkg = types.ModuleType('src')
                src_pkg.__path__ = [artquant_src_dir]
                src_pkg.__package__ = 'src'
                sys.modules['src'] = src_pkg

                # import ArtQuant helpers from the official repository tree
                from src.model.builder import load_pretrained_model
                from src.mm_utils import tokenizer_image_token, process_images
                from src.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
                from src.conversation import conv_templates

                preprocessor_path = os.path.join(self.model_config.get("artquant_root"), "preprocessor")
                if not os.path.exists(os.path.join(preprocessor_path, "config.json")):
                    preprocessor_path = self.base_model

                model_loader_name = self.model_config.get("model_loader_name")
                if not model_loader_name:
                    model_loader_name = os.path.basename(self.model_weights)
                if "deqa" not in str(model_loader_name).lower():
                    model_loader_name = "deqa_lora_prompt_apdd"

                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    self.model_weights,
                    self.base_model,
                    model_loader_name,
                    device=self.base_config.get('device', 'cuda'),
                    preprocessor_path=preprocessor_path,
                )
                self.tokenizer = tokenizer
                self.model = model
                self.image_processor = image_processor
                self.context_len = context_len
                # prepare token ids for level names
                tokenized = tokenizer(self.level_names)
                self.level_ids = [ids[1] for ids in tokenized["input_ids"]]
                # stash helper functions
                self._tokenizer_image_token = tokenizer_image_token
                self._process_images = process_images
                self._AQ_DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
                self._AQ_IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
                self._conv_templates = conv_templates
                self._loaded = True
            except Exception as e:
                # If the official loader still fails, surface the error per-sample.
                for s in batch_samples:
                    results.append({'raw_score': None, 'raw_response': '', 'parse_status': 'error', 'error': f'load_error:{e}'})
                return results

        # Build prompt template used by ArtQuant iqa_eval
        conv_mode = "mplug_owl2"
        conv = self._conv_templates[conv_mode].copy()
        inp = "How would you rate the quality of this painting?"
        AQ_DEFAULT_IMAGE_TOKEN = self._AQ_DEFAULT_IMAGE_TOKEN
        inp = inp + "\n" + AQ_DEFAULT_IMAGE_TOKEN
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " The quality of the painting is"

        # prepare batch images and input ids
        images = []
        for s in batch_samples:
            try:
                from PIL import Image
                img = Image.open(s.image_resolved).convert('RGB')
            except Exception:
                images.append(None)
                continue
            images.append(img)

        # process images into tensors
        try:
            image_tensors = self._process_images(images, self.image_processor, model_cfg=None)
        except Exception:
            # fallback: try per-image processing
            image_tensors = []
            for img in images:
                if img is None:
                    image_tensors.append(None)
                else:
                    image_tensors.append(self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0])
            image_tensors = torch.stack([t for t in image_tensors if t is not None], dim=0) if any(t is not None for t in image_tensors) else torch.empty(0)

        if image_tensors.numel() == 0:
            for _ in batch_samples:
                results.append({'raw_score': None, 'raw_response': '', 'parse_status': 'error', 'error': 'image_load_failed'})
            return results

        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        image_tensors = image_tensors.to(device=model_device, dtype=model_dtype)

        # build input ids tensor
        input_ids = self._tokenizer_image_token(prompt, self.tokenizer, self._AQ_IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model_device)

        with torch.inference_mode():
            output = self.model(input_ids=input_ids.repeat(image_tensors.shape[0], 1), images=image_tensors)
            logits = output['logits'][:, -1]
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        for i in range(len(batch_samples)):
            try:
                prob_vec = probs[i, :]
                level_probs = prob_vec[self.level_ids]
                raw_score = float(np.inner(level_probs, np.array(self.score_weight)))
                raw_response = {
                    name: float(prob)
                    for name, prob in zip(self.level_names, level_probs.tolist())
                }
                results.append({'raw_score': raw_score, 'raw_response': raw_response, 'parse_status': 'ok', 'error': ''})
            except Exception as e:
                results.append({'raw_score': None, 'raw_response': '', 'parse_status': 'error', 'error': str(e)})

        return results
