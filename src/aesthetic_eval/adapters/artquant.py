from typing import Any, List, Tuple

import os
import sys
import types

import torch
from PIL import Image

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample


class ArtQuantAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.score_input_ids = None
        self.level_names: List[str] = []
        self.level_token_ids: List[int] = []
        self.device = str(self.model_cfg.get("device", "cuda:0"))
        self.infer_mode = str(self.model_cfg.get("infer_mode", "generate")).lower().strip()
        self.conv_mode = str(self.model_cfg.get("conv_mode", "mplug_owl2"))
        self.max_new_tokens = int(self.base_cfg.get("generation", {}).get("max_new_tokens", 256))
        self.do_sample = bool(self.base_cfg.get("generation", {}).get("do_sample", False))
        self.temperature = self.base_cfg.get("generation", {}).get("temperature", None)
        self.top_p = self.base_cfg.get("generation", {}).get("top_p", None)

    def load(self) -> None:
        repo_root = str(self.model_cfg.get("artquant_repo_root", "/home/Hu_xuanwei/ArtQuant"))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # Some ArtQuant snapshots reference src.evaluate.scorer in src/__init__.py,
        # but the file may be absent. Add a tiny shim to keep imports working.
        if "src.evaluate.scorer" not in sys.modules:
            scorer_mod = types.ModuleType("src.evaluate.scorer")

            class _Scorer:  # pragma: no cover
                pass

            scorer_mod.Scorer = _Scorer
            sys.modules["src.evaluate.scorer"] = scorer_mod

        from src.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from src.conversation import SeparatorStyle, conv_templates
        from src.mm_utils import tokenizer_image_token
        from src.model.builder import load_pretrained_model

        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._conv_templates = conv_templates
        self._SeparatorStyle = SeparatorStyle
        self._tokenizer_image_token = tokenizer_image_token

        model_path = str(self.model_cfg["model_path"])
        model_base = str(self.model_cfg.get("model_base", "/home/Hu_xuanwei/model/mplug-owl2-llama2-7b"))
        preprocessor_path = str(self.model_cfg.get("preprocessor_path", "/home/Hu_xuanwei/ArtQuant/preprocessor"))
        if not os.path.exists(os.path.join(preprocessor_path, "config.json")):
            preprocessor_path = model_base
        model_name = str(self.model_cfg.get("model_name_hint", "deqa_lora_prompt_apdd"))
        if "deqa" not in model_name.lower():
            model_name = "deqa_lora_prompt_apdd"

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            load_8bit=bool(self.model_cfg.get("load_8bit", False)),
            load_4bit=bool(self.model_cfg.get("load_4bit", False)),
            device=self.device,
            preprocessor_path=preprocessor_path,
        )
        self.tokenizer.padding_side = "left"
        self.model.eval()

        # Keep generation config consistent to avoid repeated warnings from
        # checkpoint defaults (e.g., do_sample=False with top_p/temperature set).
        try:
            if not self.do_sample:
                self.model.generation_config.do_sample = False
                self.model.generation_config.temperature = 1.0
                self.model.generation_config.top_p = 1.0
        except Exception:
            pass

        if self.infer_mode == "score":
            self.level_names = list(
                self.model_cfg.get("level_names", ["excellent", "good", "fair", "poor", "bad"])
            )
            self.level_token_ids = []
            for tok in self.level_names:
                ids = self.tokenizer(tok)["input_ids"]
                self.level_token_ids.append(ids[-1])

            score_prompt = str(
                self.model_cfg.get(
                    "score_prompt",
                    "How would you rate the quality of this painting?",
                )
            )
            score_suffix = str(
                self.model_cfg.get("score_suffix", "The quality of the painting is")
            )
            conv = self._conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], score_prompt + "\n" + self._DEFAULT_IMAGE_TOKEN)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " " + score_suffix
            self.score_input_ids = (
                self._tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    self._IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(self.device)
            )

    def build_prompt(self, sample: EvalSample) -> str:
        q = str(sample.question or "").strip()
        if q:
            return q
        if self.infer_mode == "score":
            return "How would you rate the quality of this painting?"
        return "Please describe the aesthetic quality of this image."

    def _expand2square(self, pil_img: Image.Image, background_color: tuple) -> Image.Image:
        width, height = pil_img.size
        if width == height:
            return pil_img
        if width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        valid: List[EvalSample] = []
        prompts: List[str] = []
        image_tensors = []
        records = []

        bg = tuple(int(x * 255) for x in self.image_processor.image_mean)
        for s in batch_samples:
            try:
                image = Image.open(s.image_resolved).convert("RGB")
                image = self._expand2square(image, bg)
                image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half()
                valid.append(s)
                prompt_text = self.build_prompt(s)
                prompts.append(prompt_text)

                if self.infer_mode == "generate":
                    conv = self._conv_templates[self.conv_mode].copy()
                    conv.append_message(conv.roles[0], prompt_text + "\n" + self._DEFAULT_IMAGE_TOKEN)
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
                            "image_tensor": image_tensor,
                            "stop_str": stop_str,
                        }
                    )
                else:
                    image_tensors.append(image_tensor)
            except Exception:
                continue

        if not valid:
            return None, [], []

        if self.infer_mode == "generate":
            return {"records": records}, valid, prompts

        images = torch.cat(image_tensors, 0).to(self.device)
        return {"images": images}, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        if self.infer_mode == "generate":
            records = prepared["records"]
            if not records:
                return []

            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            if pad_id is None:
                pad_id = 0

            input_ids_list = [r["input_ids"].squeeze(0) for r in records]
            max_len = max(x.shape[0] for x in input_ids_list)
            input_ids = torch.full(
                (len(input_ids_list), max_len),
                pad_id,
                dtype=input_ids_list[0].dtype,
            )
            for i, ids in enumerate(input_ids_list):
                input_ids[i, -ids.shape[0] :] = ids
            attention_mask = (input_ids != pad_id).long()
            images = torch.cat([r["image_tensor"] for r in records], dim=0)
            stop_strs = [r["stop_str"] for r in records]

            gen_kwargs = {
                "do_sample": self.do_sample,
                "max_new_tokens": self.max_new_tokens,
                "use_cache": True,
            }
            if self.do_sample:
                if self.temperature is not None:
                    gen_kwargs["temperature"] = float(self.temperature)
                if self.top_p is not None:
                    gen_kwargs["top_p"] = float(self.top_p)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    images=images.to(self.device),
                    **gen_kwargs,
                )

            prompt_len = input_ids.shape[1]
            decoded = self.tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
            outs: List[str] = []
            for text, stop_str in zip(decoded, stop_strs):
                text = text.strip()
                if stop_str and text.endswith(stop_str):
                    text = text[: -len(stop_str)].strip()
                outs.append(text)
            return outs

        images = prepared["images"]
        input_ids = self.score_input_ids.repeat(images.shape[0], 1)
        with torch.inference_mode():
            output_logits = self.model(input_ids=input_ids, images=images)["logits"][:, -1]

        preds: List[str] = []
        for j in range(output_logits.shape[0]):
            scores = [float(output_logits[j, tok_id].item()) for tok_id in self.level_token_ids]
            best_idx = int(torch.tensor(scores).argmax().item())
            preds.append(self.level_names[best_idx])
        return preds
