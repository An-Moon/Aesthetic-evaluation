import hashlib
from typing import Any, List, Tuple

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration
from peft import PeftModel

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample, load_and_resize_rgb


class QwenAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.processor = None
        self.model_type = ""

    @staticmethod
    def _remove_thinking(text: str) -> str:
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        if text.lstrip().startswith("<think>"):
            return ""
        return text.strip()

    def load(self) -> None:
        model_path = self.model_cfg["model_path"]
        peft_adapter_path = self.model_cfg.get("peft_adapter_path")
        dtype_name = str(self.model_cfg.get("torch_dtype", "bfloat16"))
        torch_dtype = getattr(torch, dtype_name)
        trust_remote_code = bool(self.model_cfg.get("trust_remote_code", True))
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model_type = str(config.model_type)

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        model_classes = {
            "qwen3_vl": Qwen3VLForConditionalGeneration,
            "qwen3_5": Qwen3_5ForConditionalGeneration,
        }
        model_class = model_classes.get(self.model_type)
        if model_class is None:
            raise ValueError(f"Unsupported Qwen multimodal model_type={self.model_type!r}")

        self.model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=self.model_cfg.get("device_map", {"": 0}),
            trust_remote_code=trust_remote_code,
            attn_implementation=self.model_cfg.get("attn_implementation", "sdpa"),
        ).eval()

        if peft_adapter_path:
            self.model = PeftModel.from_pretrained(self.model, peft_adapter_path)
            if bool(self.model_cfg.get("merge_lora_on_load", False)):
                self.model = self.model.merge_and_unload()
            self.model = self.model.eval()

        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

    def build_prompt(self, sample: EvalSample) -> str:
        return self.format_prompt(sample, "Please describe the aesthetic quality of this image.")

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        image_size = int(self.base_cfg.get("preprocess", {}).get("image_size", 448))

        texts: List[str] = []
        images = []
        valid: List[EvalSample] = []
        prompts: List[str] = []

        for s in batch_samples:
            try:
                img = load_and_resize_rgb(s.image_resolved, image_size)
                prompt = self.build_prompt(s)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }]
                template_kwargs = {}
                if "enable_thinking" in self.model_cfg:
                    template_kwargs["enable_thinking"] = bool(self.model_cfg["enable_thinking"])
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                texts.append(text)
                images.append(img)
                valid.append(s)
                prompts.append(prompt)
            except Exception as exc:
                if bool(self.base_cfg.get("runtime", {}).get("strict_inference", False)):
                    raise RuntimeError(
                        f"Qwen preprocessing failed for sample_id={s.sample_id} image={s.image_resolved}"
                    ) from exc
                continue

        if not valid:
            return None, [], []

        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        return (inputs, [sample.sample_id for sample in valid]), valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        if isinstance(prepared, tuple):
            prepared, sample_ids = prepared
        else:
            sample_ids = []
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device, non_blocking=True) for k, v in prepared.items()}

        gen_cfg = self.base_cfg.get("generation", {})
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
        do_sample = bool(gen_cfg.get("do_sample", False))
        per_sample_seed = bool(gen_cfg.get("per_sample_seed", False))
        if do_sample and per_sample_seed:
            if len(sample_ids) != 1:
                raise RuntimeError(
                    "Qwen per-sample reproducible sampling requires dataloader.batch_size=1; "
                    f"received {len(sample_ids)} sample ids"
                )
            base_seed = int(self.base_cfg.get("seed", 42))
            material = f"{base_seed}:{sample_ids[0]}".encode("utf-8")
            sample_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": int(gen_cfg.get("num_beams", 1)),
            "use_cache": True,
        }
        if do_sample:
            for name in ("temperature", "top_p", "top_k"):
                if gen_cfg.get(name) is not None:
                    generate_kwargs[name] = gen_cfg[name]

        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, **generate_kwargs)

        prompt_width = inputs["input_ids"].shape[1]
        trimmed = [out[prompt_width:] for out in out_ids]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)
        if bool(self.model_cfg.get("strip_thinking_output", False)):
            decoded = [self._remove_thinking(text) for text in decoded]
        return [d.strip() for d in decoded]
