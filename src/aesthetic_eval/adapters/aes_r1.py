from typing import Any, List, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample


class AesR1Adapter(BaseAdapter):
    """Aes-R1 adapter using its native Qwen2.5-VL inference path."""

    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.processor = None

    def load(self) -> None:
        model_path = str(self.model_cfg["model_path"])
        dtype_name = str(self.model_cfg.get("torch_dtype", "bfloat16"))
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unsupported Aes-R1 torch dtype: {dtype_name!r}")
        torch_dtype = getattr(torch, dtype_name)
        local_files_only = bool(self.model_cfg.get("local_files_only", True))

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=bool(self.model_cfg.get("trust_remote_code", False)),
            local_files_only=local_files_only,
            use_fast=bool(self.model_cfg.get("use_fast", False)),
            min_pixels=int(self.model_cfg.get("min_pixels", 3136)),
            max_pixels=int(self.model_cfg.get("max_pixels", 262144)),
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=self.model_cfg.get("device_map", {"": 0}),
            trust_remote_code=bool(self.model_cfg.get("trust_remote_code", False)),
            local_files_only=local_files_only,
            attn_implementation=str(
                self.model_cfg.get("attn_implementation", "flash_attention_2")
            ),
        ).eval()
        if str(getattr(self.model.config, "model_type", "")) != "qwen2_5_vl":
            raise RuntimeError(
                f"Aes-R1 architecture mismatch: {getattr(self.model.config, 'model_type', None)!r}"
            )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

    def build_prompt(self, sample: EvalSample) -> str:
        return self.format_prompt(
            sample,
            "Please provide a detailed aesthetic critique of this image.",
        )

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        texts: List[str] = []
        images: List[Image.Image] = []
        valid: List[EvalSample] = []
        prompts: List[str] = []
        for sample in batch_samples:
            try:
                with Image.open(sample.image_resolved) as opened:
                    image = opened.convert("RGB")
                prompt = self.build_prompt(sample)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }]
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                texts.append(text)
                images.append(image)
                valid.append(sample)
                prompts.append(prompt)
            except Exception as exc:
                raise RuntimeError(
                    f"Aes-R1 preprocessing failed for sample_id={sample.sample_id} "
                    f"image={sample.image_resolved}"
                ) from exc

        if not valid:
            return None, [], []
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        return inputs, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        device = next(self.model.parameters()).device
        inputs = {name: value.to(device, non_blocking=True) for name, value in prepared.items()}
        generation = self.base_cfg.get("generation", {})
        do_sample = bool(generation.get("do_sample", False))
        kwargs = {
            "max_new_tokens": int(generation.get("max_new_tokens", 256)),
            "do_sample": do_sample,
            "num_beams": int(generation.get("num_beams", 1)),
            "repetition_penalty": float(self.model_cfg.get("repetition_penalty", 1.05)),
            "use_cache": bool(self.model_cfg.get("use_cache", False)),
        }
        if do_sample:
            for name in ("temperature", "top_p", "top_k"):
                value = generation.get(name)
                if value is not None:
                    kwargs[name] = value

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)

        prompt_width = int(inputs["input_ids"].shape[1])
        generated_only = [row[prompt_width:] for row in generated]
        decoded = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        outputs = [text.strip() for text in decoded]
        if str(self.model_cfg.get("response_mode", "full_text")) != "full_text":
            raise ValueError("Aes-R1 currently supports only response_mode=full_text")
        return outputs
