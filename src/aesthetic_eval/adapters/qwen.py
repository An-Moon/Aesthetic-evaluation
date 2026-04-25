from typing import Any, List, Tuple

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample, load_and_resize_rgb


class QwenAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.processor = None

    def load(self) -> None:
        model_path = self.model_cfg["model_path"]
        peft_adapter_path = self.model_cfg.get("peft_adapter_path")
        dtype_name = str(self.model_cfg.get("torch_dtype", "bfloat16"))
        torch_dtype = getattr(torch, dtype_name)

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=bool(self.model_cfg.get("trust_remote_code", True)),
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=self.model_cfg.get("device_map", {"": 0}),
            trust_remote_code=bool(self.model_cfg.get("trust_remote_code", True)),
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
        template = str(self.base_cfg.get("prompt", {}).get("template", "{question}"))
        return template.format(question=sample.question)

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
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(text)
                images.append(img)
                valid.append(s)
                prompts.append(prompt)
            except Exception:
                continue

        if not valid:
            return None, [], []

        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        return inputs, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device, non_blocking=True) for k, v in prepared.items()}

        gen_cfg = self.base_cfg.get("generation", {})
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
        do_sample = bool(gen_cfg.get("do_sample", False))

        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True,
            )

        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        trimmed = [out[int(in_len):] for out, in_len in zip(out_ids, input_lens)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)
        return [d.strip() for d in decoded]
