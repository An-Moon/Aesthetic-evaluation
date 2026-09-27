from typing import Any, List, Tuple

import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample, load_and_resize_rgb


class Glm4vAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.processor = None

    @staticmethod
    def _remove_thinking(text: str) -> str:
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        if text.lstrip().startswith("<think>"):
            return ""
        return text.strip()

    def load(self) -> None:
        model_path = self.model_cfg["model_path"]
        dtype_name = str(self.model_cfg.get("torch_dtype", "bfloat16"))
        torch_dtype = getattr(torch, dtype_name)
        trust_remote_code = bool(self.model_cfg.get("trust_remote_code", True))

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            device_map=self.model_cfg.get("device_map", {"": 0}),
            trust_remote_code=trust_remote_code,
            attn_implementation=self.model_cfg.get("attn_implementation", "sdpa"),
        ).eval()

    def build_prompt(self, sample: EvalSample) -> str:
        return self.format_prompt(sample, "Please describe the aesthetic quality of this image.")

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        image_size = int(self.base_cfg.get("preprocess", {}).get("image_size", 448))

        conversations = []
        valid: List[EvalSample] = []
        prompts: List[str] = []

        for sample in batch_samples:
            try:
                image = load_and_resize_rgb(sample.image_resolved, image_size)
                prompt = self.build_prompt(sample)
                conversations.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                )
                valid.append(sample)
                prompts.append(prompt)
            except Exception as exc:
                if bool(self.base_cfg.get("runtime", {}).get("strict_inference", False)):
                    raise RuntimeError(
                        f"GLM-4V preprocessing failed for sample_id={sample.sample_id} image={sample.image_resolved}"
                    ) from exc
                continue

        if not valid:
            return None, [], []

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
            enable_thinking=bool(self.model_cfg.get("enable_thinking", False)),
        )
        return inputs, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in prepared.items()
        }
        inputs.pop("token_type_ids", None)

        gen_cfg = self.base_cfg.get("generation", {})
        generate_kwargs = {
            "max_new_tokens": int(gen_cfg.get("max_new_tokens", 256)),
            "do_sample": bool(gen_cfg.get("do_sample", False)),
            "num_beams": int(gen_cfg.get("num_beams", 1)),
            "use_cache": True,
        }
        if generate_kwargs["do_sample"]:
            if gen_cfg.get("temperature") is not None:
                generate_kwargs["temperature"] = float(gen_cfg["temperature"])
            if gen_cfg.get("top_p") is not None:
                generate_kwargs["top_p"] = float(gen_cfg["top_p"])
            if gen_cfg.get("top_k") is not None:
                generate_kwargs["top_k"] = int(gen_cfg["top_k"])

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)

        prompt_width = inputs["input_ids"].shape[1]
        decoded = [
            self.processor.decode(ids[prompt_width:], skip_special_tokens=True).strip()
            for ids in generated_ids
        ]
        if bool(self.model_cfg.get("strip_thinking_output", True)):
            decoded = [self._remove_thinking(text) for text in decoded]
        return decoded
