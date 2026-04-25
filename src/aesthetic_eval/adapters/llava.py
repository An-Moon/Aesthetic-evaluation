from typing import Any, List, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    LlavaProcessor,
)

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample, load_and_resize_rgb


class LlavaAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.use_chat_template = True

    def load(self) -> None:
        model_path = self.model_cfg["model_path"]
        dtype_name = str(self.model_cfg.get("torch_dtype", "float16"))
        torch_dtype = getattr(torch, dtype_name)
        trust_remote_code = bool(self.model_cfg.get("trust_remote_code", True))
        device_map = self.model_cfg.get("device_map", {"": 0})

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            ).eval()
        except ValueError:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            ).eval()

        use_fast = bool(self.model_cfg.get("use_fast_tokenizer", True))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left",
                trust_remote_code=trust_remote_code,
                use_fast=use_fast,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left",
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )

        image_token = str(self.model_cfg.get("image_token", "<image>"))
        image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id == self.tokenizer.unk_token_id:
            added = self.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
            if added > 0 and hasattr(self.model, "resize_token_embeddings"):
                self.model.resize_token_embeddings(len(self.tokenizer))

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                tokenizer=self.tokenizer,
                trust_remote_code=trust_remote_code,
                use_fast=use_fast,
            )
            self.use_chat_template = hasattr(self.processor, "apply_chat_template")
        except Exception:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            vision_tower = getattr(cfg, "mm_vision_tower", "openai/clip-vit-large-patch14-336")
            vision_tower_local = self.model_cfg.get("vision_tower_local_path")
            image_processor = None
            if vision_tower_local:
                try:
                    image_processor = CLIPImageProcessor.from_pretrained(
                        vision_tower_local,
                        local_files_only=True,
                    )
                except Exception:
                    image_processor = None

            if image_processor is None:
                try:
                    image_processor = CLIPImageProcessor.from_pretrained(
                        vision_tower,
                        local_files_only=True,
                    )
                except Exception:
                    image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

            vision_image_size = int(self.model_cfg.get("vision_image_size", 336))
            if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
                image_processor.size["shortest_edge"] = vision_image_size
            if hasattr(image_processor, "crop_size") and isinstance(image_processor.crop_size, dict):
                image_processor.crop_size["height"] = vision_image_size
                image_processor.crop_size["width"] = vision_image_size

            self.processor = LlavaProcessor(image_processor=image_processor, tokenizer=self.tokenizer)
            self.processor.image_token = image_token
            self.processor.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
            if getattr(self.processor, "patch_size", None) is None:
                self.processor.patch_size = int(self.model_cfg.get("vision_patch_size", 14))
            if getattr(self.processor, "vision_feature_select_strategy", None) is None:
                self.processor.vision_feature_select_strategy = "default"
            self.use_chat_template = False

    def build_prompt(self, sample: EvalSample) -> str:
        template = str(self.base_cfg.get("prompt", {}).get("template", "{question}"))
        return template.format(question=sample.question)

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        image_size = int(self.base_cfg.get("preprocess", {}).get("image_size", 448))

        images = []
        valid: List[EvalSample] = []
        prompts: List[str] = []

        for s in batch_samples:
            try:
                img = load_and_resize_rgb(s.image_resolved, image_size)
                prompt = self.build_prompt(s)
                images.append(img)
                valid.append(s)
                prompts.append(prompt)
            except Exception:
                continue

        if not valid:
            return None, [], []

        if self.use_chat_template and hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": p},
                    ],
                }
                for p in prompts
            ]
            texts = [
                self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
        else:
            texts = [f"USER: <image>\n{p}\nASSISTANT:" for p in prompts]

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        return inputs, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device, non_blocking=True) for k, v in prepared.items()}

        gen_cfg = self.base_cfg.get("generation", {})
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
        do_sample = bool(gen_cfg.get("do_sample", False))

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [o.strip() for o in outputs]
