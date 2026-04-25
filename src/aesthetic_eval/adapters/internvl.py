from typing import Any, List, Tuple

import torch
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample, load_and_resize_rgb


class InternVLAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.tokenizer = None
        self.to_tensor = transforms.ToTensor()
        self.infer_dtype = torch.float16

    def load(self) -> None:
        model_path = self.model_cfg["model_path"]
        dtype_name = str(self.model_cfg.get("torch_dtype", "float16"))
        torch_dtype = getattr(torch, dtype_name)
        self.infer_dtype = torch_dtype
        device_map = self.model_cfg.get("device_map", {"": 0})

        trust_remote_code = bool(self.model_cfg.get("trust_remote_code", True))
        use_fast = bool(self.model_cfg.get("use_fast_tokenizer", True))

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=use_fast,
            )
        except Exception:
            # Some local caches have a corrupted tokenizer.json for fast tokenizers.
            # Fallback to slow tokenizer to keep inference runnable.
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=bool(self.model_cfg.get("trust_remote_code", True)),
        ).eval()

    def build_prompt(self, sample: EvalSample) -> str:
        template = str(self.base_cfg.get("prompt", {}).get("template", "{question}"))
        prompt = template.format(question=sample.question).strip()
        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt
        return prompt

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        image_size = int(self.base_cfg.get("preprocess", {}).get("image_size", 448))
        valid: List[EvalSample] = []
        tensors = []
        prompts = []

        for s in batch_samples:
            try:
                img = load_and_resize_rgb(s.image_resolved, image_size)
                t = self.to_tensor(img)
                tensors.append(t)
                valid.append(s)
                prompts.append(self.build_prompt(s))
            except Exception:
                continue

        if not valid:
            return None, [], []

        stacked = torch.stack(tensors)
        return stacked, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        stacked, prompts = prepared
        device = next(self.model.parameters()).device
        stacked = stacked.to(device=device, dtype=self.infer_dtype, non_blocking=True)

        gen_cfg = self.base_cfg.get("generation", {})
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
        do_sample = bool(gen_cfg.get("do_sample", False))
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        outs: List[str] = []
        batch_err = ""
        chat_err = ""
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=self.infer_dtype):
                if hasattr(self.model, "batch_chat"):
                    try:
                        batch_outs = self.model.batch_chat(
                            self.tokenizer,
                            pixel_values=stacked,
                            questions=prompts,
                            generation_config=generation_config,
                            num_patches_list=[1] * len(prompts),
                        )
                        cleaned = [str(x).strip() for x in batch_outs]
                        if any(cleaned):
                            return cleaned
                    except Exception as e:
                        batch_err = str(e)

                for img, prompt in zip(stacked, prompts):
                    try:
                        resp = self.model.chat(
                            self.tokenizer,
                            pixel_values=img.unsqueeze(0),
                            question=prompt,
                            generation_config=generation_config,
                        )
                    except Exception as e1:
                        if not chat_err:
                            chat_err = f"question-call: {e1}"
                        try:
                            resp = self.model.chat(
                                self.tokenizer,
                                img.unsqueeze(0),
                                prompt,
                                generation_config,
                                history=[],
                            )
                        except Exception as e2:
                            if not chat_err:
                                chat_err = f"positional-call: {e2}"
                            resp = ""
                    outs.append(str(resp).strip())

        if not any(outs):
            raise RuntimeError(
                "InternVL returned all-empty outputs for a batch; "
                f"batch_chat_err={batch_err}; chat_err={chat_err}"
            )
        return outs
