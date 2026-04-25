from typing import Any, List, Tuple

import torch
from PIL import Image

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample


class QSITAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = str(self.model_cfg.get("device", "cuda:0"))
        self.infer_mode = str(self.model_cfg.get("infer_mode", "score")).lower().strip()
        self.generate_batch_size = int(self.model_cfg.get("generate_batch_size", 8))
        self.max_new_tokens = int(self.base_cfg.get("generation", {}).get("max_new_tokens", 128))
        self.do_sample = bool(self.base_cfg.get("generation", {}).get("do_sample", False))
        self.score_tokens = ["Excellent", "Good", "Fair", "Poor", "Bad"]
        self.score_weights = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float32)
        self.pad_token_id = None

    def load(self) -> None:
        from transformers import AutoProcessor, AutoTokenizer, LlavaOnevisionForConditionalGeneration

        model_path = str(self.model_cfg["model_path"])
        dtype_name = str(self.model_cfg.get("torch_dtype", "float16"))
        torch_dtype = getattr(torch, dtype_name)

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0
        tokenized = self.tokenizer(self.score_tokens)["input_ids"]
        self.score_token_ids = [ids[0] for ids in tokenized]

    def build_prompt(self, sample: EvalSample) -> str:
        q = str(sample.question or "").strip()
        if q:
            return q
        return "How would you rate the quality of this image?"

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        valid: List[EvalSample] = []
        prompts: List[str] = []
        records = []

        score_query = str(
            self.model_cfg.get(
                "score_query",
                "Assume you are an image quality evaluator. Your rating should be chosen from the following five "
                "categories: Excellent, Good, Fair, Poor, and Bad (from high to low). How would you rate the quality of this image?",
            )
        )

        for s in batch_samples:
            try:
                img = Image.open(s.image_resolved).convert("RGB")
                user_prompt = self.build_prompt(s)
                message = score_query if self.infer_mode == "score" else user_prompt
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            {"type": "image"},
                        ],
                    }
                ]
                chat_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                records.append({"image": img, "chat_prompt": chat_prompt})
                valid.append(s)
                prompts.append(user_prompt)
            except Exception:
                continue

        if not valid:
            return None, [], []
        return {"records": records}, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        outs: List[str] = []
        records = prepared["records"]

        if self.infer_mode != "score":
            for start in range(0, len(records), self.generate_batch_size):
                chunk = records[start:start + self.generate_batch_size]
                images = [r["image"] for r in chunk]
                prompts = [r["chat_prompt"] for r in chunk]

                inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt")
                inputs = {
                    k: (v.to(self.device, torch.float16) if k == "pixel_values" else v.to(self.device))
                    for k, v in inputs.items()
                }
                prompt_lens = inputs["attention_mask"].sum(dim=-1).tolist()

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.do_sample,
                        pad_token_id=self.pad_token_id,
                    )

                for i, seq in enumerate(output_ids):
                    start_idx = int(prompt_lens[i])
                    gen_ids = seq[start_idx:]
                    text = self.processor.decode(gen_ids, skip_special_tokens=True)
                    if "assistant" in text:
                        text = text.split("assistant")[-1]
                    outs.append(text.strip())
            return outs

        for rec in records:
            inputs = self.processor(
                images=rec["image"],
                text=rec["chat_prompt"],
                return_tensors="pt",
            ).to(self.device, torch.float16)

            if self.infer_mode == "score":
                prefix_text = str(self.model_cfg.get("score_prefix", "The quality of this image is "))
                prefix_ids = self.tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(self.device)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=-1)
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]) 

                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=1,
                        output_logits=True,
                        return_dict_in_generate=True,
                        do_sample=False,
                        pad_token_id=self.pad_token_id,
                    )
                last_logits = output.logits[-1][0]
                sel_logits = torch.tensor(
                    [float(last_logits[idx].item()) for idx in self.score_token_ids],
                    dtype=torch.float32,
                    device=self.score_weights.device,
                )
                probs = torch.softmax(sel_logits, dim=-1)
                score = float((probs * self.score_weights).sum().item())
                outs.append(f"{score:.6f}")
            else:
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.do_sample,
                        pad_token_id=self.pad_token_id,
                    )
                text = self.processor.decode(output_ids[0][2:], skip_special_tokens=True)
                if "assistant" in text:
                    text = text.split("assistant")[-1]
                outs.append(text.strip())

        return outs
