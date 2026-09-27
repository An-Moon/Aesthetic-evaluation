from typing import Dict, List

import torch
from PIL import Image, ImageFile

from aesthetic_score_eval.adapters.base import BaseScoreAdapter
from aesthetic_score_eval.data import ScoreSample

ImageFile.LOAD_TRUNCATED_IMAGES = True


class QSITScoreAdapter(BaseScoreAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = str(model_cfg.get("device", "cuda:0"))
        self.score_tokens = list(model_cfg.get("score_tokens", ["Excellent", "Good", "Fair", "Poor", "Bad"]))
        self.score_weights = torch.tensor(
            [float(x) for x in model_cfg.get("score_weights", [1.0, 0.75, 0.5, 0.25, 0.0])],
            dtype=torch.float32,
        )
        self.score_query = str(
            model_cfg.get(
                "score_query",
                "Assume you are an image aesthetics evaluator. Your rating should be chosen from the following "
                "five categories: Excellent, Good, Fair, Poor, and Bad (from high to low). "
                "How would you rate the aesthetics of this image?",
            )
        )
        self.score_prefix = str(model_cfg.get("score_prefix", "The aesthetics of this image is "))
        self.pad_token_id = None

    def _resolve_dtype(self):
        dtype_name = str(self.model_cfg.get("torch_dtype", "float16")).lower().strip()
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float32":
            return torch.float32
        return torch.float16

    def load(self) -> None:
        try:
            from transformers import AutoProcessor, AutoTokenizer, LlavaOnevisionForConditionalGeneration
        except ImportError as e:
            raise RuntimeError(
                "Q-SiT requires transformers with LlavaOnevisionForConditionalGeneration. "
                "Older LLaVA-style environments often cannot load this architecture. "
                "Use a Transformers >=4.57 runtime for Q-SiT, or test a cloned unified env "
                "with a newer transformers version before replacing the shared llava env."
            ) from e

        model_path = str(self.model_cfg["model_path"])
        torch_dtype = self._resolve_dtype()
        trust_remote_code = bool(self.model_cfg.get("trust_remote_code", True))

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=bool(self.model_cfg.get("low_cpu_mem_usage", True)),
            trust_remote_code=trust_remote_code,
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.tokenizer.padding_side = "left"
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        tokenized = self.tokenizer(self.score_tokens)["input_ids"]
        self.score_token_ids = [ids[0] for ids in tokenized]

    def _build_prompt(self) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.score_query},
                    {"type": "image"},
                ],
            }
        ]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True)

    def score_batch(self, batch_samples: List[ScoreSample]) -> List[Dict]:
        outputs: List[Dict] = []
        dtype = self._resolve_dtype()
        prompt = self._build_prompt()

        for sample in batch_samples:
            try:
                image = Image.open(sample.image_resolved).convert("RGB")
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, dtype)
                prefix_ids = self.tokenizer(self.score_prefix, return_tensors="pt")["input_ids"].to(self.device)
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
                logits = torch.tensor(
                    [float(last_logits[idx].item()) for idx in self.score_token_ids],
                    dtype=torch.float32,
                )
                probs = torch.softmax(logits, dim=-1)
                raw_score = float((probs * self.score_weights).sum().item())
                raw_response = {
                    tok: {"logit": float(logit), "prob": float(prob)}
                    for tok, logit, prob in zip(self.score_tokens, logits.tolist(), probs.tolist())
                }
                outputs.append(
                    {
                        "raw_score": raw_score,
                        "raw_response": raw_response,
                        "parse_status": "ok",
                        "error": "",
                    }
                )
            except Exception as e:
                outputs.append({"raw_score": None, "raw_response": "", "parse_status": "error", "error": str(e)})

        return outputs
