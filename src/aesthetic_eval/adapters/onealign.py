from typing import Any, List, Tuple

import os
import sys

from PIL import Image

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample


class OneAlignAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.scorer = None
        self.device = str(self.model_cfg.get("device", "cuda:0"))

    def load(self) -> None:
        repo_root = str(self.model_cfg.get("onealign_repo_root", "/home/Hu_xuanwei/Q-Align"))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from q_align import QAlignAestheticScorer

        model_path = str(self.model_cfg.get("model_path", "q-future/one-align"))
        self.scorer = QAlignAestheticScorer(pretrained=model_path, device=self.device).eval()

    def build_prompt(self, sample: EvalSample) -> str:
        q = str(sample.question or "").strip()
        if q:
            return q
        return "How would you rate the aesthetics of this image?"

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        valid: List[EvalSample] = []
        images: List[Image.Image] = []
        prompts: List[str] = []

        for s in batch_samples:
            try:
                img = Image.open(s.image_resolved).convert("RGB")
                valid.append(s)
                images.append(img)
                prompts.append(self.build_prompt(s))
            except Exception:
                continue

        if not valid:
            return None, [], []
        return {"images": images}, valid, prompts

    def generate_batch(self, prepared: Any) -> List[str]:
        images: List[Image.Image] = prepared["images"]
        scores = self.scorer(images)
        return [f"{float(x):.6f}" for x in scores]
