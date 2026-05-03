from abc import ABC, abstractmethod
from typing import Dict, List

from aesthetic_score_eval.data import ScoreSample


class BaseScoreAdapter(ABC):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        self.base_cfg = base_cfg
        self.model_cfg = model_cfg

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def score_batch(self, batch_samples: List[ScoreSample]) -> List[Dict]:
        """Return one dict per sample: {'raw_score': float|None, 'error': str}"""
        raise NotImplementedError
