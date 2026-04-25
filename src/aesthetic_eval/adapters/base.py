from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from aesthetic_eval.data import EvalSample


class BaseAdapter(ABC):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        self.base_cfg = base_cfg
        self.model_cfg = model_cfg

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, sample: EvalSample) -> str:
        raise NotImplementedError

    @abstractmethod
    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        raise NotImplementedError

    @abstractmethod
    def generate_batch(self, prepared: Any) -> List[str]:
        raise NotImplementedError
