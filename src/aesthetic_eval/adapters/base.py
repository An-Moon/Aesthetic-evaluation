from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
from typing import Any, List, Tuple

from aesthetic_eval.data import EvalSample


def require_directory(config: dict, key: str, *, component: str) -> str:
    value = str(config.get(key, "") or "").strip()
    if not value:
        raise ValueError(f"{component} requires model config key '{key}'")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{component} directory does not exist for '{key}': {path}")
    return str(path.resolve())


class BaseAdapter(ABC):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        self.base_cfg = base_cfg
        self.model_cfg = model_cfg
        self._text_context = self._load_text_context()

    def _load_text_context(self) -> dict:
        path_value = str(self.base_cfg.get("prompt", {}).get("text_context_file", "") or "").strip()
        if not path_value:
            return {}
        path = Path(path_value)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        supported_protocols = {
            "artimuse_8shot_text_v1": 8,
            "artimuse_16shot_text_v1": 16,
            "uniaa_description_1shot_text_v1": 1,
        }
        protocol_name = str(payload.get("protocol_name", ""))
        expected_examples = supported_protocols.get(protocol_name)
        if expected_examples is None or len(payload.get("examples", [])) != expected_examples:
            raise ValueError(f"Invalid text-context protocol: {path}")
        payload["resolved_path"] = str(path.resolve())
        payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return payload

    def format_prompt(self, sample: EvalSample, fallback: str) -> str:
        question = str(sample.question or "").strip() or fallback
        template = str(self.base_cfg.get("prompt", {}).get("template", "{question}"))
        question = template.format(question=question).strip()
        if not self._text_context:
            return question
        examples = self._text_context["examples"]
        if any(str(item.get("image", "")) == sample.image for item in examples):
            raise RuntimeError(f"Text-context leakage detected for image {sample.image}")
        blocks = [str(self._text_context["instruction"]).strip(), "Textual reference examples (no demonstration images):"]
        for index, item in enumerate(examples, 1):
            example_question = str(item.get("question", ""))
            example_reference = str(item.get("reference", ""))
            blocks.append(f"Example {index}\nQuestion: {example_question}\nReference answer: {example_reference}")
        blocks.append(f"Current image\nQuestion: {question}\nAnswer:")
        return "\n\n".join(blocks)

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
