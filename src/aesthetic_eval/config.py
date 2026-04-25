import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class LoadedConfig:
    base: Dict[str, Any]
    model: Dict[str, Any]


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return data


def parse_device_map(value: Any) -> Any:
    if isinstance(value, str):
        v = value.strip()
        if v in {"auto", "balanced", "sequential"}:
            return v
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return value
    return value


def merge_configs(base_config_path: str, model_config_path: str) -> LoadedConfig:
    base = load_yaml(base_config_path)
    model = load_yaml(model_config_path)

    merged_base = copy.deepcopy(base)
    merged_model = copy.deepcopy(model)

    if "device_map" in merged_model:
        merged_model["device_map"] = parse_device_map(merged_model["device_map"])

    return LoadedConfig(base=merged_base, model=merged_model)
