import copy
import json
import os
import re
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


_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")


def _expand_env_text(value: str) -> str:
    def _replace(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(3)
        env_value = os.environ.get(name)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        return match.group(0)

    return os.path.expandvars(_ENV_PATTERN.sub(_replace, value))


def expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_text(value)
    if isinstance(value, list):
        return [expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_values(item) for key, item in value.items()}
    return value


def merge_configs(base_config_path: str, model_config_path: str) -> LoadedConfig:
    base = expand_env_values(load_yaml(base_config_path))
    model = expand_env_values(load_yaml(model_config_path))

    merged_base = copy.deepcopy(base)
    merged_model = copy.deepcopy(model)

    if "device_map" in merged_model:
        merged_model["device_map"] = parse_device_map(merged_model["device_map"])

    return LoadedConfig(base=merged_base, model=merged_model)
