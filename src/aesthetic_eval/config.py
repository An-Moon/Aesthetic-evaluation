import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_config_value(value: Any, *, source: str) -> Any:
    """Expand environment variables and user paths without silent fallbacks."""
    if isinstance(value, dict):
        return {key: _expand_config_value(item, source=source) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_value(item, source=source) for item in value]
    if not isinstance(value, str):
        return value

    missing = sorted({name for name in _ENV_PATTERN.findall(value) if name not in os.environ})
    if missing:
        raise ValueError(
            f"Unresolved required environment variable(s) {missing} in config {source}"
        )
    expanded = os.path.expandvars(value)
    if "${" in expanded:
        raise ValueError(f"Unresolved environment expression in config {source}: {value}")
    return os.path.expanduser(expanded)


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
    return _expand_config_value(data, source=str(p))


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
