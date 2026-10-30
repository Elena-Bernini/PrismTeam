"""Configuration loading and merging.

The configuration is a plain nested ``dict`` loaded from YAML files. Values are
merged in this order (later wins): ``configs/default.yaml``, the environment
file ``configs/env/<env>.yaml``, extra ``--config`` files and ``--set`` overrides.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

# Project root = three levels above this file: src/prism/common/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` and return it.

    Nested dictionaries are merged key by key; any other value in ``override``
    replaces the value in ``base``.
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary (an empty file returns ``{}``)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def parse_override_value(raw: str) -> Any:
    """Parse the right-hand side of a ``--set key=value`` override.

    YAML parsing is reused so that ``true``, ``3``, ``0.5``, ``null`` and
    ``[a, b]`` become the expected Python types.
    """
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def set_dotted(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``cfg['a']['b']['c'] = value`` given ``dotted_key='a.b.c'``."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def get_dotted(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Return ``cfg['a']['b']`` for ``dotted_key='a.b'`` or ``default`` if missing."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def load_config(
    env: Optional[str] = None,
    extra_files: Optional[Iterable[str]] = None,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the final configuration dictionary.

    Parameters
    ----------
    env:
        Environment name (``local`` or ``cluster``). Defaults to the
        ``PRISM_ENV`` environment variable, then ``local``.
    extra_files:
        Additional YAML files merged after the environment file.
    overrides:
        ``key.subkey=value`` strings merged last.
    """
    cfg = load_yaml(CONFIG_DIR / "default.yaml")

    env_name = env or os.environ.get("PRISM_ENV", "local")
    env_file = CONFIG_DIR / "env" / f"{env_name}.yaml"
    if not env_file.exists():
        raise FileNotFoundError(f"Unknown environment '{env_name}': {env_file} not found")
    cfg = deep_merge(cfg, load_yaml(env_file))
    cfg["env"] = env_name

    for extra in extra_files or []:
        cfg = deep_merge(cfg, load_yaml(Path(extra)))

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set override '{item}', expected key=value")
        key, raw_value = item.split("=", 1)
        set_dotted(cfg, key.strip(), parse_override_value(raw_value.strip()))

    cfg["room_types"] = load_yaml(CONFIG_DIR / "room_types.yaml")
    return cfg


def resolve_path(path_like: Any) -> Path:
    """Resolve a config path: absolute paths are kept, relative ones are anchored at the project root."""
    path = Path(str(path_like)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)
