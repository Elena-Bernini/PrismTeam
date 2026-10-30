# SHARED between prism-local and prism-cluster — keep in sync manually.
"""Small config loader and stage launcher helpers for Prism scripts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

STAGE_TO_SCRIPT = {
    "01": "scripts/01_preprocessing.py",
    "01_preprocessing": "scripts/01_preprocessing.py",
    "preprocessing": "scripts/01_preprocessing.py",
    "02": "scripts/02_retrieval_pipelines.py",
    "02_retrieval_pipelines": "scripts/02_retrieval_pipelines.py",
    "retrieval": "scripts/02_retrieval_pipelines.py",
    "03": "scripts/03_partial_layout_retrieval.py",
    "03_partial_layout_retrieval": "scripts/03_partial_layout_retrieval.py",
    "partial": "scripts/03_partial_layout_retrieval.py",
    "04": "scripts/04_evaluate_retrieval.py",
    "04_evaluate_retrieval": "scripts/04_evaluate_retrieval.py",
    "evaluate": "scripts/04_evaluate_retrieval.py",
    "05": "scripts/05_compare_pipeline_evaluations.py",
    "05_compare_pipeline_evaluations": "scripts/05_compare_pipeline_evaluations.py",
    "compare": "scripts/05_compare_pipeline_evaluations.py",
}


def _minimal_yaml_parser(text: str) -> Dict[str, Any]:
    """Parse the simple YAML subset used by the Prism config files."""
    result: Dict[str, Any] = {}
    stack: List[tuple[int, Any]] = [(-1, result)]

    def parse_scalar(value: str) -> Any:
        value = value.strip()
        if value in {"null", "None", "~", ""}:
            return None
        if value in {"true", "True"}:
            return True
        if value in {"false", "False"}:
            return False
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    current_key_for_list: Dict[int, str] = {}
    for raw_line in text.splitlines():
        if "#" in raw_line:
            raw_line = raw_line.split("#", 1)[0]
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("-"):
            item = parse_scalar(line[1:].strip())
            if not isinstance(parent, list):
                raise ValueError("YAML list item without list parent in Prism config")
            parent.append(item)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                # Look ahead style is not available, default to dict; room_zones uses PyYAML or fallback elsewhere.
                new_obj: Dict[str, Any] = {}
                parent[key] = new_obj
                stack.append((indent, new_obj))
            else:
                parent[key] = parse_scalar(value)
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, using PyYAML when available and a minimal fallback otherwise."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except Exception:
        return _minimal_yaml_parser(text)


def cli_name(key: str) -> str:
    """Convert a config key like processed_root into --processed-root."""
    return "--" + key.replace("_", "-")


def build_command(project_root: Path, stage: str, config_path: Path, extra_args: List[str]) -> List[str]:
    """Build the Python command for a Prism stage from a config file."""
    if stage not in STAGE_TO_SCRIPT:
        known = ", ".join(sorted(STAGE_TO_SCRIPT))
        raise ValueError(f"Unknown stage '{stage}'. Known stages: {known}")
    config = load_yaml(config_path)
    args = config.get("args", config)
    if not isinstance(args, Mapping):
        raise ValueError(f"Config file {config_path} must contain a mapping or an 'args' mapping.")

    script_path = project_root / STAGE_TO_SCRIPT[stage]
    command = [sys.executable, str(script_path)]
    for key, value in args.items():
        if value is None or value is False:
            continue
        if isinstance(value, list):
            for item in value:
                command.extend([cli_name(key), str(item)])
        elif value is True:
            command.append(cli_name(key))
        else:
            command.extend([cli_name(key), str(value)])
    command.extend(extra_args)
    return command


def run_stage(project_root: Path, stage: str, config_path: Path, extra_args: List[str]) -> int:
    """Execute one Prism stage from config and return its process return code."""
    command = build_command(project_root, stage, config_path, extra_args)
    print("Running:", " ".join(command))
    return subprocess.call(command, cwd=str(project_root))
