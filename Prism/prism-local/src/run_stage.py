#!/usr/bin/env python3
"""Run a numbered Prism pipeline stage from a YAML configuration file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage_config import run_stage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Prism stage from config.")
    parser.add_argument("--stage", required=True, help="Stage key, e.g. 01_preprocessing or retrieval.")
    parser.add_argument("--config", required=True, type=Path, help="YAML config for the stage.")
    args, extra = parser.parse_known_args()
    raise SystemExit(run_stage(PROJECT_ROOT, args.stage, args.config, extra))


if __name__ == "__main__":
    main()
