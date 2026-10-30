"""Shared command-line interface for every numbered script.

Every phase accepts the same core flags::

    --mode preview|full     preview = small fixed subset, full = whole dataset
    --n N                   preview subset size (default: preview.n in config)
    --env local|cluster     which configs/env/*.yaml to merge
    --config extra.yaml     extra config files (repeatable)
    --set key=value         config overrides (repeatable)
    --device auto|cpu|cuda|mps
    --dataset rplan|resplan
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np

from src.prism.common.config import load_config, resolve_path
from src.prism.common.device import configure_platform
from src.prism.common.workspace import Workspace


def build_parser(description: str) -> argparse.ArgumentParser:
    """Create an ``ArgumentParser`` pre-populated with the shared flags."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["preview", "full"], default=None,
                        help="preview: small fixed subset with visual report; full: whole dataset.")
    parser.add_argument("--n", type=int, default=None, help="Preview subset size.")
    parser.add_argument("--env", default=None, help="Environment config (local | cluster).")
    parser.add_argument("--config", action="append", default=[], help="Extra YAML config file.")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        help="Override a config value, e.g. --set graph.model=gat")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | mps")
    parser.add_argument("--dataset", default=None, help="rplan | resplan")
    return parser


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup(args: argparse.Namespace, phase: str) -> Tuple[Dict[str, Any], Workspace]:
    """Load configuration, apply CLI shortcuts, seed RNGs and print a run header.

    Returns the merged config dictionary and the resolved :class:`Workspace`.
    """
    configure_platform()
    cfg = load_config(env=args.env, extra_files=args.config, overrides=args.overrides)
    if args.mode:
        cfg["mode"] = args.mode
    if args.n:
        cfg["preview"]["n"] = int(args.n)
    if args.device:
        cfg["runtime"]["device"] = args.device
    if args.dataset:
        cfg["dataset"] = args.dataset

    # Model caches (useful on the cluster where /homes has a 100 GB quota).
    if cfg["paths"].get("hf_home"):
        os.environ["HF_HOME"] = str(resolve_path(cfg["paths"]["hf_home"]))
    if cfg["paths"].get("torch_home"):
        os.environ["TORCH_HOME"] = str(resolve_path(cfg["paths"]["torch_home"]))

    seed_everything(int(cfg["runtime"]["seed"]))
    ws = Workspace(cfg)
    print("=" * 78)
    print(f"Prism | {phase} | dataset={ws.dataset} mode={ws.mode} env={cfg['env']}")
    print(f"  processed: {ws.processed}")
    print(f"  artifacts: {ws.artifacts}")
    print(f"  outputs:   {ws.outputs}")
    print(f"  python:    {sys.version.split()[0]} on {sys.platform}")
    print("=" * 78)
    return cfg, ws


class Timer:
    """Context manager that prints how long a block took."""

    def __init__(self, label: str) -> None:
        """Store the label printed when the block ends."""
        self.label = label
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        """Start the clock."""
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Stop the clock and print the elapsed time."""
        self.elapsed = time.perf_counter() - self.start
        print(f"  [{self.label}] {self.elapsed:.1f}s")
