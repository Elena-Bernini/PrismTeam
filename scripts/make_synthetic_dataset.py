"""Create small SYNTHETIC RPLAN (Graph2Plan pickle) and ResPlan (ResPlan.pkl) datasets (smoke tests only).

Useful to try the whole pipeline on a new machine before the real RPLAN files
are in place. By default it writes to a separate data root so it can never be
confused with (or overwrite) the real dataset:

    python scripts/make_synthetic_dataset.py                       # -> data_synthetic/
    python scripts/run_all.py --mode preview --set paths.data_root=data_synthetic \
        --set paths.artifacts_root=artifacts_synthetic --set paths.outputs_root=outputs_synthetic
"""

import argparse

import sys
from pathlib import Path

# Make `import prism` work however the script is launched (python scripts/x.py,
# VS Code "Run Python File", python -m scripts.x) even without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.common.config import resolve_path
from prism.datasets.synthetic import write_synthetic_resplan, write_synthetic_rplan


def main():
    """Write the synthetic RPLAN pickle + snapshots and the synthetic ResPlan pickle + split."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data_synthetic")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = resolve_path(args.data_root) / "raw" / "rplan" / "Interface" / "static" / "Data"
    info = write_synthetic_rplan(root / "data_train_converted.pkl", root / "snapshot_train", n=args.n, seed=args.seed)
    print(info)
    print(write_synthetic_resplan(resolve_path(args.data_root) / "raw" / "resplan", n=args.n, seed=args.seed + 1))


if __name__ == "__main__":
    main()
