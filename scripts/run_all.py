"""Run several phases in sequence with the same options (works on Windows, macOS and Linux).

    python scripts/run_all.py --mode preview                      # phases 01-12
    python scripts/run_all.py --mode preview --from 3 --to 5      # only 03..05
    python scripts/run_all.py --mode full --env cluster --pack    # --pack is passed to phase 05

Every other argument (``--set``, ``--device``, ``--n``, ``--env`` ...) is forwarded
to each phase. Phases that need PyTorch are skipped with a warning when it is
not installed (06, and the GNN / deep-vision parts of 07, 08, 10).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASES = {
    1: "01_inspect_dataset.py",
    2: "02_convert_to_layout_json.py",
    3: "03_build_graphs.py",
    4: "04_render_images.py",
    5: "05_statistics_and_splits.py",
    6: "06_train_gnn.py",
    7: "07_graph_embed_index.py",
    8: "08_vision_embed_index.py",
    9: "09_retrieval_full.py",
    10: "10_retrieval_partial.py",
    11: "11_evaluate.py",
    12: "12_compare_pipelines.py",
}
TORCH_ONLY = {6}


def torch_installed() -> bool:
    """True when PyTorch can be imported by this interpreter."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def main() -> None:
    """Parse the range of phases and run them one after the other."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--to", dest="end", type=int, default=12)
    parser.add_argument("--pack", action="store_true", help="Forwarded to phase 05.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failing phase.")
    args, forward = parser.parse_known_args()

    has_torch = torch_installed()
    summary = []
    for num in range(args.start, args.end + 1):
        script = PHASES[num]
        if num in TORCH_ONLY and not has_torch:
            print(f"\n### skip {script}: PyTorch is not installed")
            summary.append((script, "skipped (no torch)", 0.0))
            continue
        cmd = [sys.executable, str(HERE / script)] + forward
        if num == 5 and args.pack:
            cmd.append("--pack")
        print(f"\n### {' '.join(cmd)}")
        t0 = time.perf_counter()
        code = subprocess.call(cmd)
        dt = time.perf_counter() - t0
        summary.append((script, "ok" if code == 0 else f"FAILED ({code})", dt))
        if code != 0 and not args.keep_going:
            break
    print("\n" + "=" * 60)
    for script, status, dt in summary:
        print(f"  {script:32s} {status:20s} {dt:7.1f}s")
    if any(s.startswith("FAILED") for _, s, _ in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
