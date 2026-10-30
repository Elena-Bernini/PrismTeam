"""Phase 00 - check the Python environment (libraries, devices, data files).

Run this first on every machine (Pier's Windows+CUDA PC, Elena's M1 Mac, the
cluster). It reports which phases can run and writes a small report.

    python scripts/00_check_env.py
    python scripts/00_check_env.py --env cluster
"""

import importlib
import platform

import scripts._bootstrap as _bootstrap  # noqa: F401

from prism.common.cli import build_parser, setup
from prism.common.device import describe_device, resolve_device
from prism.common.io import write_json
from prism.common.report import HtmlReport
from prism.retrieval.index import faiss_available

PHASE = "00_check_env"
LIBS = ["numpy", "PIL", "matplotlib", "yaml", "tqdm", "networkx", "torch", "torchvision",
        "transformers", "timm", "faiss"]


def library_versions():
    """Return ``{library: version or None}`` for every library the project may use."""
    out = {}
    for name in LIBS:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:
            out[name] = None
    return out


def main():
    """Print and save the environment summary."""
    args = build_parser(__doc__).parse_args()
    cfg, ws = setup(args, PHASE)
    libs = library_versions()
    device = resolve_device(cfg["runtime"]["device"])
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": device,
        "device_description": describe_device(device),
        "faiss_backend": "faiss" if faiss_available() else "numpy (exact search fallback)",
        "libraries": libs,
        "rplan_pickle_found": ws.raw_path("rplan.pkl").exists(),
        "rplan_pickle_path": str(ws.raw_path("rplan.pkl")),
        "rplan_snapshots_found": ws.raw_path("rplan.snapshot_dir").exists(),
    }
    torch_ok = libs["torch"] is not None
    info["phases"] = {
        "01-05 preprocessing": "ok",
        "06 GNN training": "ok" if torch_ok else "needs torch",
        "07 graph index": "ok (descriptor)" + ("" if torch_ok else "; gnn needs torch"),
        "08 vision index": "ok (colorhist)" + (" + deep encoders" if torch_ok else "; deep encoders need torch"),
        "09-12 retrieval/eval": "ok",
    }
    for k, v in info.items():
        print(f"  {k}: {v}")
    out = ws.phase_dir(PHASE)
    write_json(out / "environment.json", info)
    rep = HtmlReport("00 - Environment check", f"{ws.dataset} · {ws.mode}")
    rep.kpis({"device": info["device_description"], "index backend": info["faiss_backend"],
              "python": info["python"], "RPLAN pickle": "found" if info["rplan_pickle_found"] else "MISSING"})
    rep.heading("Libraries")
    rep.table([{"library": k, "version": v or "not installed"} for k, v in libs.items()])
    rep.heading("What can run here")
    rep.table([{"phase": k, "status": v} for k, v in info["phases"].items()])
    if not info["rplan_pickle_found"]:
        rep.note(f"RPLAN pickle not found at {info['rplan_pickle_path']}. See README 'Where to put the data', "
                 "or generate a synthetic dataset with scripts/make_synthetic_dataset.py.", warn=True)
    rep.save(out / "report.html")


if __name__ == "__main__":
    main()
