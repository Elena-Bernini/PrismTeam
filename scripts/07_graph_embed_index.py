"""Phase 07 - graph pipeline: embed every floor plan and build the vector index.

Encoders (``graph.encoders`` or ``--encoders``):

- ``descriptor``  hand-crafted interpretable descriptor (no training), standardised
                  with statistics of the train split;
- ``gnn``         embeddings of the GNN trained in phase 06.

Each encoder produces ``artifacts/<dataset>/<mode>/indexes/graph_<encoder>/``.

    python scripts/07_graph_embed_index.py --mode preview
    python scripts/07_graph_embed_index.py --mode full --encoders descriptor gnn
"""

import random

import numpy as np

import scripts._bootstrap as _bootstrap  # noqa: F401

from prism.common.cli import Timer, build_parser, setup
from prism.common.device import resolve_device, torch_available
from prism.common.io import progress
from prism.common.report import HtmlReport, finalize_report
from prism.datasets.schema import Taxonomy
from prism.datasets.store import LayoutStore, load_splits
from prism.graph.descriptor import graph_descriptor
from prism.retrieval.index import VectorIndex, fit_scaler
from prism.retrieval.visual import retrieval_figure

PHASE = "07_graph_embed_index"


def main():
    """Build the requested graph indexes and show example neighbours."""
    parser = build_parser(__doc__)
    parser.add_argument("--encoders", nargs="+", default=None, help="descriptor and/or gnn")
    args = parser.parse_args()
    cfg, ws = setup(args, PHASE)
    out = ws.phase_dir(PHASE)
    taxonomy = Taxonomy(cfg, ws.dataset)
    store = LayoutStore(ws)
    splits = load_splits(ws)
    ids = store.graph_ids()
    rcfg = cfg["retrieval"]
    encoders = args.encoders or list(cfg["graph"]["encoders"])

    rep = HtmlReport("07 - Graph pipeline: embeddings and index", f"{ws.dataset} · {ws.mode}")
    built = {}
    for enc in encoders:
        if enc == "descriptor":
            with Timer("descriptor"):
                vecs, names = [], None
                for sid in progress(ids, "descriptor"):
                    v, names = graph_descriptor(store.graph(sid), store.layout(sid), taxonomy)
                    vecs.append(v)
                X = np.stack(vecs)
                train_rows = [i for i, s in enumerate(ids) if s in set(splits["train"])] or list(range(len(ids)))
                scaler = fit_scaler(X[train_rows])
                index = VectorIndex(ids, X, rcfg["metric"], rcfg["backend"], rcfg["index"], scaler,
                                    meta={"pipeline": "graph", "encoder": "descriptor", "feature_names": names})
        elif enc == "gnn":
            if not torch_available():
                print("  [skip] gnn: PyTorch not installed")
                rep.note("GNN index skipped: PyTorch is not installed on this machine.", warn=True)
                continue
            from prism.graph.gnn_encoder import GnnEncoder, checkpoint_path
            ckpt = checkpoint_path(ws.gnn_dir)
            if not ckpt.exists():
                print(f"  [skip] gnn: no checkpoint at {ckpt} (run phase 06)")
                rep.note(f"GNN index skipped: no checkpoint at {ckpt}. Run phase 06 first.", warn=True)
                continue
            with Timer("gnn embeddings"):
                encoder = GnnEncoder(ckpt, taxonomy, resolve_device(cfg["runtime"]["device"]))
                X = encoder.encode([store.graph(s) for s in progress(ids, "load graphs")])
                index = VectorIndex(ids, X, rcfg["metric"], rcfg["backend"], rcfg["index"], None,
                                    meta={"pipeline": "graph", "encoder": "gnn", "model_config": encoder.cfg})
        else:
            raise ValueError(f"Unknown graph encoder '{enc}'")
        name = f"graph_{enc}"
        index.save(ws.index_dir(name))
        built[name] = index
        print(f"  saved index {name}: {X.shape} ({index.backend})")

    rep.kpis({"layouts": len(ids), "indexes": ", ".join(built) or "none",
              "backend": next(iter(built.values())).backend if built else "-"})
    rng = random.Random(int(cfg["runtime"]["seed"]))
    queries = rng.sample(splits["test"] or ids, min(int(rcfg["grids"]), len(splits["test"] or ids)))
    for name, index in built.items():
        rep.heading(f"{name}  (dim {index.raw.shape[1]})")
        for q in queries:
            rep.figure(retrieval_figure(store, q, index.search_by_id(q, 7), name), f"{name} · query {q}")
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
