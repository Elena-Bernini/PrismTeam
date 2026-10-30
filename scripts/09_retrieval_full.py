"""Phase 09 - retrieval with complete floor plans as queries.

For every index built in phases 07/08 (or the ones given with ``--indexes``),
query plans from the ``retrieval.query_split`` split (default: test) retrieve
their top-K neighbours from the whole gallery (the query itself is excluded).
With ``retrieval.hybrid`` every (graph, vision) pair is also fused with
Reciprocal Rank Fusion (Elena's hybrid retrieval).

Results are written to ``outputs/.../09_retrieval_full/results/<pipeline>.jsonl``
and evaluated in phase 11.

    python scripts/09_retrieval_full.py --mode preview
    python scripts/09_retrieval_full.py --mode full --indexes graph_gnn vision_dinov2_base_render
"""

import random

import scripts._bootstrap as _bootstrap  # noqa: F401

from prism.common.cli import build_parser, setup
from prism.common.io import clean_dir, progress, write_jsonl
from prism.common.report import HtmlReport, finalize_report
from prism.datasets.schema import Taxonomy
from prism.datasets.store import LayoutStore, load_splits
from prism.evaluation.metrics import evaluate_pair, summarize
from prism.retrieval.fusion import reciprocal_rank_fusion
from prism.retrieval.index import VectorIndex, list_indexes
from prism.retrieval.visual import comparison_figure

PHASE = "09_retrieval_full"


def select_queries(cfg, ws, splits, available):
    """Fixed random sample of query ids from the configured split."""
    rcfg = cfg["retrieval"]
    n = int(rcfg["num_queries_preview"] if ws.mode == "preview" else rcfg["num_queries"])
    pool = [s for s in splits.get(rcfg["query_split"], []) if s in available] or sorted(available)
    rng = random.Random(int(cfg["runtime"]["seed"]))
    return sorted(rng.sample(pool, min(n, len(pool))))


def main():
    """Run every pipeline on the query set, save rankings, show comparisons."""
    parser = build_parser(__doc__)
    parser.add_argument("--indexes", nargs="+", default=None)
    args = parser.parse_args()
    cfg, ws = setup(args, PHASE)
    out = ws.phase_dir(PHASE)
    rcfg = cfg["retrieval"]
    k = int(rcfg["top_k"])
    taxonomy = Taxonomy(cfg, ws.dataset)
    store = LayoutStore(ws)
    splits = load_splits(ws)
    names = args.indexes or list_indexes(ws.indexes_root)
    if not names:
        raise RuntimeError("No index found. Run phases 07/08 first.")
    indexes = {n: VectorIndex.load(ws.index_dir(n), rcfg["backend"]) for n in names}
    available = set.intersection(*(set(ix.ids) for ix in indexes.values()))
    queries = select_queries(cfg, ws, splits, available)
    print(f"  {len(queries)} queries x {len(indexes)} indexes")

    rankings = {n: {} for n in indexes}
    deep = {n: {} for n in indexes}  # longer lists used for fusion
    for n, ix in indexes.items():
        for q in progress(queries, n):
            res = ix.search_by_id(q, max(k, 5 * k))
            deep[n][q] = res
            rankings[n][q] = res[:k]
    if rcfg.get("hybrid", True):
        graph_ix = [n for n in indexes if indexes[n].meta.get("pipeline") == "graph"]
        vision_ix = [n for n in indexes if indexes[n].meta.get("pipeline") == "vision"]
        for g in graph_ix:
            for v in vision_ix:
                hname = f"hybrid_{g}+{v}"
                rankings[hname] = {q: reciprocal_rank_fusion([deep[g][q], deep[v][q]], int(rcfg["rrf_k"]), k)
                                   for q in queries}

    res_dir = clean_dir(out / "results")
    for n, per_q in rankings.items():
        write_jsonl(res_dir / f"{n}.jsonl",
                    ({"query_id": q, "pipeline": n,
                      "results": [{"rank": i + 1, **r} for i, r in enumerate(per_q[q])]} for q in queries))

    # Quick look at architectural quality (the full evaluation is phase 11).
    summaries = {}

    def summary(sid):
        """Cached layout summary."""
        if sid not in summaries:
            summaries[sid] = summarize(store.graph(sid), taxonomy)
        return summaries[sid]

    rows = []
    for n, per_q in rankings.items():
        scores = [evaluate_pair(summary(q), summary(r["id"]))["architectural_score"] for q in queries for r in per_q[q]]
        rows.append({"pipeline": n, "queries": len(queries), f"mean architectural score @{k}": sum(scores) / max(1, len(scores))})

    rep = HtmlReport("09 - Retrieval with complete queries", f"{ws.dataset} · {ws.mode}")
    rep.kpis({"queries": len(queries), "pipelines": len(rankings), "top-k": k, "gallery": len(available)})
    rep.table(rows)
    rep.text("Each row of a figure is one pipeline: query render on the left, top results on the right.")
    for q in queries[: int(rcfg["grids"])]:
        rep.figure(comparison_figure(store, q, {n: per_q[q] for n, per_q in rankings.items()}, k=min(k, 6)),
                   f"Query {q}")
    finalize_report(rep, out, ws.mode, "If the rankings look sensible, run phase 11 to evaluate them.")


if __name__ == "__main__":
    main()
