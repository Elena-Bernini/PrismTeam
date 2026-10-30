"""Phase 10 - partial layout retrieval (incomplete queries).

For each query plan some rooms are removed (``partial.strategy``; boundary and
front door are always kept). The partial plan is then encoded by EVERY
pipeline in the same space as its index:

- graph descriptor / GNN: partial graph (visible rooms, door node, recomputed door hops);
- vision: partial render (visible rooms coloured, removed rooms grey).

Stage 1 retrieves ``partial.candidate_pool`` complete plans from the index;
stage 2 re-ranks them with explicit compatibility (room / adjacency / relation /
grid coverage, size, front door, vector similarity). The original complete
plan is a proxy ground truth: we report whether it is recovered in the top-K
before and after re-ranking. Hybrid = RRF of a graph and a vision pool, then re-ranking.

    python scripts/10_retrieval_partial.py --mode preview
    python scripts/10_retrieval_partial.py --mode full --set partial.strategy=keep_connected --set partial.keep_count=3
"""

import hashlib
import random

import numpy as np

import scripts._bootstrap as _bootstrap  # noqa: F401

from prism.common.cli import build_parser, setup
from prism.common.device import resolve_device, torch_available
from prism.common.io import clean_dir, progress, write_csv, write_json, write_jsonl
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import grouped_bars
from prism.datasets.schema import Taxonomy
from prism.datasets.store import LayoutStore, load_splits
from prism.evaluation.metrics import summarize
from prism.graph.build import subgraph, sublayout
from prism.partial.compatibility import compatibility
from prism.partial.masking import choose_visible_rooms
from prism.retrieval.fusion import reciprocal_rank_fusion
from prism.retrieval.index import VectorIndex, list_indexes
from prism.retrieval.query_encoders import make_query_encoder
from prism.retrieval.visual import retrieval_figure
from prism.vision.render import render_layout

PHASE = "10_retrieval_partial"


def query_rng(seed, sid):
    """Per-query RNG so that every pipeline sees exactly the same partial query."""
    return random.Random(int(hashlib.md5(f"{seed}:{sid}".encode()).hexdigest()[:8], 16))


def rank_of(sid, ranking):
    """1-based rank of ``sid`` in a ranking (None if absent)."""
    for i, r in enumerate(ranking, start=1):
        if r["id"] == sid:
            return i
    return None


def rerank(q_summary, pool, summary_fn, weights, k):
    """Stage 2: sort the candidate pool by compatibility with the partial query."""
    scored = []
    for r in pool:
        comp = compatibility(q_summary, summary_fn(r["id"]), r["score"], weights)
        scored.append({"id": r["id"], "score": comp["compatibility_score"], "stage1_score": r["score"], **comp})
    scored.sort(key=lambda x: (x["compatibility_score"], x["room_coverage"], x["adjacency_coverage"],
                               x["stage1_score"]), reverse=True)
    return scored


def main():
    """Build partial queries, run every pipeline, re-rank, summarise."""
    parser = build_parser(__doc__)
    parser.add_argument("--indexes", nargs="+", default=None)
    args = parser.parse_args()
    cfg, ws = setup(args, PHASE)
    out = ws.phase_dir(PHASE)
    pcfg, rcfg = cfg["partial"], cfg["retrieval"]
    k, pool_size = int(rcfg["top_k"]), int(pcfg["candidate_pool"])
    taxonomy = Taxonomy(cfg, ws.dataset)
    store = LayoutStore(ws)
    splits = load_splits(ws)
    device = resolve_device(cfg["runtime"]["device"])

    names = args.indexes or list_indexes(ws.indexes_root)
    indexes, encoders = {}, {}
    for n in names:
        ix = VectorIndex.load(ws.index_dir(n), rcfg["backend"])
        if (ix.meta.get("encoder") != "descriptor" and ix.meta.get("encoder") != "colorhist") and not torch_available():
            print(f"  [skip] {n}: needs PyTorch to encode partial queries")
            continue
        indexes[n] = ix
        encoders[n] = make_query_encoder(ix.meta, ws, cfg, taxonomy, device)
    if not indexes:
        raise RuntimeError("No usable index. Run phases 07/08 first.")

    available = set.intersection(*(set(ix.ids) for ix in indexes.values()))
    nq = int(rcfg["num_queries_preview"] if ws.mode == "preview" else rcfg["num_queries"])
    pool_ids = [s for s in splits.get(rcfg["query_split"], []) if s in available] or sorted(available)
    queries = sorted(random.Random(int(cfg["runtime"]["seed"])).sample(pool_ids, min(nq, len(pool_ids))))

    # Build the partial queries once.
    partial = {}
    for q in queries:
        layout, graph = store.layout(q), store.graph(q)
        vis, rem = choose_visible_rooms(layout, graph, pcfg, taxonomy, query_rng(cfg["runtime"]["seed"], q))
        pg = subgraph(graph, vis)
        partial[q] = {"visible": sorted(vis), "removed": sorted(rem), "layout": sublayout(layout, vis),
                      "graph": pg, "summary": summarize(pg, taxonomy)}

    summaries = {}

    def summary_fn(sid):
        """Cached summary of a complete gallery plan."""
        if sid not in summaries:
            summaries[sid] = summarize(store.graph(sid), taxonomy)
        return summaries[sid]

    stage1 = {}
    for n, ix in indexes.items():
        vecs = encoders[n].encode([store.layout(q) for q in queries], [partial[q]["graph"] for q in queries],
                                  [partial[q]["visible"] for q in queries])
        res = ix.search(vecs, pool_size)
        stage1[n] = dict(zip(queries, res))
    if rcfg.get("hybrid", True):
        g_ix = [n for n in indexes if indexes[n].meta.get("pipeline") == "graph"]
        v_ix = [n for n in indexes if indexes[n].meta.get("pipeline") == "vision"]
        for g in g_ix:
            for v in v_ix:
                stage1[f"hybrid_{g}+{v}"] = {q: reciprocal_rank_fusion([stage1[g][q], stage1[v][q]],
                                                                        int(rcfg["rrf_k"]), pool_size) for q in queries}

    res_dir = clean_dir(out / "results")
    per_query, records = [], {}
    for n, per_q in stage1.items():
        recs = []
        for q in progress(queries, f"rerank {n}"):
            s1 = per_q[q]
            s2 = rerank(partial[q]["summary"], s1, summary_fn, pcfg["weights"], k) if pcfg.get("rerank", True) else s1
            r1, r2 = rank_of(q, s1), rank_of(q, s2)
            top = s2[:k]
            per_query.append({
                "pipeline": n, "query_id": q, "visible_rooms": len(partial[q]["visible"]),
                "removed_rooms": len(partial[q]["removed"]),
                "original_rank_stage1": r1, "original_rank_reranked": r2,
                "recovered_at_k_stage1": float(r1 is not None and r1 <= k),
                "recovered_at_k_reranked": float(r2 is not None and r2 <= k),
                "recovered_in_pool": float(r1 is not None),
                "mean_topk_compatibility": float(np.mean([t.get("compatibility_score", 0) for t in top])) if top else 0.0,
                "top1_compatibility": float(top[0].get("compatibility_score", 0)) if top else 0.0,
                "hard_constraint_rate_topk": float(np.mean([t.get("hard_constraints_satisfied", 0) for t in top])) if top else 0.0,
                "mean_topk_door_compatibility": float(np.mean([t.get("door_compatibility", 0) for t in top])) if top else 0.0,
            })
            recs.append({"query_id": q, "pipeline": n, "visible_room_ids": partial[q]["visible"],
                         "removed_room_ids": partial[q]["removed"],
                         "stage1": [{"rank": i + 1, **r} for i, r in enumerate(s1[:k])],
                         "results": [{"rank": i + 1, **r} for i, r in enumerate(top)]})
        records[n] = recs
        write_jsonl(res_dir / f"{n}.jsonl", recs)

    summary_rows = []
    for n in stage1:
        rows = [r for r in per_query if r["pipeline"] == n]
        summary_rows.append({
            "pipeline": n, "queries": len(rows),
            f"recall@{k} stage1": float(np.mean([r["recovered_at_k_stage1"] for r in rows])),
            f"recall@{k} reranked": float(np.mean([r["recovered_at_k_reranked"] for r in rows])),
            f"in pool@{pool_size}": float(np.mean([r["recovered_in_pool"] for r in rows])),
            "mean top-k compatibility": float(np.mean([r["mean_topk_compatibility"] for r in rows])),
            "hard constraint rate": float(np.mean([r["hard_constraint_rate_topk"] for r in rows])),
            "door compatibility": float(np.mean([r["mean_topk_door_compatibility"] for r in rows])),
        })
    write_csv(out / "partial_metrics_by_query.csv", per_query)
    write_csv(out / "partial_summary.csv", summary_rows)
    write_json(out / "partial_summary.json", {"config": pcfg, "top_k": k, "summary": summary_rows})

    rep = HtmlReport("10 - Partial layout retrieval", f"{ws.dataset} · {ws.mode}")
    rep.kpis({"queries": len(queries), "strategy": pcfg["strategy"], "pipelines": len(stage1),
              "candidate pool": pool_size, "mean removed rooms": float(np.mean([len(p["removed"]) for p in partial.values()]))})
    rep.table(summary_rows)
    rep.figure(grouped_bars([r["pipeline"] for r in summary_rows],
                            {"stage 1": [r[f"recall@{k} stage1"] for r in summary_rows],
                             "re-ranked": [r[f"recall@{k} reranked"] for r in summary_rows]},
                            f"Original plan recovered in top-{k}", "share of queries"))
    rep.text("Figures: the partial query (grey = removed rooms, yellow = front door) followed by the re-ranked "
             "complete plans; the caption shows compatibility score and '(original)' marks the source plan.")
    for n, recs in records.items():
        rep.heading(n, 3)
        for rec in recs[: int(rcfg["grids"])]:
            q = rec["query_id"]
            qimg = render_layout(store.layout(q), taxonomy, cfg["render"], visible_room_ids=rec["visible_room_ids"])
            extra = {r["id"]: f"room {r.get('room_coverage', 0):.2f} adj {r.get('adjacency_coverage', 0):.2f}"
                     for r in rec["results"]}
            rep.figure(retrieval_figure(store, q, rec["results"], f"{n} · removed {rec['removed_room_ids']}",
                                        query_image=qimg, k=min(k, 6), extra=extra))
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
