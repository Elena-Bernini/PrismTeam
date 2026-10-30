"""Phase 11 - evaluate the rankings of phase 09 (complete queries).

For every (query, retrieved) pair: room-type / count / area similarity,
adjacency and relation similarity, grid similarity, constraint satisfaction,
front-door agreement and Pier's architectural score.

Recall@K needs a notion of relevance. A gallery plan is RELEVANT to a query
when it has the same unified room composition (same number of bedrooms,
bathrooms, kitchens, ...) AND an adjacency similarity >= ``evaluation.
relevance_adjacency_tau``. We report Recall@K = hits / min(K, |relevant|),
Hit@K, Precision@K and MRR. Optional: graph edit distance on a few pairs
(``--set evaluation.ged.enabled=true``).

    python scripts/11_evaluate.py --mode preview
    python scripts/11_evaluate.py --mode full
"""

from collections import defaultdict

import numpy as np

import scripts._bootstrap as _bootstrap  # noqa: F401

from prism.common.cli import build_parser, setup
from prism.common.io import progress, read_jsonl, write_csv, write_json
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import grouped_bars, plt
from prism.datasets.schema import Taxonomy
from prism.datasets.store import LayoutStore
from prism.evaluation.metrics import evaluate_pair, graph_edit_distance, ranking_metrics, relevant_set, summarize

PHASE = "11_evaluate"
KEY_METRICS = ["architectural_score", "room_type_count_similarity", "adjacency_type_similarity",
               "relation_triplet_similarity", "room_area_by_type_similarity", "constraint_score",
               "door_score", "same_composition"]


def mean_of(rows, key):
    """Mean of a numeric column, ignoring NaN/None."""
    vals = [float(r[key]) for r in rows if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    """Compute metrics for every pipeline and write CSV/JSON/HTML outputs."""
    args = build_parser(__doc__).parse_args()
    cfg, ws = setup(args, PHASE)
    out = ws.phase_dir(PHASE)
    taxonomy = Taxonomy(cfg, ws.dataset)
    store = LayoutStore(ws)
    k = int(cfg["retrieval"]["top_k"])
    tau = float(cfg["evaluation"]["relevance_adjacency_tau"])
    res_dir = ws.outputs / "09_retrieval_full" / "results"
    files = sorted(res_dir.glob("*.jsonl"))
    if not files:
        raise RuntimeError(f"No rankings in {res_dir}. Run phase 09 first.")

    gallery = {sid: summarize(store.graph(sid), taxonomy) for sid in progress(store.graph_ids(), "summaries")}
    buckets = defaultdict(set)
    for sid, s in gallery.items():
        buckets[s.composition_signature].add(sid)

    per_result, per_query = [], []
    ged_cfg = cfg["evaluation"]["ged"]
    ged_budget = int(ged_cfg["max_pairs"]) if ged_cfg["enabled"] else 0
    for f in files:
        recs = read_jsonl(f)
        pipeline = f.stem
        for rec in progress(recs, pipeline):
            q = rec["query_id"]
            qs = gallery[q]
            relevant = relevant_set(qs, gallery, buckets, tau)
            ranked = [r["id"] for r in rec["results"]][:k]
            rm = ranking_metrics(ranked, relevant, k)
            rows = []
            for r in rec["results"][:k]:
                m = evaluate_pair(qs, gallery[r["id"]])
                row = {"pipeline": pipeline, "query_id": q, "rank": r["rank"], "retrieved_id": r["id"],
                       "vector_score": r.get("score"), "relevant": int(r["id"] in relevant), **m}
                if ged_budget > 0:
                    row["ged_normalised"] = graph_edit_distance(store.graph(q), store.graph(r["id"]),
                                                                float(ged_cfg["timeout_s"]))
                    ged_budget -= 1
                rows.append(row)
            per_result.extend(rows)
            qrow = {"pipeline": pipeline, "query_id": q, **rm}
            for key in KEY_METRICS:
                qrow[f"mean_{key}"] = mean_of(rows, key)
            per_query.append(qrow)

    pipelines = sorted({r["pipeline"] for r in per_query})
    summary = []
    for p in pipelines:
        qs = [r for r in per_query if r["pipeline"] == p]
        rs = [r for r in per_result if r["pipeline"] == p]
        row = {"pipeline": p, "queries": len(qs),
               f"recall@{k}": mean_of(qs, "recall_at_k"), f"hit@{k}": mean_of(qs, "hit_at_k"),
               f"precision@{k}": mean_of(qs, "precision_at_k"), "mrr": mean_of(qs, "mrr"),
               "queries_with_relevant": sum(1 for r in qs if r["num_relevant"] > 0)}
        for key in KEY_METRICS:
            row[key] = mean_of(rs, key)
        if any("ged_normalised" in r for r in rs):
            row["ged_normalised"] = mean_of(rs, "ged_normalised")
        summary.append(row)
    by_rank = []
    for p in pipelines:
        for rank in range(1, k + 1):
            rs = [r for r in per_result if r["pipeline"] == p and r["rank"] == rank]
            if rs:
                by_rank.append({"pipeline": p, "rank": rank, "architectural_score": mean_of(rs, "architectural_score"),
                                "relevant_rate": mean_of(rs, "relevant")})

    write_csv(out / "metrics_per_result.csv", per_result)
    write_csv(out / "metrics_by_query.csv", per_query)
    write_csv(out / "metrics_by_rank.csv", by_rank)
    write_csv(out / "pipeline_summary.csv", summary)
    write_json(out / "pipeline_summary.json", {"top_k": k, "relevance_tau": tau, "summary": summary})

    rep = HtmlReport("11 - Evaluation of complete-query retrieval", f"{ws.dataset} · {ws.mode}")
    rep.kpis({"pipelines": len(pipelines), "queries": len(per_query) // max(1, len(pipelines)), "top-k": k,
              "relevance": f"same composition & adjacency >= {tau}"})
    rep.table(summary)
    rep.figure(grouped_bars(KEY_METRICS + [f"recall@{k}"], {r["pipeline"]: [r[m] for m in KEY_METRICS + [f"recall@{k}"]]
                                                           for r in summary}, "Mean metrics over top-K results"))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for p in pipelines:
        rows = [r for r in by_rank if r["pipeline"] == p]
        ax.plot([r["rank"] for r in rows], [r["architectural_score"] for r in rows], marker="o", label=p)
    ax.set_xlabel("rank")
    ax.set_ylabel("architectural score")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    rep.figure(fig, "Quality should decrease with rank for a good retriever")
    rep.note("The vector similarity of different pipelines is not comparable; compare the architectural metrics and "
             "Recall@K instead.")
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
