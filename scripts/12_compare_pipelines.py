"""Phase 12 - side-by-side comparison of pipelines and runs.

By default it merges the evaluation of this run (phase 11) and the partial
retrieval summary (phase 10) into one table and a few plots. Other runs can be
added with ``--evaluation path/to/11_evaluate:label`` (repeatable), e.g. to
compare a run with ResNet-50 against a run with DINOv2, or GAE vs contrastive.

    python scripts/12_compare_pipelines.py --mode preview
    python scripts/12_compare_pipelines.py --mode full \
        --evaluation outputs/rplan/full_gat/11_evaluate:gat
"""

from pathlib import Path

import scripts._bootstrap as _bootstrap  # noqa: F401

from prism.common.cli import build_parser, setup
from prism.common.config import resolve_path
from prism.common.io import read_csv, write_csv
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import grouped_bars

PHASE = "12_compare_pipelines"


def to_float(v):
    """Parse a CSV cell as float (NaN when not numeric)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def numeric_row(row):
    """Convert numeric CSV strings into floats so tables are formatted consistently."""
    out = {}
    for key, value in row.items():
        f = to_float(value)
        out[key] = value if (f != f) else (int(f) if key == "queries" else f)  # f != f -> NaN
    return out


def main():
    """Collect summaries and build the comparison report."""
    parser = build_parser(__doc__)
    parser.add_argument("--evaluation", action="append", default=[], help="path/to/11_evaluate[:label]")
    args = parser.parse_args()
    cfg, ws = setup(args, PHASE)
    out = ws.phase_dir(PHASE)
    k = int(cfg["retrieval"]["top_k"])

    sources = [(ws.outputs / "11_evaluate", "this run")]
    for item in args.evaluation:
        path, _, label = item.partition(":")
        sources.append((resolve_path(path), label or Path(path).parent.name))

    rows = []
    for path, label in sources:
        f = path / "pipeline_summary.csv"
        if not f.exists():
            print(f"  [skip] {f} not found")
            continue
        for r in read_csv(f):
            rows.append({"run": label, **numeric_row(r)})
    partial_rows = []
    pf = ws.outputs / "10_retrieval_partial" / "partial_summary.csv"
    if pf.exists():
        partial_rows = [numeric_row(r) for r in read_csv(pf)]
    if not rows and not partial_rows:
        raise RuntimeError("Nothing to compare: run phases 10 and 11 first.")

    write_csv(out / "comparison_full_queries.csv", rows)
    rep = HtmlReport("12 - Pipeline comparison", f"{ws.dataset} · {ws.mode}")
    if rows:
        rep.heading("Complete queries (phase 11)")
        rep.table(rows)
        metrics = ["architectural_score", "room_type_count_similarity", "adjacency_type_similarity",
                   "relation_triplet_similarity", "door_score", f"recall@{k}"]
        series = {f"{r['run']} · {r['pipeline']}": [to_float(r.get(m)) for m in metrics] for r in rows}
        rep.figure(grouped_bars(metrics, series, "Complete-query retrieval"))
        best = max(rows, key=lambda r: to_float(r.get("architectural_score")))
        rep.note(f"Highest architectural score: {best['run']} · {best['pipeline']} "
                 f"({to_float(best['architectural_score']):.3f}).")
    if partial_rows:
        rep.heading("Partial queries (phase 10)")
        rep.table(partial_rows)
        cols = [c for c in partial_rows[0].keys() if c not in ("pipeline", "queries")]
        rep.figure(grouped_bars(cols, {r["pipeline"]: [to_float(r[c]) for c in cols] for r in partial_rows},
                                "Partial-query retrieval"))
    rep.heading("How to read this")
    rep.text("Graph pipelines are expected to win on room composition, adjacency and relations; vision pipelines "
             "on overall shape and visual similarity. Hybrid (RRF) rows show whether the two views are complementary. "
             "For partial queries, compare recall before and after re-ranking.")
    finalize_report(rep, out, ws.mode, "When the full runs are done, use this report for the project write-up.")


if __name__ == "__main__":
    main()
