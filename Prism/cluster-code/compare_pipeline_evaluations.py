#!/usr/bin/env python3
"""
Compare retrieval evaluation outputs across multiple pipelines.

This script is intended to be run after:
  1. preprocessing.py
  2. retrieval_pipelines.py
  3. evaluate_retrieval.py

It reads one or more evaluation folders produced by evaluate_retrieval.py,
extracts per-result metrics, aggregates them by pipeline, and writes a compact
comparison report with CSV summaries and plots.

Example:
    python cluster-code/compare_pipeline_evaluations.py \
        --evaluation outputs/evaluation_resnet50:resnet50 \
        --evaluation outputs/evaluation_dinov2:dinov2 \
        --outputs-root outputs/evaluation_comparison \
        --clean
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import argparse
import csv
import json
import math
import shutil

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Metrics expected from evaluate_retrieval.py.
# The script is intentionally tolerant: if a metric is missing, it is skipped.
DEFAULT_METRICS = [
    "architectural_score",
    "room_type_similarity",
    "room_count_similarity",
    "area_similarity",
    "adjacency_similarity",
    "relation_similarity",
    "grid_similarity",
    "constraint_score",
    "hard_constraint_satisfied",
    "faiss_similarity",
]


@dataclass
class EvaluationSpec:
    """Input evaluation directory plus a short label used in reports."""

    root: Path
    label: str


def parse_evaluation_argument(value: str) -> EvaluationSpec:
    """
    Parse an evaluation argument.

    Supported forms:
      - outputs/evaluation_resnet50
      - outputs/evaluation_resnet50:resnet50
      - "C:/path/with spaces/evaluation:label"

    The split is performed from the right so Windows drive letters are safe.
    """
    if ":" in value:
        left, right = value.rsplit(":", 1)

        # If this looks like a Windows drive path without a user label, keep it as a path.
        # Example: C:\Users\...\evaluation
        if len(left) == 1 and left.isalpha():
            path = Path(value)
            label = path.name
        else:
            path = Path(left)
            label = right.strip() or path.name
    else:
        path = Path(value)
        label = path.name

    return EvaluationSpec(root=path, label=label)


def ensure_clean_dir(path: Path, clean: bool = False) -> None:
    """Create an output directory, optionally removing the previous content."""
    if clean and path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file as a list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None for missing/non-numeric values."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if math.isnan(number):
        return None

    return number


def mean(values: List[float]) -> float:
    """Compute a safe mean value."""
    if not values:
        return float("nan")

    return float(np.mean(values))


def std(values: List[float]) -> float:
    """Compute a safe standard deviation value."""
    if not values:
        return float("nan")

    return float(np.std(values))


def clean_label(label: str) -> str:
    """Normalize a label for filenames."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)


def display_pipeline_name(eval_label: str, pipeline_name: str) -> str:
    """
    Convert internal pipeline names into human-readable labels.

    Example:
      - evaluation_resnet50 + graph_pipeline  -> Graph
      - evaluation_resnet50 + vision_pipeline -> Vision ResNet50
      - evaluation_dinov2 + vision_pipeline   -> Vision DINOv2
    """
    normalized = eval_label.lower().replace("evaluation_", "").replace("retrieval_", "")

    if pipeline_name == "graph_pipeline":
        return "Graph"

    if pipeline_name == "vision_pipeline":
        if "dinov2" in normalized or "dino" in normalized:
            return "Vision DINOv2"
        if "resnet50" in normalized:
            return "Vision ResNet50"
        if "resnet18" in normalized:
            return "Vision ResNet18"
        if "vit" in normalized:
            return "Vision ViT"
        return f"Vision {eval_label}"

    return f"{eval_label} {pipeline_name}"


def load_evaluation_rows(spec: EvaluationSpec) -> List[Dict[str, Any]]:
    """
    Load per-result rows from one evaluation directory.

    Expected file:
      metrics_per_result.csv

    Each row is enriched with:
      - evaluation_label
      - original_pipeline
      - comparison_pipeline
    """
    metrics_path = spec.root / "metrics_per_result.csv"
    rows = read_csv_rows(metrics_path)

    enriched = []
    for row in rows:
        original_pipeline = row.get("pipeline", row.get("pipeline_name", "unknown_pipeline"))
        comparison_pipeline = display_pipeline_name(spec.label, original_pipeline)

        item = dict(row)
        item["evaluation_label"] = spec.label
        item["original_pipeline"] = original_pipeline
        item["comparison_pipeline"] = comparison_pipeline
        enriched.append(item)

    return enriched


def aggregate_by_pipeline(rows: List[Dict[str, Any]], metrics: List[str]) -> List[Dict[str, Any]]:
    """Aggregate per-result rows into one summary row per comparison pipeline."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["comparison_pipeline"]].append(row)

    summaries = []

    for pipeline_name, pipeline_rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "pipeline": pipeline_name,
            "num_results": len(pipeline_rows),
            "num_queries": len(set(row.get("query_id", "") for row in pipeline_rows)),
        }

        for metric in metrics:
            values = [
                value
                for value in (to_float(row.get(metric)) for row in pipeline_rows)
                if value is not None
            ]

            if not values:
                continue

            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = std(values)

        summaries.append(summary)

    return summaries


def aggregate_by_rank(rows: List[Dict[str, Any]], metric: str = "architectural_score") -> List[Dict[str, Any]]:
    """Aggregate a selected metric by rank and pipeline."""
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)

    for row in rows:
        pipeline = row["comparison_pipeline"]
        rank = to_float(row.get("rank"))
        value = to_float(row.get(metric))

        if rank is None or value is None:
            continue

        grouped[(pipeline, int(rank))].append(value)

    output = []
    for (pipeline, rank), values in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        output.append({
            "pipeline": pipeline,
            "rank": rank,
            f"{metric}_mean": mean(values),
            f"{metric}_std": std(values),
            "count": len(values),
        })

    return output


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to a CSV file."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    # Keep important fields first when present.
    preferred = ["pipeline", "rank", "num_results", "num_queries", "count"]
    fieldnames = [f for f in preferred if f in fieldnames] + [
        f for f in fieldnames if f not in preferred
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    """Write a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_bar_plot(summary_rows: List[Dict[str, Any]], output_path: Path, metric: str) -> None:
    """Create a bar plot comparing pipelines for one metric."""
    rows = [row for row in summary_rows if f"{metric}_mean" in row]

    if not rows:
        return

    labels = [row["pipeline"] for row in rows]
    values = [float(row[f"{metric}_mean"]) for row in rows]

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(labels)), 5))
    ax.bar(labels, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Pipeline")
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def make_rank_curve(rank_rows: List[Dict[str, Any]], output_path: Path, metric: str) -> None:
    """Create a rank curve plot for one metric."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    key = f"{metric}_mean"
    for row in rank_rows:
        if key in row:
            grouped[row["pipeline"]].append(row)

    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for pipeline, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["rank"]))
        x = [int(row["rank"]) for row in rows]
        y = [float(row[key]) for row in rows]
        ax.plot(x, y, marker="o", label=pipeline)

    ax.set_xlabel("Rank")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Retrieval Rank")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def format_value(value: Any) -> str:
    """Format numeric table values for Markdown."""
    number = to_float(value)
    if number is None:
        return ""

    return f"{number:.4f}"


def write_markdown_report(
    path: Path,
    evaluation_specs: List[EvaluationSpec],
    summary_rows: List[Dict[str, Any]],
    metrics: List[str],
) -> None:
    """Write a standalone Markdown comparison report."""
    lines: List[str] = []

    lines.append("# Retrieval Pipeline Evaluation Comparison")
    lines.append("")
    lines.append("This report compares retrieval pipelines using the evaluation outputs produced after preprocessing and retrieval.")
    lines.append("The comparison is based on per-result architectural metrics computed from the preprocessed layout and graph JSON files.")
    lines.append("")
    lines.append("## Input Evaluations")
    lines.append("")

    for spec in evaluation_specs:
        lines.append(f"- `{spec.label}`: `{spec.root}`")

    lines.append("")
    lines.append("## Summary Table")
    lines.append("")

    table_metrics = [
        "architectural_score",
        "room_type_similarity",
        "adjacency_similarity",
        "relation_similarity",
        "constraint_score",
        "hard_constraint_satisfied",
        "faiss_similarity",
    ]

    available_metrics = [
        metric for metric in table_metrics
        if any(f"{metric}_mean" in row for row in summary_rows)
    ]

    header = ["Pipeline"] + [metric.replace("_", " ").title() for metric in available_metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in summary_rows:
        values = [row["pipeline"]]
        for metric in available_metrics:
            values.append(format_value(row.get(f"{metric}_mean")))
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- `architectural_score` is the aggregate score used to summarize architectural retrieval quality.")
    lines.append("- `room_type_similarity` measures whether the retrieved layout has a similar room-type composition.")
    lines.append("- `adjacency_similarity` measures whether the retrieved layout preserves similar semantic room adjacencies.")
    lines.append("- `relation_similarity` measures whether spatial relation distributions are similar.")
    lines.append("- `constraint_score` and `hard_constraint_satisfied` measure whether simple architectural constraints are respected.")
    lines.append("- `faiss_similarity` is the vector-space similarity used for retrieval; it should not be interpreted as architectural quality by itself.")
    lines.append("")
    lines.append("## Recommended Use")
    lines.append("")
    lines.append("Use this comparison to decide which representation is more suitable for semantic architectural search.")
    lines.append("A visual encoder may produce high vector similarity while still retrieving layouts with different room composition or topology.")
    lines.append("A graph-based representation is expected to perform better when the target is room-level semantic and topological compatibility.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare retrieval evaluation folders generated by evaluate_retrieval.py."
    )
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        help=(
            "Evaluation directory, optionally followed by a label: "
            "outputs/evaluation_resnet50:resnet50. "
            "Use this argument multiple times."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs/evaluation_comparison"),
        help="Directory where comparison outputs are written.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before writing new results.",
    )
    args = parser.parse_args()

    specs = [parse_evaluation_argument(value) for value in args.evaluation]

    ensure_clean_dir(args.outputs_root, clean=args.clean)
    plots_dir = args.outputs_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for spec in specs:
        print(f"Loading evaluation: {spec.label} from {spec.root}")
        all_rows.extend(load_evaluation_rows(spec))

    summary_rows = aggregate_by_pipeline(all_rows, DEFAULT_METRICS)
    rank_rows = aggregate_by_rank(all_rows, metric="architectural_score")

    write_csv(args.outputs_root / "all_metrics_per_result.csv", all_rows)
    write_csv(args.outputs_root / "pipeline_summary.csv", summary_rows)
    write_csv(args.outputs_root / "rank_curve_architectural_score.csv", rank_rows)

    write_json(args.outputs_root / "pipeline_summary.json", summary_rows)

    write_markdown_report(
        path=args.outputs_root / "pipeline_comparison_report.md",
        evaluation_specs=specs,
        summary_rows=summary_rows,
        metrics=DEFAULT_METRICS,
    )

    for metric in [
        "architectural_score",
        "room_type_similarity",
        "adjacency_similarity",
        "constraint_score",
        "hard_constraint_satisfied",
    ]:
        make_bar_plot(
            summary_rows=summary_rows,
            output_path=plots_dir / f"{metric}_bar.png",
            metric=metric,
        )

    make_rank_curve(
        rank_rows=rank_rows,
        output_path=plots_dir / "architectural_score_rank_curve.png",
        metric="architectural_score",
    )

    print("")
    print("Comparison completed.")
    print(f"Summary CSV: {args.outputs_root / 'pipeline_summary.csv'}")
    print(f"Markdown report: {args.outputs_root / 'pipeline_comparison_report.md'}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
