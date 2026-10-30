#!/usr/bin/env python3
"""
Evaluate graph-based and vision-based floorplan retrieval results.

This script is intended to run after the RPLAN preprocessing step and after the
retrieval pipelines have produced top-k JSON files. It does not rebuild indexes,
it does not extract embeddings, and it does not require CUDA. It only reads the
preprocessed layout/graph JSON files and the retrieval result JSON files, then
computes architectural similarity metrics between each query floorplan and its
retrieved candidates.

Expected input structure:

    data/processed/rplan/
    ├── layouts_json/
    ├── graphs_json/
    └── retrieval/                  # optional, not required by this script

    outputs/retrieval_gpu_full/
    ├── graph_pipeline/
    │   └── <query_id>_top10.json
    ├── vision_pipeline/
    │   └── <query_id>_top10.json
    └── pipeline_comparison/

Typical usage:

    python cluster-code/evaluate_retrieval.py \
        --processed-root data/processed/rplan \
        --retrieval-root outputs/retrieval_gpu_full \
        --outputs-root outputs/evaluation_resnet50 \
        --top-k 10
"""

from __future__ import annotations

import logging

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Matplotlib is only used to save summary plots. The script still works without
# showing interactive windows, which makes it safe for cluster execution.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("04_evaluate_retrieval.py")


def print(*args: object, **kwargs: object) -> None:  # noqa: A001 - compatibility shim for legacy progress output
    """Route legacy progress output through logging."""
    sep = str(kwargs.get("sep", " "))
    logger.info(sep.join(str(arg) for arg in args))



# -----------------------------------------------------------------------------
# RPLAN / Graph2Plan semantic schema
# -----------------------------------------------------------------------------

ROOM_TYPE_ID_TO_NAME = {
    0: "LivingRoom",
    1: "MasterRoom",
    2: "Kitchen",
    3: "Bathroom",
    4: "DiningRoom",
    5: "ChildRoom",
    6: "StudyRoom",
    7: "SecondRoom",
    8: "GuestRoom",
    9: "Balcony",
    10: "Entrance",
    11: "Storage",
    12: "Wall-in",
}

ROOM_TYPE_NAMES = [ROOM_TYPE_ID_TO_NAME[i] for i in sorted(ROOM_TYPE_ID_TO_NAME)]

# For user-facing constraints, some RPLAN room types are grouped into broader
# functional concepts. This does not replace the original room type. It is only
# used for architectural constraints such as bedroom count.
BEDROOM_TYPES = {"MasterRoom", "SecondRoom", "GuestRoom", "ChildRoom"}


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class RetrievalItem:
    """One retrieved candidate for a query."""

    query_id: str
    retrieved_id: str
    rank: int
    similarity: Optional[float]
    distance: Optional[float]
    source_json: str


@dataclass
class LayoutData:
    """Cached structural information for one floorplan."""

    layout_id: str
    rooms: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Generic I/O helpers
# -----------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    """Read a JSON file using UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    """Write a JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    # Keep a stable column order: first columns from the first row, then any
    # additional columns encountered in later rows.
    fieldnames = list(rows[0].keys())
    known = set(fieldnames)
    for row in rows[1:]:
        for key in row.keys():
            if key not in known:
                fieldnames.append(key)
                known.add(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: Any) -> Optional[float]:
    """Convert a value to float when possible, otherwise return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------------
# Retrieval result parsing
# -----------------------------------------------------------------------------

def find_result_files(pipeline_dir: Path) -> List[Path]:
    """Find retrieval result JSON files in a pipeline output directory."""
    if not pipeline_dir.exists():
        return []

    # The retrieval pipeline usually writes files such as <query_id>_top10.json.
    # We avoid report/config files by requiring the JSON to contain a results list.
    candidates = sorted(pipeline_dir.glob("*.json"))
    result_files = []
    for path in candidates:
        try:
            obj = read_json(path)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("results"), list):
            result_files.append(path)
    return result_files


def parse_retrieval_file(path: Path, top_k: int) -> List[RetrievalItem]:
    """Parse one top-k retrieval JSON file into RetrievalItem objects."""
    obj = read_json(path)

    query_id = str(
        obj.get("query_id")
        or obj.get("query")
        or obj.get("id")
        or path.stem.split("_top")[0]
    )

    items: List[RetrievalItem] = []
    for index, result in enumerate(obj.get("results", [])[:top_k], start=1):
        retrieved_id = str(
            result.get("id")
            or result.get("retrieved_id")
            or result.get("layout_id")
            or result.get("sample_id")
        )

        if retrieved_id in {"None", ""}:
            continue

        rank = int(result.get("rank", index))
        similarity = safe_float(result.get("similarity", result.get("score")))
        distance = safe_float(result.get("distance", result.get("dist")))

        items.append(
            RetrievalItem(
                query_id=query_id,
                retrieved_id=retrieved_id,
                rank=rank,
                similarity=similarity,
                distance=distance,
                source_json=str(path),
            )
        )

    return items


def load_pipeline_results(retrieval_root: Path, pipeline_name: str, top_k: int) -> List[RetrievalItem]:
    """Load all retrieval results for one pipeline."""
    pipeline_dir = retrieval_root / pipeline_name
    files = find_result_files(pipeline_dir)

    items: List[RetrievalItem] = []
    for path in files:
        items.extend(parse_retrieval_file(path, top_k=top_k))

    return items


# -----------------------------------------------------------------------------
# Layout / graph loading
# -----------------------------------------------------------------------------

class LayoutStore:
    """Lazy loader for preprocessed layout and graph JSON files."""

    def __init__(self, processed_root: Path):
        self.processed_root = processed_root
        self.layout_dir = processed_root / "layouts_json"
        self.graph_dir = processed_root / "graphs_json"
        self._cache: Dict[str, LayoutData] = {}

    def get(self, layout_id: str) -> LayoutData:
        """Load one layout and graph, with caching."""
        layout_id = str(layout_id)
        if layout_id in self._cache:
            return self._cache[layout_id]

        layout_path = self.layout_dir / f"{layout_id}.json"
        graph_path = self.graph_dir / f"{layout_id}.json"

        if not layout_path.exists():
            raise FileNotFoundError(f"Missing layout JSON: {layout_path}")

        layout = read_json(layout_path)
        rooms = layout.get("rooms", [])

        # Prefer graph edges when available because they are the graph pipeline's
        # direct input. Fall back to layout edges for robustness.
        edges = layout.get("edges", [])
        if graph_path.exists():
            graph = read_json(graph_path)
            edges = graph.get("edges", edges)

        data = LayoutData(layout_id=layout_id, rooms=rooms, edges=edges)
        self._cache[layout_id] = data
        return data


# -----------------------------------------------------------------------------
# Room and graph descriptors
# -----------------------------------------------------------------------------

def get_room_type_id(room: Dict[str, Any]) -> int:
    """Return the original RPLAN room type id from a room dictionary."""
    for key in ("room_type_id", "raw_type_id", "type_id"):
        if key in room:
            return int(room[key])
    return -1


def get_room_type_name(room: Dict[str, Any]) -> str:
    """Return the canonical room type name for a room dictionary."""
    if "room_type_name" in room:
        return str(room["room_type_name"])
    if "type_name" in room:
        return str(room["type_name"])
    if "type" in room:
        return str(room["type"])

    room_type_id = get_room_type_id(room)
    return ROOM_TYPE_ID_TO_NAME.get(room_type_id, f"Unknown_{room_type_id}")


def room_type_counter(layout: LayoutData) -> Counter:
    """Count room types in a layout."""
    return Counter(get_room_type_name(room) for room in layout.rooms)


def room_type_id_counter(layout: LayoutData) -> Counter:
    """Count raw room type ids in a layout."""
    return Counter(get_room_type_id(room) for room in layout.rooms)


def room_area_by_type(layout: LayoutData) -> Counter:
    """Aggregate normalized room area by room type."""
    areas: Counter = Counter()
    for room in layout.rooms:
        room_type = get_room_type_name(room)
        areas[room_type] += float(room.get("area_ratio", 0.0))
    return areas


def bedroom_count(layout: LayoutData) -> int:
    """Count bedroom-like rooms using the RPLAN room taxonomy."""
    counts = room_type_counter(layout)
    return sum(counts.get(room_type, 0) for room_type in BEDROOM_TYPES)


def room_type_at(layout: LayoutData, node_id: int) -> Optional[str]:
    """Return the room type for a node id if present."""
    for room in layout.rooms:
        if int(room.get("id", -999)) == int(node_id):
            return get_room_type_name(room)
    return None


def adjacency_type_counter(layout: LayoutData) -> Counter:
    """Count undirected semantic adjacencies between room types.

    The exact node ids differ across floorplans, so this descriptor compares
    edges at the room-type level. For example, an edge between a LivingRoom and a
    Kitchen contributes the same key regardless of which node id each room has.
    """
    counter: Counter = Counter()

    for edge in layout.edges:
        source_type = room_type_at(layout, int(edge.get("source", -1)))
        target_type = room_type_at(layout, int(edge.get("target", -1)))

        if source_type is None or target_type is None:
            continue

        key = tuple(sorted((source_type, target_type)))
        counter[key] += 1

    return counter


def directed_relation_triplet_counter(layout: LayoutData) -> Counter:
    """Count directed semantic relation triplets.

    Each key is (source_room_type, relation_id, target_room_type). This preserves
    the Graph2Plan convention that relation_id describes the source node with
    respect to the target node.
    """
    counter: Counter = Counter()

    for edge in layout.edges:
        source_type = room_type_at(layout, int(edge.get("source", -1)))
        target_type = room_type_at(layout, int(edge.get("target", -1)))
        relation_id = int(edge.get("relation_id", -1))

        if source_type is None or target_type is None:
            continue

        counter[(source_type, relation_id, target_type)] += 1

    return counter


def relation_id_counter(layout: LayoutData) -> Counter:
    """Count Graph2Plan relation ids."""
    return Counter(int(edge.get("relation_id", -1)) for edge in layout.edges)


def grid_type_counter(layout: LayoutData) -> Counter:
    """Count room types in the Graph2Plan 5x5 grid representation."""
    counter: Counter = Counter()
    for room in layout.rooms:
        room_type = get_room_type_name(room)
        grid = room.get("grid_location_5x5", None)
        if isinstance(grid, list) and len(grid) >= 2:
            counter[(room_type, int(grid[0]), int(grid[1]))] += 1
    return counter


# -----------------------------------------------------------------------------
# Similarity functions
# -----------------------------------------------------------------------------

def generalized_jaccard(counter_a: Counter, counter_b: Counter) -> float:
    """Generalized Jaccard similarity for non-negative counters."""
    keys = set(counter_a.keys()) | set(counter_b.keys())
    if not keys:
        return 1.0

    numerator = 0.0
    denominator = 0.0
    for key in keys:
        a = float(counter_a.get(key, 0.0))
        b = float(counter_b.get(key, 0.0))
        numerator += min(a, b)
        denominator += max(a, b)

    if denominator == 0.0:
        return 1.0
    return numerator / denominator


def set_jaccard(values_a: Iterable[Any], values_b: Iterable[Any]) -> float:
    """Jaccard similarity between two sets."""
    a = set(values_a)
    b = set(values_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def room_count_similarity(query: LayoutData, retrieved: LayoutData) -> float:
    """Similarity based on total room count."""
    q = len(query.rooms)
    r = len(retrieved.rooms)
    denominator = max(q, r, 1)
    return max(0.0, 1.0 - abs(q - r) / denominator)


# -----------------------------------------------------------------------------
# Constraint satisfaction
# -----------------------------------------------------------------------------

def constraint_metrics(query: LayoutData, retrieved: LayoutData) -> Dict[str, Any]:
    """Evaluate simple architectural constraints derived from the query layout."""
    q_counts = room_type_counter(query)
    r_counts = room_type_counter(retrieved)

    q_bedrooms = bedroom_count(query)
    r_bedrooms = bedroom_count(retrieved)

    same_kitchen_count = int(q_counts.get("Kitchen", 0) == r_counts.get("Kitchen", 0))
    same_bathroom_count = int(q_counts.get("Bathroom", 0) == r_counts.get("Bathroom", 0))
    bedroom_count_diff_le_1 = int(abs(q_bedrooms - r_bedrooms) <= 1)
    room_count_diff_le_1 = int(abs(len(query.rooms) - len(retrieved.rooms)) <= 1)

    # If the query has a living room, the retrieved layout should also contain
    # at least one living room. If the query has no living room, this constraint
    # is considered satisfied because there is no living-room requirement.
    livingroom_covered = int(
        q_counts.get("LivingRoom", 0) == 0 or r_counts.get("LivingRoom", 0) >= 1
    )

    values = [
        same_kitchen_count,
        same_bathroom_count,
        bedroom_count_diff_le_1,
        room_count_diff_le_1,
        livingroom_covered,
    ]

    return {
        "constraint_same_kitchen_count": same_kitchen_count,
        "constraint_same_bathroom_count": same_bathroom_count,
        "constraint_bedroom_count_diff_le_1": bedroom_count_diff_le_1,
        "constraint_room_count_diff_le_1": room_count_diff_le_1,
        "constraint_livingroom_covered": livingroom_covered,
        "constraint_score": float(np.mean(values)),
        "hard_constraints_satisfied": int(all(values)),
    }


# -----------------------------------------------------------------------------
# Main per-result metric computation
# -----------------------------------------------------------------------------

def evaluate_pair(query: LayoutData, retrieved: LayoutData) -> Dict[str, Any]:
    """Compute all architectural similarity metrics for a query-result pair."""
    q_type_counts = room_type_counter(query)
    r_type_counts = room_type_counter(retrieved)

    q_type_id_counts = room_type_id_counter(query)
    r_type_id_counts = room_type_id_counter(retrieved)

    q_area_by_type = room_area_by_type(query)
    r_area_by_type = room_area_by_type(retrieved)

    q_adjacencies = adjacency_type_counter(query)
    r_adjacencies = adjacency_type_counter(retrieved)

    q_relations = relation_id_counter(query)
    r_relations = relation_id_counter(retrieved)

    q_triplets = directed_relation_triplet_counter(query)
    r_triplets = directed_relation_triplet_counter(retrieved)

    q_grid = grid_type_counter(query)
    r_grid = grid_type_counter(retrieved)

    type_count_similarity = generalized_jaccard(q_type_counts, r_type_counts)
    type_id_count_similarity = generalized_jaccard(q_type_id_counts, r_type_id_counts)
    type_set_similarity = set_jaccard(q_type_counts.keys(), r_type_counts.keys())
    area_similarity = generalized_jaccard(q_area_by_type, r_area_by_type)
    adjacency_similarity = generalized_jaccard(q_adjacencies, r_adjacencies)
    relation_similarity = generalized_jaccard(q_relations, r_relations)
    relation_triplet_similarity = generalized_jaccard(q_triplets, r_triplets)
    grid_similarity = generalized_jaccard(q_grid, r_grid)
    count_similarity = room_count_similarity(query, retrieved)

    constraints = constraint_metrics(query, retrieved)

    # A compact score used only for analysis/reporting. It is deliberately based
    # on architectural descriptors, not on the FAISS retrieval score.
    architectural_score = (
        0.25 * type_count_similarity
        + 0.15 * count_similarity
        + 0.15 * area_similarity
        + 0.20 * adjacency_similarity
        + 0.10 * relation_similarity
        + 0.10 * grid_similarity
        + 0.05 * constraints["constraint_score"]
    )

    return {
        "query_num_rooms": len(query.rooms),
        "retrieved_num_rooms": len(retrieved.rooms),
        "room_count_abs_diff": abs(len(query.rooms) - len(retrieved.rooms)),
        "room_count_similarity": count_similarity,
        "room_type_count_similarity": type_count_similarity,
        "room_type_id_count_similarity": type_id_count_similarity,
        "room_type_set_jaccard": type_set_similarity,
        "room_area_by_type_similarity": area_similarity,
        "adjacency_type_similarity": adjacency_similarity,
        "relation_id_similarity": relation_similarity,
        "relation_triplet_similarity": relation_triplet_similarity,
        "grid_type_similarity": grid_similarity,
        "architectural_score": architectural_score,
        **constraints,
    }


# -----------------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------------

def mean(values: Sequence[float]) -> float:
    """Safe mean for numeric values."""
    if not values:
        return 0.0
    return float(statistics.mean(values))


def aggregate_by_pipeline(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-result metrics at pipeline level."""
    by_pipeline: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pipeline[row["pipeline"]].append(row)

    summary_rows = []
    for pipeline, items in sorted(by_pipeline.items()):
        summary_rows.append(
            {
                "pipeline": pipeline,
                "num_results": len(items),
                "num_queries": len({item["query_id"] for item in items}),
                "mean_faiss_similarity": mean(
                    [item["faiss_similarity"] for item in items if item["faiss_similarity"] != ""]
                ),
                "mean_architectural_score": mean([item["architectural_score"] for item in items]),
                "mean_room_type_count_similarity": mean(
                    [item["room_type_count_similarity"] for item in items]
                ),
                "mean_room_count_similarity": mean([item["room_count_similarity"] for item in items]),
                "mean_room_area_by_type_similarity": mean(
                    [item["room_area_by_type_similarity"] for item in items]
                ),
                "mean_adjacency_type_similarity": mean(
                    [item["adjacency_type_similarity"] for item in items]
                ),
                "mean_relation_id_similarity": mean([item["relation_id_similarity"] for item in items]),
                "mean_relation_triplet_similarity": mean(
                    [item["relation_triplet_similarity"] for item in items]
                ),
                "mean_grid_type_similarity": mean([item["grid_type_similarity"] for item in items]),
                "mean_constraint_score": mean([item["constraint_score"] for item in items]),
                "hard_constraint_result_rate": mean(
                    [item["hard_constraints_satisfied"] for item in items]
                ),
            }
        )
    return summary_rows


def aggregate_by_rank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate metrics by retrieval rank for each pipeline."""
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["pipeline"], int(row["rank"]))].append(row)

    rank_rows = []
    for (pipeline, rank), items in sorted(groups.items()):
        rank_rows.append(
            {
                "pipeline": pipeline,
                "rank": rank,
                "num_items": len(items),
                "mean_architectural_score": mean([item["architectural_score"] for item in items]),
                "mean_room_type_count_similarity": mean(
                    [item["room_type_count_similarity"] for item in items]
                ),
                "mean_adjacency_type_similarity": mean(
                    [item["adjacency_type_similarity"] for item in items]
                ),
                "mean_constraint_score": mean([item["constraint_score"] for item in items]),
                "hard_constraint_result_rate": mean(
                    [item["hard_constraints_satisfied"] for item in items]
                ),
            }
        )
    return rank_rows


def aggregate_by_query(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate top-k metrics for each query and pipeline."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["pipeline"], row["query_id"])].append(row)

    query_rows = []
    for (pipeline, query_id), items in sorted(groups.items()):
        items_sorted = sorted(items, key=lambda x: int(x["rank"]))
        query_rows.append(
            {
                "pipeline": pipeline,
                "query_id": query_id,
                "num_results": len(items_sorted),
                "mean_architectural_score_at_k": mean(
                    [item["architectural_score"] for item in items_sorted]
                ),
                "best_architectural_score_at_k": max(
                    [item["architectural_score"] for item in items_sorted], default=0.0
                ),
                "mean_room_type_count_similarity_at_k": mean(
                    [item["room_type_count_similarity"] for item in items_sorted]
                ),
                "mean_adjacency_type_similarity_at_k": mean(
                    [item["adjacency_type_similarity"] for item in items_sorted]
                ),
                "mean_constraint_score_at_k": mean(
                    [item["constraint_score"] for item in items_sorted]
                ),
                "hit_hard_constraints_at_k": int(
                    any(item["hard_constraints_satisfied"] for item in items_sorted)
                ),
                "best_rank_hard_constraints": next(
                    (
                        int(item["rank"])
                        for item in items_sorted
                        if item["hard_constraints_satisfied"]
                    ),
                    "",
                ),
            }
        )
    return query_rows


# -----------------------------------------------------------------------------
# Plotting and markdown reporting
# -----------------------------------------------------------------------------

def save_summary_bar_plot(summary_rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Save a compact bar plot comparing the main metrics by pipeline."""
    if not summary_rows:
        return

    metrics = [
        "mean_architectural_score",
        "mean_room_type_count_similarity",
        "mean_adjacency_type_similarity",
        "mean_constraint_score",
        "hard_constraint_result_rate",
    ]

    pipelines = [row["pipeline"] for row in summary_rows]
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(pipelines))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, row in enumerate(summary_rows):
        values = [float(row[metric]) for metric in metrics]
        offsets = x - 0.4 + width / 2 + i * width
        ax.bar(offsets, values, width=width, label=row["pipeline"])

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval evaluation summary")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Architectural\nscore",
            "Room type\nsimilarity",
            "Adjacency\nsimilarity",
            "Constraint\nscore",
            "Hard constraint\nrate",
        ]
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def save_rank_curve(rank_rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Save a line plot showing how architectural score changes by rank."""
    if not rank_rows:
        return

    by_pipeline: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rank_rows:
        by_pipeline[row["pipeline"]].append(row)

    fig, ax = plt.subplots(figsize=(10, 6))

    for pipeline, items in sorted(by_pipeline.items()):
        items = sorted(items, key=lambda x: int(x["rank"]))
        ranks = [int(item["rank"]) for item in items]
        scores = [float(item["mean_architectural_score"]) for item in items]
        ax.plot(ranks, scores, marker="o", label=pipeline)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Mean architectural score")
    ax.set_title("Architectural quality by retrieval rank")
    ax.grid(alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def format_float(value: Any, digits: int = 4) -> str:
    """Format a numeric value for markdown tables."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown_report(
    output_path: Path,
    processed_root: Path,
    retrieval_root: Path,
    top_k: int,
    summary_rows: List[Dict[str, Any]],
    rank_rows: List[Dict[str, Any]],
    query_rows: List[Dict[str, Any]],
) -> None:
    """Write a standalone markdown report summarizing the evaluation."""
    lines: List[str] = []

    lines.append("# Retrieval Evaluation Report")
    lines.append("")
    lines.append("This report evaluates graph-based and vision-based floorplan retrieval results after preprocessing.")
    lines.append("The evaluation does not use the embedding similarity alone. Instead, each retrieved floorplan is compared with the query floorplan using architectural descriptors extracted from the preprocessed `layout_json` and `graph_json` files.")
    lines.append("")

    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Processed dataset root: `{processed_root}`")
    lines.append(f"- Retrieval results root: `{retrieval_root}`")
    lines.append(f"- Evaluation depth: top-{top_k}")
    lines.append("")

    lines.append("## Main Metrics")
    lines.append("")
    lines.append("- `architectural_score`: weighted aggregate of room-type similarity, room-count similarity, area similarity, adjacency similarity, relation similarity, grid similarity, and constraint satisfaction.")
    lines.append("- `room_type_count_similarity`: generalized Jaccard similarity between room-type histograms.")
    lines.append("- `adjacency_type_similarity`: generalized Jaccard similarity between semantic adjacency multisets, where edges are represented as room-type pairs.")
    lines.append("- `relation_id_similarity`: generalized Jaccard similarity between Graph2Plan relation-id histograms.")
    lines.append("- `constraint_score`: average satisfaction of simple query-derived constraints such as matching kitchen count, matching bathroom count, bedroom count within ±1, room count within ±1, and living-room coverage.")
    lines.append("- `hard_constraint_result_rate`: fraction of retrieved results satisfying all simple constraints.")
    lines.append("")

    lines.append("## Pipeline Summary")
    lines.append("")
    if summary_rows:
        headers = [
            "Pipeline",
            "Queries",
            "Results",
            "Architectural",
            "Room types",
            "Adjacency",
            "Constraints",
            "Hard constraints",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in summary_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["pipeline"]),
                        str(row["num_queries"]),
                        str(row["num_results"]),
                        format_float(row["mean_architectural_score"]),
                        format_float(row["mean_room_type_count_similarity"]),
                        format_float(row["mean_adjacency_type_similarity"]),
                        format_float(row["mean_constraint_score"]),
                        format_float(row["hard_constraint_result_rate"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No summary rows were produced.")
    lines.append("")

    lines.append("## Rank-Level Behavior")
    lines.append("")
    lines.append("The rank-level table is saved as `metrics_by_rank.csv`. It can be used to check whether architectural quality decreases as retrieval rank increases.")
    lines.append("")

    lines.append("## Query-Level Behavior")
    lines.append("")
    lines.append("The query-level table is saved as `metrics_by_query.csv`. It reports mean and best top-k architectural scores for each query and pipeline.")
    lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append("- `metrics_per_result.csv`: one row per query-result pair.")
    lines.append("- `metrics_by_query.csv`: top-k aggregates for each query and pipeline.")
    lines.append("- `metrics_by_rank.csv`: rank-wise aggregate metrics.")
    lines.append("- `pipeline_summary.json`: machine-readable summary.")
    lines.append("- `pipeline_summary.md`: this report.")
    lines.append("- `plots/summary_bar.png`: compact pipeline comparison plot.")
    lines.append("- `plots/rank_curve.png`: rank-wise architectural score plot.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# Evaluation driver
# -----------------------------------------------------------------------------

def evaluate_pipeline(
    pipeline_name: str,
    retrieval_items: List[RetrievalItem],
    store: LayoutStore,
) -> List[Dict[str, Any]]:
    """Evaluate all retrieval items from one pipeline."""
    rows: List[Dict[str, Any]] = []

    for item in retrieval_items:
        try:
            query_layout = store.get(item.query_id)
            retrieved_layout = store.get(item.retrieved_id)
        except FileNotFoundError as exc:
            print(f"[warning] Skipping missing layout for {item.query_id} -> {item.retrieved_id}: {exc}")
            continue

        metrics = evaluate_pair(query_layout, retrieved_layout)

        rows.append(
            {
                "pipeline": pipeline_name,
                "query_id": item.query_id,
                "retrieved_id": item.retrieved_id,
                "rank": item.rank,
                "faiss_similarity": item.similarity if item.similarity is not None else "",
                "faiss_distance": item.distance if item.distance is not None else "",
                "source_json": item.source_json,
                **metrics,
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RPLAN retrieval results.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/rplan"),
        help="Root folder containing layouts_json/ and graphs_json/.",
    )
    parser.add_argument(
        "--retrieval-root",
        type=Path,
        default=Path("outputs/retrieval_gpu_full"),
        help="Root folder containing graph_pipeline/ and vision_pipeline/ retrieval outputs.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Folder where evaluation outputs will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieved results to evaluate per query.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["graph_pipeline", "vision_pipeline"],
        help="Pipeline subdirectories to evaluate under retrieval-root.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous evaluation outputs before running.",
    )
    return parser.parse_args()


def clean_outputs(outputs_root: Path) -> None:
    """Remove previous evaluation outputs from the output directory."""
    if not outputs_root.exists():
        return

    for path in outputs_root.iterdir():
        if path.is_dir():
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            path.rmdir()
        else:
            path.unlink()


def main() -> None:
    args = parse_args()

    if args.clean:
        clean_outputs(args.outputs_root)

    args.outputs_root.mkdir(parents=True, exist_ok=True)
    plots_dir = args.outputs_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not (args.processed_root / "layouts_json").exists():
        raise FileNotFoundError(f"Missing layouts_json directory under {args.processed_root}")
    if not (args.processed_root / "graphs_json").exists():
        raise FileNotFoundError(f"Missing graphs_json directory under {args.processed_root}")
    if not args.retrieval_root.exists():
        raise FileNotFoundError(f"Missing retrieval root: {args.retrieval_root}")

    store = LayoutStore(args.processed_root)

    all_rows: List[Dict[str, Any]] = []
    input_overview: Dict[str, Any] = {}

    for pipeline_name in args.pipelines:
        retrieval_items = load_pipeline_results(
            retrieval_root=args.retrieval_root,
            pipeline_name=pipeline_name,
            top_k=args.top_k,
        )

        input_overview[pipeline_name] = {
            "num_retrieval_items": len(retrieval_items),
            "num_queries": len({item.query_id for item in retrieval_items}),
        }

        print(
            f"Evaluating {pipeline_name}: "
            f"{input_overview[pipeline_name]['num_queries']} queries, "
            f"{len(retrieval_items)} query-result pairs"
        )

        rows = evaluate_pipeline(pipeline_name, retrieval_items, store)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No retrieval results were evaluated. Check retrieval-root and pipeline names.")

    summary_rows = aggregate_by_pipeline(all_rows)
    rank_rows = aggregate_by_rank(all_rows)
    query_rows = aggregate_by_query(all_rows)

    write_csv(args.outputs_root / "metrics_per_result.csv", all_rows)
    write_csv(args.outputs_root / "metrics_by_rank.csv", rank_rows)
    write_csv(args.outputs_root / "metrics_by_query.csv", query_rows)

    summary = {
        "processed_root": str(args.processed_root),
        "retrieval_root": str(args.retrieval_root),
        "outputs_root": str(args.outputs_root),
        "top_k": args.top_k,
        "input_overview": input_overview,
        "pipeline_summary": summary_rows,
    }

    write_json(args.outputs_root / "pipeline_summary.json", summary)

    save_summary_bar_plot(summary_rows, plots_dir / "summary_bar.png")
    save_rank_curve(rank_rows, plots_dir / "rank_curve.png")

    write_markdown_report(
        output_path=args.outputs_root / "pipeline_summary.md",
        processed_root=args.processed_root,
        retrieval_root=args.retrieval_root,
        top_k=args.top_k,
        summary_rows=summary_rows,
        rank_rows=rank_rows,
        query_rows=query_rows,
    )

    print("\nEvaluation completed.")
    print(f"Per-result metrics: {args.outputs_root / 'metrics_per_result.csv'}")
    print(f"Query metrics:      {args.outputs_root / 'metrics_by_query.csv'}")
    print(f"Rank metrics:       {args.outputs_root / 'metrics_by_rank.csv'}")
    print(f"Summary JSON:       {args.outputs_root / 'pipeline_summary.json'}")
    print(f"Summary report:     {args.outputs_root / 'pipeline_summary.md'}")
    print(f"Plots:              {plots_dir}")


if __name__ == "__main__":
    main()
