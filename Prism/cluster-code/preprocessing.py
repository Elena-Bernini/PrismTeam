#!/usr/bin/env python3
"""
Unified RPLAN / Graph2Plan preprocessing pipeline.

This script converts the preprocessed Graph2Plan/RPLAN training data into a
project-friendly representation:

1. Dataset inspection reports.
2. Per-layout JSON files with official RPLAN room type IDs and names.
3. Per-layout graph JSON files with room nodes and spatial relation edges.
4. Dataset-level statistics.
5. A small number of visual quality-control previews.
6. A Markdown report summarizing the preprocessing run.

The script intentionally does not generate previews for the full dataset by
 default. Preview images are only meant for debugging and documentation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so the script works on headless cluster nodes.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
from tqdm import tqdm

# SciPy may emit a deprecation warning while unpickling MATLAB structs created
# by older Graph2Plan preprocessing scripts. The warning is harmless here.
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -----------------------------------------------------------------------------
# Official Graph2Plan / RPLAN metadata
# -----------------------------------------------------------------------------

# Official room labels used by the Graph2Plan preprocessing pipeline. The room
# boxes in data_train_converted.pkl store the room type as the fifth value:
#   box = (x0, y0, x1, y1, room_type_id)
# The current pipeline expects room_type_id values in the 0-12 range for rooms.
ROOM_TYPE_ID_TO_NAME: Dict[int, str] = {
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

# Extra labels documented by Graph2Plan. They should not normally appear as room
# boxes in data_train_converted.pkl, but keeping them here makes diagnostics more
# explicit if an unexpected type appears.
EXTRA_TYPE_ID_TO_NAME: Dict[int, str] = {
    13: "External",
    14: "ExteriorWall",
    15: "FrontDoor",
    16: "InteriorWall",
    17: "InteriorDoor",
}

ALL_TYPE_ID_TO_NAME: Dict[int, str] = {
    **ROOM_TYPE_ID_TO_NAME,
    **EXTRA_TYPE_ID_TO_NAME,
}

# Official relation labels used by Graph2Plan. An edge has the form:
#   (u, v, r)
# where r describes the position of room u relative to room v.
RELATION_ID_TO_NAME: Dict[int, str] = {
    0: "left-above",
    1: "left-below",
    2: "left-of",
    3: "above",
    4: "inside",
    5: "surrounding",
    6: "below",
    7: "right-of",
    8: "right-above",
    9: "right-below",
}

# A compact color map used only for visual quality control images. The pipeline
# does not use colors as data.
ROOM_COLORS: Dict[str, str] = {
    "LivingRoom": "#eeeeee",
    "MasterRoom": "#fff2a8",
    "SecondRoom": "#fff2a8",
    "GuestRoom": "#fff2a8",
    "ChildRoom": "#fff2a8",
    "StudyRoom": "#f4e6ff",
    "Bathroom": "#bde7ff",
    "Kitchen": "#ffd1dc",
    "DiningRoom": "#ffe4c4",
    "Balcony": "#c9e8a2",
    "Entrance": "#dddddd",
    "Storage": "#ddd0c8",
    "Wall-in": "#d9d9d9",
    "Unknown": "#ffffff",
}


# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelinePaths:
    """Centralized filesystem layout for the preprocessing run."""

    dataset_root: Path
    processed_root: Path
    audit_root: Path

    @property
    def pkl_path(self) -> Path:
        return self.dataset_root / "Interface/static/Data/data_train_converted.pkl"

    @property
    def snapshot_dir(self) -> Path:
        return self.dataset_root / "Interface/static/Data/snapshot_train"

    @property
    def layouts_dir(self) -> Path:
        return self.processed_root / "layouts_json"

    @property
    def graphs_dir(self) -> Path:
        return self.processed_root / "graphs_json"

    @property
    def metadata_dir(self) -> Path:
        return self.processed_root / "metadata"

    @property
    def layout_previews_dir(self) -> Path:
        return self.audit_root / "layout_previews"

    @property
    def graph_previews_dir(self) -> Path:
        return self.audit_root / "graph_previews"

    def ensure_dirs(self) -> None:
        """Create all output directories used by the pipeline."""
        for directory in [
            self.processed_root,
            self.audit_root,
            self.layouts_dir,
            self.graphs_dir,
            self.metadata_dir,
            self.layout_previews_dir,
            self.graph_previews_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def clean_outputs(paths: PipelinePaths) -> None:
    """Remove generated outputs while keeping the raw dataset untouched."""
    for directory in [
        paths.layouts_dir,
        paths.graphs_dir,
        paths.metadata_dir,
        paths.layout_previews_dir,
        paths.graph_previews_dir,
    ]:
        if directory.exists():
            shutil.rmtree(directory)

    # Keep the audit root itself, but remove JSON/Markdown reports created by
    # previous runs so the new report is unambiguous.
    if paths.audit_root.exists():
        for pattern in ["*.json", "*.md"]:
            for path in paths.audit_root.glob(pattern):
                path.unlink()

    paths.ensure_dirs()


def to_json_safe(value: Any, max_items: int = 8) -> Any:
    """Convert Python/NumPy/MATLAB-like values into JSON-safe objects.

    This helper is used only for compact inspection reports. It intentionally
    summarizes complex object arrays rather than expanding them deeply.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        summary: Dict[str, Any] = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

        flat = value.flatten()
        if value.dtype == object:
            summary["preview_types"] = [str(type(x)) for x in flat[:max_items]]
        else:
            summary["preview"] = [to_json_safe(x) for x in flat[:max_items]]
        return summary

    if isinstance(value, (list, tuple)):
        return [to_json_safe(x) for x in value[:max_items]]

    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}

    if hasattr(value, "__dict__"):
        return {
            "type": str(type(value)),
            "attributes": list(value.__dict__.keys()),
        }

    return str(value)


def recursively_to_python(value: Any) -> Any:
    """Recursively convert NumPy values into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return recursively_to_python(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, (list, tuple)):
        return [recursively_to_python(x) for x in value]

    if isinstance(value, dict):
        return {str(k): recursively_to_python(v) for k, v in value.items()}

    return value


def load_pickle(path: Path) -> Dict[str, Any]:
    """Load the Graph2Plan pickle file and fail with a clear message."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data_train_converted.pkl: {path}\n"
            "Expected location: <dataset_root>/Interface/static/Data/data_train_converted.pkl"
        )

    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def compute_grid_location(
    cx: float,
    cy: float,
    width: int = 256,
    height: int = 256,
    grid_size: int = 5,
) -> List[int]:
    """Assign a centroid to a grid cell in the 5x5 Graph2Plan-style grid.

    Coordinates follow image convention: origin is top-left, x grows to the
    right, and y grows downward.
    """
    gx = min(grid_size - 1, max(0, int(cx / width * grid_size)))
    gy = min(grid_size - 1, max(0, int(cy / height * grid_size)))
    return [gx, gy]


def room_type_name(room_type_id: int) -> str:
    """Return the official room type name when known."""
    return ALL_TYPE_ID_TO_NAME.get(room_type_id, f"Unknown_{room_type_id}")


def relation_name(relation_id: int) -> str:
    """Return the official spatial relation name when known."""
    return RELATION_ID_TO_NAME.get(relation_id, f"relation_{relation_id}")


# -----------------------------------------------------------------------------
# Step 1: dataset inspection
# -----------------------------------------------------------------------------


def inspect_dataset(paths: PipelinePaths) -> Dict[str, Any]:
    """Inspect raw files and the first training sample."""
    print("\n[1/6] Inspecting dataset")

    if not paths.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {paths.dataset_root}")

    files = [p for p in paths.dataset_root.rglob("*") if p.is_file()]
    extension_counts = Counter(p.suffix.lower() for p in files)

    file_audit = {
        "dataset_root": str(paths.dataset_root),
        "num_files": len(files),
        "extensions": dict(extension_counts),
        "pkl_exists": paths.pkl_path.exists(),
        "pkl_path": str(paths.pkl_path),
        "snapshot_dir_exists": paths.snapshot_dir.exists(),
        "snapshot_dir": str(paths.snapshot_dir),
        "first_100_files": [str(p) for p in files[:100]],
    }

    save_json(paths.audit_root / "rplan_file_audit.json", file_audit)

    obj = load_pickle(paths.pkl_path)
    data = obj["data"]
    sample = data[0]
    sample_attributes = [a for a in dir(sample) if not a.startswith("_")]

    sample_report = {
        "pkl_keys": list(obj.keys()),
        "num_samples": len(data),
        "sample_attributes": sample_attributes,
        "sample_attribute_summaries": {},
    }

    for attr in sample_attributes:
        value = getattr(sample, attr)
        sample_report["sample_attribute_summaries"][attr] = {
            "python_type": str(type(value)),
            "summary": to_json_safe(value),
        }

    save_json(paths.audit_root / "train_converted_sample_report.json", sample_report)

    print(f"  Found {len(files)} files under {paths.dataset_root}")
    print(f"  PKL samples: {len(data)}")
    print(f"  Saved: {paths.audit_root / 'rplan_file_audit.json'}")
    print(f"  Saved: {paths.audit_root / 'train_converted_sample_report.json'}")

    return {
        "num_raw_files": len(files),
        "num_pkl_samples": len(data),
        "sample_attributes": sample_attributes,
    }


# -----------------------------------------------------------------------------
# Step 2: convert Graph2Plan samples to layout JSON
# -----------------------------------------------------------------------------


def convert_sample_to_layout(sample: Any, pkl_path: Path) -> Dict[str, Any]:
    """Convert one Graph2Plan sample into a layout-level JSON object."""
    sample_id = str(sample.name)

    boundary = np.asarray(sample.boundary)
    boxes = np.asarray(sample.box)
    edges = np.asarray(sample.edge)

    # The Graph2Plan converted interface data uses a normalized 256x256 layout
    # coordinate system. Coordinates are image coordinates, not Cartesian axes.
    width = 256
    height = 256
    building_area = width * height

    rooms: List[Dict[str, Any]] = []

    for room_id, box in enumerate(boxes):
        x_min, y_min, x_max, y_max, type_id = [int(v) for v in box]

        room_width = max(0, x_max - x_min)
        room_height = max(0, y_max - y_min)
        area_pixels = room_width * room_height
        cx = x_min + room_width / 2.0
        cy = y_min + room_height / 2.0

        room: Dict[str, Any] = {
            # Local node index inside this specific layout graph.
            "id": room_id,
            # Official Graph2Plan/RPLAN room category ID from the source data.
            "room_type_id": type_id,
            # Official Graph2Plan/RPLAN room category name.
            "room_type_name": room_type_name(type_id),
            "bbox": [x_min, y_min, x_max, y_max],
            "width": room_width,
            "height": room_height,
            "area_pixels": area_pixels,
            "area_ratio": area_pixels / building_area,
            "centroid": [cx, cy],
            "grid_location_5x5": compute_grid_location(cx, cy, width, height),
        }

        # rBoundary stores the room polygon. It is more accurate for
        # visualization than the rectangular bounding box, especially for
        # L-shaped rooms.
        if hasattr(sample, "rBoundary"):
            try:
                room["polygon"] = recursively_to_python(sample.rBoundary[room_id])
            except Exception:
                room["polygon"] = None
        else:
            room["polygon"] = None

        rooms.append(room)

    graph_edges: List[Dict[str, Any]] = []
    for edge_id, edge in enumerate(edges):
        source, target, rel_id = [int(v) for v in edge]
        graph_edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "relation_id": rel_id,
                # In Graph2Plan, this describes source relative to target.
                "relation": relation_name(rel_id),
            }
        )

    return {
        "id": sample_id,
        "coordinate_system": {
            "width": width,
            "height": height,
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
        },
        "source": {
            "dataset": "RPLAN_Graph2Plan_processed",
            "pkl_file": str(pkl_path),
        },
        "boundary": recursively_to_python(boundary),
        "rooms": rooms,
        "edges": graph_edges,
        "raw": {
            "order": recursively_to_python(sample.order) if hasattr(sample, "order") else None,
        },
    }


def convert_layouts(paths: PipelinePaths, limit: Optional[int]) -> Dict[str, Any]:
    """Convert the PKL training samples into per-layout JSON files."""
    print("\n[2/6] Converting samples to layout JSON")

    obj = load_pickle(paths.pkl_path)
    data = obj["data"]
    total_available = len(data)

    if limit is not None:
        data = data[:limit]

    metadata: List[Dict[str, Any]] = []
    unknown_room_type_ids: Counter[int] = Counter()
    generated_ids: List[str] = []

    for sample in tqdm(data, desc="Converting layouts"):
        layout = convert_sample_to_layout(sample, paths.pkl_path)
        out_path = paths.layouts_dir / f"{layout['id']}.json"
        save_json(out_path, layout)

        room_type_counts = Counter(room["room_type_name"] for room in layout["rooms"])
        room_type_id_counts = Counter(str(room["room_type_id"]) for room in layout["rooms"])

        for room in layout["rooms"]:
            if room["room_type_id"] not in ROOM_TYPE_ID_TO_NAME:
                unknown_room_type_ids[room["room_type_id"]] += 1

        metadata.append(
            {
                "id": layout["id"],
                "num_rooms": len(layout["rooms"]),
                "num_edges": len(layout["edges"]),
                "room_type_counts": dict(room_type_counts),
                "room_type_id_counts": dict(room_type_id_counts),
                "layout_json": str(out_path),
            }
        )
        generated_ids.append(layout["id"])

    save_json(paths.metadata_dir / "layouts_metadata.json", metadata)
    save_json(paths.metadata_dir / "layout_ids.json", generated_ids)

    print(f"  Converted layouts: {len(metadata)} / {total_available}")
    print(f"  Saved layout JSON files to: {paths.layouts_dir}")
    print(f"  Saved metadata to: {paths.metadata_dir / 'layouts_metadata.json'}")

    return {
        "total_available": total_available,
        "num_converted_layouts": len(metadata),
        "unknown_room_type_ids": dict(unknown_room_type_ids),
    }


# -----------------------------------------------------------------------------
# Step 3: convert layout JSON files to graph JSON files
# -----------------------------------------------------------------------------


def layout_to_graph(layout: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a layout JSON object into a compact graph JSON object."""
    nodes: List[Dict[str, Any]] = []
    for room in layout["rooms"]:
        nodes.append(
            {
                "id": room["id"],
                "room_type_id": room["room_type_id"],
                "room_type_name": room["room_type_name"],
                "area_ratio": room["area_ratio"],
                "bbox": room["bbox"],
                "centroid": room["centroid"],
                "grid_location_5x5": room["grid_location_5x5"],
            }
        )

    edges: List[Dict[str, Any]] = []
    for edge in layout["edges"]:
        edges.append(
            {
                "source": edge["source"],
                "target": edge["target"],
                "relation_id": edge["relation_id"],
                "relation": edge["relation"],
            }
        )

    return {
        "id": layout["id"],
        "coordinate_system": layout.get("coordinate_system", {}),
        "nodes": nodes,
        "edges": edges,
    }


def convert_graphs(paths: PipelinePaths) -> Dict[str, Any]:
    """Build graph JSON files from previously generated layout JSON files."""
    print("\n[3/6] Converting layout JSON files to graph JSON")

    layout_paths = sorted(paths.layouts_dir.glob("*.json"))
    if not layout_paths:
        raise RuntimeError(f"No layout JSON files found in {paths.layouts_dir}")

    for layout_path in tqdm(layout_paths, desc="Building graphs"):
        layout = load_json(layout_path)
        graph = layout_to_graph(layout)
        save_json(paths.graphs_dir / f"{graph['id']}.json", graph)

    print(f"  Converted graphs: {len(layout_paths)}")
    print(f"  Saved graph JSON files to: {paths.graphs_dir}")

    return {"num_converted_graphs": len(layout_paths)}


# -----------------------------------------------------------------------------
# Step 4: statistics
# -----------------------------------------------------------------------------


def compute_statistics(paths: PipelinePaths) -> Dict[str, Any]:
    """Compute dataset-level statistics from the generated layout JSON files."""
    print("\n[4/6] Computing dataset statistics")

    layout_paths = sorted(paths.layouts_dir.glob("*.json"))
    if not layout_paths:
        raise RuntimeError(f"No layout JSON files found in {paths.layouts_dir}")

    room_type_name_counter: Counter[str] = Counter()
    room_type_id_counter: Counter[str] = Counter()
    relation_counter: Counter[str] = Counter()
    relation_id_counter: Counter[str] = Counter()
    room_count_per_plan: List[int] = []
    edge_count_per_plan: List[int] = []
    area_by_room_type: Dict[str, List[float]] = defaultdict(list)
    unknown_room_type_ids: Counter[str] = Counter()
    unknown_relation_ids: Counter[str] = Counter()

    for path in tqdm(layout_paths, desc="Computing statistics"):
        layout = load_json(path)
        rooms = layout["rooms"]
        edges = layout["edges"]

        room_count_per_plan.append(len(rooms))
        edge_count_per_plan.append(len(edges))

        for room in rooms:
            type_id = int(room["room_type_id"])
            type_id_str = str(type_id)
            type_name = room["room_type_name"]

            room_type_id_counter[type_id_str] += 1
            room_type_name_counter[type_name] += 1
            area_by_room_type[type_name].append(float(room["area_ratio"]))

            if type_id not in ROOM_TYPE_ID_TO_NAME:
                unknown_room_type_ids[type_id_str] += 1

        for edge in edges:
            rel_id = int(edge["relation_id"])
            rel_id_str = str(rel_id)
            relation_id_counter[rel_id_str] += 1
            relation_counter[edge["relation"]] += 1

            if rel_id not in RELATION_ID_TO_NAME:
                unknown_relation_ids[rel_id_str] += 1

    room_count_array = np.array(room_count_per_plan, dtype=np.float32)
    edge_count_array = np.array(edge_count_per_plan, dtype=np.float32)

    stats = {
        "num_layouts": len(layout_paths),
        "room_type_name_counts": dict(room_type_name_counter),
        "room_type_id_counts": dict(room_type_id_counter),
        "relation_type_counts": dict(relation_counter),
        "relation_id_counts": dict(relation_id_counter),
        "unknown_room_type_ids": dict(unknown_room_type_ids),
        "unknown_relation_ids": dict(unknown_relation_ids),
        "room_count_per_plan": {
            "min": int(np.min(room_count_array)),
            "max": int(np.max(room_count_array)),
            "mean": float(np.mean(room_count_array)),
            "median": float(np.median(room_count_array)),
            "std": float(np.std(room_count_array)),
        },
        "edge_count_per_plan": {
            "min": int(np.min(edge_count_array)),
            "max": int(np.max(edge_count_array)),
            "mean": float(np.mean(edge_count_array)),
            "median": float(np.median(edge_count_array)),
            "std": float(np.std(edge_count_array)),
        },
        "area_by_room_type_mean": {
            room_type: float(np.mean(values)) for room_type, values in area_by_room_type.items()
        },
        "area_by_room_type_std": {
            room_type: float(np.std(values)) for room_type, values in area_by_room_type.items()
        },
    }

    save_json(paths.audit_root / "rplan_processed_statistics.json", stats)

    print(f"  Saved statistics to: {paths.audit_root / 'rplan_processed_statistics.json'}")
    print(f"  Layouts: {stats['num_layouts']}")
    print(f"  Room types: {dict(room_type_name_counter)}")

    return stats


# -----------------------------------------------------------------------------
# Step 5: visual quality-control previews
# -----------------------------------------------------------------------------


def polygon_points(raw_polygon: Any) -> List[List[float]]:
    """Extract polygon points as [[x, y], ...] from JSON data."""
    if not isinstance(raw_polygon, list):
        return []

    points: List[List[float]] = []
    for p in raw_polygon:
        if isinstance(p, list) and len(p) >= 2:
            try:
                points.append([float(p[0]), float(p[1])])
            except (TypeError, ValueError):
                continue
    return points


def draw_boundary(ax: plt.Axes, layout: Dict[str, Any]) -> None:
    """Draw the outer floorplan boundary."""
    boundary = layout.get("boundary", None)
    if not boundary:
        return

    points = [[p[0], p[1]] for p in boundary if isinstance(p, list) and len(p) >= 2]
    if len(points) >= 3:
        patch = Polygon(
            points,
            closed=True,
            facecolor="none",
            edgecolor="black",
            linewidth=2.5,
            zorder=8,
        )
        ax.add_patch(patch)


def draw_room(ax: plt.Axes, room: Dict[str, Any], show_node: bool = False) -> None:
    """Draw one room using its polygon when available, otherwise its bbox."""
    room_name = room.get("room_type_name", "Unknown")
    room_type_id = room.get("room_type_id", "NA")
    color = ROOM_COLORS.get(room_name, ROOM_COLORS["Unknown"])

    points = polygon_points(room.get("polygon", None))
    if len(points) >= 3:
        patch = Polygon(
            points,
            closed=True,
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.82,
            zorder=2,
        )
        ax.add_patch(patch)
    else:
        x0, y0, x1, y1 = room["bbox"]
        patch = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.82,
            zorder=2,
        )
        ax.add_patch(patch)

    cx, cy = room["centroid"]

    if show_node:
        ax.scatter(cx, cy, s=58, edgecolors="black", zorder=6)
        text_y = cy - 7
        va = "top"
    else:
        text_y = cy
        va = "center"

    ax.text(
        cx,
        text_y,
        f"{room['id']}\n{room_name}\ntype={room_type_id}",
        ha="center",
        va=va,
        fontsize=5.2,
        zorder=7,
    )


def setup_layout_axes(ax: plt.Axes, title: str) -> None:
    """Use the correct Graph2Plan coordinate convention for plotting."""
    ax.set_xlim(0, 256)
    # y-axis is inverted because images use top-left origin and y grows downward.
    ax.set_ylim(256, 0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")


def draw_layout_comparison(paths: PipelinePaths, sample_id: str) -> Path:
    """Draw original snapshot and polygon-based layout rendering side by side."""
    layout_path = paths.layouts_dir / f"{sample_id}.json"
    snapshot_path = paths.snapshot_dir / f"{sample_id}.png"

    layout = load_json(layout_path)
    output_path = paths.layout_previews_dir / f"{sample_id}_layout_comparison.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].set_title(f"Original snapshot: {sample_id}")
    if snapshot_path.exists():
        img = Image.open(snapshot_path).convert("RGB")
        axes[0].imshow(img)
    else:
        axes[0].text(
            0.5,
            0.5,
            f"Snapshot not found:\n{snapshot_path}",
            ha="center",
            va="center",
        )
    axes[0].axis("off")

    for room in layout["rooms"]:
        draw_room(axes[1], room, show_node=False)
    draw_boundary(axes[1], layout)
    setup_layout_axes(axes[1], f"Layout JSON render: {sample_id}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def draw_graph_edges(ax: plt.Axes, layout: Dict[str, Any], graph: Dict[str, Any]) -> None:
    """Draw relation edges from source room to target room.

    The label follows Graph2Plan semantics: relation = source relative to target.
    """
    rooms_by_id = {room["id"]: room for room in layout["rooms"]}

    for edge in graph["edges"]:
        source = edge["source"]
        target = edge["target"]

        if source not in rooms_by_id or target not in rooms_by_id:
            continue

        room_a = rooms_by_id[source]
        room_b = rooms_by_id[target]
        x1, y1 = room_a["centroid"]
        x2, y2 = room_b["centroid"]

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=1.4, alpha=0.85),
            zorder=5,
        )

        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        ax.text(
            mx,
            my,
            f"{source}->{target}\n{edge['relation']}",
            ha="center",
            va="center",
            fontsize=4.8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
            zorder=9,
        )


def draw_graph_comparison(paths: PipelinePaths, sample_id: str) -> Path:
    """Draw original snapshot and graph rendering side by side."""
    layout_path = paths.layouts_dir / f"{sample_id}.json"
    graph_path = paths.graphs_dir / f"{sample_id}.json"
    snapshot_path = paths.snapshot_dir / f"{sample_id}.png"

    layout = load_json(layout_path)
    graph = load_json(graph_path)
    output_path = paths.graph_previews_dir / f"{sample_id}_graph_preview.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].set_title(f"Original snapshot: {sample_id}")
    if snapshot_path.exists():
        img = Image.open(snapshot_path).convert("RGB")
        axes[0].imshow(img)
    else:
        axes[0].text(
            0.5,
            0.5,
            f"Snapshot not found:\n{snapshot_path}",
            ha="center",
            va="center",
        )
    axes[0].axis("off")

    for room in layout["rooms"]:
        draw_room(axes[1], room, show_node=True)
    draw_graph_edges(axes[1], layout, graph)
    draw_boundary(axes[1], layout)
    setup_layout_axes(axes[1], f"Graph render: {sample_id}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def choose_preview_ids(
    available_ids: Sequence[str],
    requested_ids: Optional[Sequence[str]],
    n: int,
    seed: int,
) -> List[str]:
    """Choose IDs for preview generation."""
    available_set = set(available_ids)

    if requested_ids:
        selected = [str(x) for x in requested_ids if str(x) in available_set]
        missing = [str(x) for x in requested_ids if str(x) not in available_set]
        if missing:
            print(f"  Warning: requested preview IDs not found: {missing}")
        return selected

    if n <= 0:
        return []

    rng = random.Random(seed)
    return rng.sample(list(available_ids), min(n, len(available_ids)))


def generate_previews(
    paths: PipelinePaths,
    layout_preview_n: int,
    graph_preview_n: int,
    preview_seed: int,
    layout_preview_ids: Optional[Sequence[str]] = None,
    graph_preview_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Generate a limited number of layout and graph quality-control previews."""
    print("\n[5/6] Generating visual quality-control previews")

    layout_paths = sorted(paths.layouts_dir.glob("*.json"))
    graph_paths = sorted(paths.graphs_dir.glob("*.json"))
    layout_ids = [p.stem for p in layout_paths]
    graph_ids = [p.stem for p in graph_paths]

    selected_layout_ids = choose_preview_ids(
        layout_ids,
        layout_preview_ids,
        layout_preview_n,
        preview_seed,
    )
    selected_graph_ids = choose_preview_ids(
        graph_ids,
        graph_preview_ids,
        graph_preview_n,
        preview_seed + 1,
    )

    layout_outputs: List[str] = []
    for sample_id in tqdm(selected_layout_ids, desc="Layout previews"):
        layout_outputs.append(str(draw_layout_comparison(paths, sample_id)))

    graph_outputs: List[str] = []
    for sample_id in tqdm(selected_graph_ids, desc="Graph previews"):
        graph_outputs.append(str(draw_graph_comparison(paths, sample_id)))

    print(f"  Layout previews: {len(layout_outputs)} -> {paths.layout_previews_dir}")
    print(f"  Graph previews: {len(graph_outputs)} -> {paths.graph_previews_dir}")

    return {
        "layout_preview_count": len(layout_outputs),
        "graph_preview_count": len(graph_outputs),
        "layout_preview_ids": selected_layout_ids,
        "graph_preview_ids": selected_graph_ids,
        "layout_preview_files": layout_outputs,
        "graph_preview_files": graph_outputs,
    }


# -----------------------------------------------------------------------------
# Step 6: final report
# -----------------------------------------------------------------------------


def write_report(
    paths: PipelinePaths,
    args: argparse.Namespace,
    inspect_summary: Dict[str, Any],
    layout_summary: Dict[str, Any],
    graph_summary: Dict[str, Any],
    stats: Dict[str, Any],
    preview_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Write machine-readable and human-readable summaries."""
    print("\n[6/6] Writing preprocessing report")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(paths.dataset_root),
        "processed_root": str(paths.processed_root),
        "audit_root": str(paths.audit_root),
        "limit": args.limit,
        "inspect_summary": inspect_summary,
        "layout_summary": layout_summary,
        "graph_summary": graph_summary,
        "statistics_file": str(paths.audit_root / "rplan_processed_statistics.json"),
        "preview_summary": preview_summary,
        "room_type_mapping": ROOM_TYPE_ID_TO_NAME,
        "extra_type_mapping": EXTRA_TYPE_ID_TO_NAME,
        "relation_mapping": RELATION_ID_TO_NAME,
        "coordinate_system": {
            "width": 256,
            "height": 256,
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "relation_semantics": "edge relation describes source room relative to target room",
        },
    }

    save_json(paths.audit_root / "preprocessing_summary.json", summary)

    report_lines: List[str] = []
    report_lines.append("# RPLAN / Graph2Plan Preprocessing Report")
    report_lines.append("")
    report_lines.append(f"Generated at: `{summary['timestamp']}`")
    report_lines.append("")
    report_lines.append("## Input and Output Paths")
    report_lines.append("")
    report_lines.append(f"- Dataset root: `{paths.dataset_root}`")
    report_lines.append(f"- Processed root: `{paths.processed_root}`")
    report_lines.append(f"- Audit root: `{paths.audit_root}`")
    report_lines.append("")
    report_lines.append("## Converted Data")
    report_lines.append("")
    report_lines.append(f"- Available PKL samples: `{layout_summary['total_available']}`")
    report_lines.append(f"- Converted layouts: `{layout_summary['num_converted_layouts']}`")
    report_lines.append(f"- Converted graphs: `{graph_summary['num_converted_graphs']}`")
    report_lines.append(f"- Layout JSON directory: `{paths.layouts_dir}`")
    report_lines.append(f"- Graph JSON directory: `{paths.graphs_dir}`")
    report_lines.append(f"- Metadata directory: `{paths.metadata_dir}`")
    report_lines.append("")
    report_lines.append("## Coordinate System")
    report_lines.append("")
    report_lines.append("The converted layouts use a 256 x 256 image coordinate system.")
    report_lines.append("")
    report_lines.append("- Origin: top-left")
    report_lines.append("- x-axis: grows to the right")
    report_lines.append("- y-axis: grows downward")
    report_lines.append("- Edge relation: source room relative to target room")
    report_lines.append("")
    report_lines.append("This is why visualizations invert the y-axis with `ylim(256, 0)`.")
    report_lines.append("")
    report_lines.append("## Room Type Mapping")
    report_lines.append("")
    report_lines.append("| ID | Room type |")
    report_lines.append("|---:|---|")
    for type_id, name in ROOM_TYPE_ID_TO_NAME.items():
        report_lines.append(f"| {type_id} | {name} |")
    report_lines.append("")
    report_lines.append("## Spatial Relation Mapping")
    report_lines.append("")
    report_lines.append("| ID | Relation |")
    report_lines.append("|---:|---|")
    for rel_id, name in RELATION_ID_TO_NAME.items():
        report_lines.append(f"| {rel_id} | {name} |")
    report_lines.append("")
    report_lines.append("## Dataset Statistics")
    report_lines.append("")
    report_lines.append(f"- Number of layouts: `{stats['num_layouts']}`")
    report_lines.append(f"- Room count per plan: `{stats['room_count_per_plan']}`")
    report_lines.append(f"- Edge count per plan: `{stats['edge_count_per_plan']}`")
    report_lines.append("")
    report_lines.append("### Room type counts")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(stats["room_type_name_counts"], indent=2))
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("### Relation counts")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(stats["relation_type_counts"], indent=2))
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("## Visual Quality Control")
    report_lines.append("")
    report_lines.append(f"- Layout preview count: `{preview_summary['layout_preview_count']}`")
    report_lines.append(f"- Graph preview count: `{preview_summary['graph_preview_count']}`")
    report_lines.append(f"- Layout preview directory: `{paths.layout_previews_dir}`")
    report_lines.append(f"- Graph preview directory: `{paths.graph_previews_dir}`")
    report_lines.append("")
    report_lines.append("Preview images are not model inputs. They are only generated for debugging and documentation.")
    report_lines.append("")
    report_lines.append("## Main Output Files")
    report_lines.append("")
    report_lines.append(f"- `{paths.audit_root / 'rplan_file_audit.json'}`")
    report_lines.append(f"- `{paths.audit_root / 'train_converted_sample_report.json'}`")
    report_lines.append(f"- `{paths.audit_root / 'rplan_processed_statistics.json'}`")
    report_lines.append(f"- `{paths.audit_root / 'preprocessing_summary.json'}`")
    report_lines.append(f"- `{paths.audit_root / 'preprocessing_report.md'}`")

    report_path = paths.audit_root / "preprocessing_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"  Saved: {paths.audit_root / 'preprocessing_summary.json'}")
    print(f"  Saved: {report_path}")

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_id_list(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated IDs passed from the command line."""
    if value is None or value.strip() == "":
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified preprocessing pipeline for RPLAN / Graph2Plan data."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/rplan/extracted"),
        help="Root directory containing the unzipped Graph2Plan/RPLAN data.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/rplan"),
        help="Output root for layout and graph JSON files.",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("outputs/dataset_audit"),
        help="Output root for reports, statistics, and preview images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N samples. Useful for debugging before a full run.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete previous processed outputs and audit reports before running.",
    )
    parser.add_argument(
        "--layout-preview-n",
        type=int,
        default=50,
        help="Number of random layout comparison previews to generate.",
    )
    parser.add_argument(
        "--graph-preview-n",
        type=int,
        default=20,
        help="Number of random graph comparison previews to generate.",
    )
    parser.add_argument(
        "--layout-preview-ids",
        type=str,
        default=None,
        help="Optional comma-separated layout IDs to preview instead of random IDs.",
    )
    parser.add_argument(
        "--graph-preview-ids",
        type=str,
        default=None,
        help="Optional comma-separated graph IDs to preview instead of random IDs.",
    )
    parser.add_argument(
        "--preview-seed",
        type=int,
        default=42,
        help="Random seed used for selecting preview examples.",
    )
    parser.add_argument(
        "--skip-previews",
        action="store_true",
        help="Skip layout and graph preview image generation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = PipelinePaths(
        dataset_root=args.dataset_root,
        processed_root=args.processed_root,
        audit_root=args.audit_root,
    )

    if args.clean:
        print("Cleaning previous generated outputs...")
        clean_outputs(paths)
    else:
        paths.ensure_dirs()

    inspect_summary = inspect_dataset(paths)
    layout_summary = convert_layouts(paths, limit=args.limit)
    graph_summary = convert_graphs(paths)
    stats = compute_statistics(paths)

    if args.skip_previews:
        preview_summary = {
            "layout_preview_count": 0,
            "graph_preview_count": 0,
            "layout_preview_ids": [],
            "graph_preview_ids": [],
            "layout_preview_files": [],
            "graph_preview_files": [],
        }
    else:
        preview_summary = generate_previews(
            paths=paths,
            layout_preview_n=args.layout_preview_n,
            graph_preview_n=args.graph_preview_n,
            preview_seed=args.preview_seed,
            layout_preview_ids=parse_id_list(args.layout_preview_ids),
            graph_preview_ids=parse_id_list(args.graph_preview_ids),
        )

    write_report(
        paths=paths,
        args=args,
        inspect_summary=inspect_summary,
        layout_summary=layout_summary,
        graph_summary=graph_summary,
        stats=stats,
        preview_summary=preview_summary,
    )

    print("\nPreprocessing completed successfully.")
    print(f"Layout JSON files: {paths.layouts_dir}")
    print(f"Graph JSON files:  {paths.graphs_dir}")
    print(f"Report:            {paths.audit_root / 'preprocessing_report.md'}")


if __name__ == "__main__":
    main()
