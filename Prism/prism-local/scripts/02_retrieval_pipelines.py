#!/usr/bin/env python3
"""
Post-preprocessing retrieval pipelines for RPLAN / Graph2Plan floorplan search.

This script implements the two retrieval branches used after preprocessing:

1. Graph retrieval pipeline
   - Input:  data/processed/rplan/graphs_json/*.json
             data/processed/rplan/layouts_json/*.json
   - Encoder: hand-crafted structural descriptor from the layout graph
   - Index: FAISS vector index
   - Output: top-k similar floorplans, JSON results, and grid visualizations

2. Vision retrieval pipeline
   - Input:  data/raw/rplan/extracted/Interface/static/Data/snapshot_train/*.png
   - Encoder: ResNet / DINOv2 / timm / color-histogram descriptor
   - Index: FAISS vector index
   - Output: top-k similar floorplans, JSON results, and grid visualizations

The script is designed to be run from the project root, for example:

  python scripts/02_retrieval_pipelines.py \
    --pipeline both \
    --build \
    --processed-root data/processed/rplan \
    --outputs-root outputs/retrieval \
    --vision-backbone resnet50 \
    --vision-weights imagenet \
    --random-queries 10 \
    --top-k 10

The implementation intentionally keeps the graph encoder simple and interpretable.
The final retrieval stage uses a vector similarity index for both graph and
vision vectors, so the same indexing/search interface is shared by the two
pipelines. The vision encoder can run on CUDA when a compatible GPU and
PyTorch installation are available.
"""

from __future__ import annotations

import logging

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("02_retrieval_pipelines.py")


def print(*args: object, **kwargs: object) -> None:  # noqa: A001 - compatibility shim for legacy progress output
    """Route legacy progress output through logging."""
    sep = str(kwargs.get("sep", " "))
    logger.info(sep.join(str(arg) for arg in args))


# Use a headless backend so the script also works on clusters and SSH sessions.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# -----------------------------------------------------------------------------
# RPLAN / Graph2Plan constants
# -----------------------------------------------------------------------------

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

ROOM_TYPE_IDS: List[int] = sorted(ROOM_TYPE_ID_TO_NAME.keys())
RELATION_IDS: List[int] = sorted(RELATION_ID_TO_NAME.keys())


# -----------------------------------------------------------------------------
# Paths and configuration
# -----------------------------------------------------------------------------

@dataclass
class PipelinePaths:
    """Centralized path container used by all retrieval stages."""

    dataset_root: Path
    processed_root: Path
    outputs_root: Path

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
    def retrieval_root(self) -> Path:
        return self.processed_root / "retrieval"

    @property
    def graph_artifact_dir(self) -> Path:
        return self.retrieval_root / "graph_pipeline"

    @property
    def vision_artifact_dir(self) -> Path:
        return self.retrieval_root / "vision_pipeline"

    @property
    def graph_output_dir(self) -> Path:
        return self.outputs_root / "graph_pipeline"

    @property
    def vision_output_dir(self) -> Path:
        return self.outputs_root / "vision_pipeline"

    @property
    def comparison_output_dir(self) -> Path:
        return self.outputs_root / "pipeline_comparison"

    def ensure_dirs(self) -> None:
        """Create all directories used by the retrieval pipelines."""
        for path in [
            self.retrieval_root,
            self.graph_artifact_dir,
            self.vision_artifact_dir,
            self.graph_output_dir,
            self.vision_output_dir,
            self.comparison_output_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def clean_pipeline_outputs(self) -> None:
        """Remove artifacts generated by this retrieval script only."""
        for path in [
            self.graph_artifact_dir,
            self.vision_artifact_dir,
            self.graph_output_dir,
            self.vision_output_dir,
            self.comparison_output_dir,
            self.outputs_root / "random_queries.csv",
            self.outputs_root / "retrieval_pipelines_report.md",
        ]:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        self.ensure_dirs()


# -----------------------------------------------------------------------------
# Generic file and vector utilities
# -----------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    """Save a human-readable UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    """Save a list of dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return row-wise L2-normalized vectors."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def standardize_features(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Standardize features and return the fitted scaler statistics."""
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    X_scaled = ((X - mean) / std).astype(np.float32)
    scaler = {
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
        "eps": eps,
    }
    return X_scaled, scaler


def require_faiss():
    """Import FAISS lazily and provide a clear error message if it is missing."""
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for this script. Install it with one of:\n"
            "  pip install faiss-cpu\n"
            "  conda install -c pytorch -c conda-forge faiss-cpu\n"
            "Then re-run the command."
        ) from exc
    return faiss


def build_faiss_index(
    vectors: np.ndarray,
    metric: str = "cosine",
    index_type: str = "flat",
    hnsw_m: int = 32,
):
    """
    Build a FAISS index from vectors.

    For cosine retrieval, vectors are L2-normalized and indexed with inner product.
    For L2 retrieval, raw vectors are indexed with squared Euclidean distance.
    """
    faiss = require_faiss()
    vectors = np.asarray(vectors, dtype=np.float32)

    if metric == "cosine":
        index_vectors = l2_normalize(vectors)
        faiss_metric = "inner_product"
    elif metric == "l2":
        index_vectors = vectors.astype(np.float32)
        faiss_metric = "l2"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    dim = index_vectors.shape[1]

    if index_type == "flat":
        if metric == "cosine":
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
    elif index_type == "hnsw":
        # HNSW is useful when the dataset grows beyond exact-search comfort.
        # For this project, flat exact search is already fine for ~75k layouts,
        # but HNSW is included as an optional scalable alternative.
        if metric == "cosine":
            index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_L2)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
    else:
        raise ValueError(f"Unsupported FAISS index type: {index_type}")

    index.add(index_vectors)
    return index, index_vectors.astype(np.float32), faiss_metric


def write_faiss_index(index: Any, path: Path) -> None:
    """Persist a FAISS index to disk."""
    faiss = require_faiss()
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def read_faiss_index(path: Path) -> Any:
    """Load a FAISS index from disk."""
    faiss = require_faiss()
    if not path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {path}")
    return faiss.read_index(str(path))


def faiss_search_existing_vector(
    index: Any,
    vectors_for_search: np.ndarray,
    query_index: int,
    ids: Sequence[str],
    top_k: int,
    metric: str,
) -> List[Dict[str, Any]]:
    """Search a FAISS index using an existing item vector as query."""
    query = vectors_for_search[query_index:query_index + 1].astype(np.float32)
    distances, indices = index.search(query, top_k + 1)

    results: List[Dict[str, Any]] = []
    for value, idx in zip(distances[0], indices[0]):
        idx = int(idx)
        if idx < 0 or idx == query_index:
            continue

        if metric == "cosine":
            similarity = float(value)
            distance = float(1.0 - value)
        else:
            distance = float(value)
            similarity = float(-value)

        results.append({
            "index": idx,
            "id": str(ids[idx]),
            "similarity": similarity,
            "distance": distance,
        })

        if len(results) >= top_k:
            break

    return results


def find_id_index(ids: Sequence[str], query_id: str) -> int:
    """Return the integer position of a layout ID."""
    query_id = str(query_id)
    lookup = {str(item_id): i for i, item_id in enumerate(ids)}
    if query_id not in lookup:
        raise ValueError(f"Query id '{query_id}' is not available in this pipeline.")
    return lookup[query_id]


def maybe_limit_paths(paths: List[Path], max_items: Optional[int]) -> List[Path]:
    """Limit a sorted path list for local smoke tests."""
    if max_items is None:
        return paths
    return paths[:max_items]


# -----------------------------------------------------------------------------
# Graph feature extraction
# -----------------------------------------------------------------------------

def room_type_id(node: Dict[str, Any]) -> int:
    """Read the room type id with backward compatibility for older outputs."""
    if "room_type_id" in node:
        return int(node["room_type_id"])
    if "raw_type_id" in node:
        return int(node["raw_type_id"])
    return -1


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Compute polygon area with the shoelace formula."""
    if len(points) < 3:
        return 0.0
    x = np.asarray([p[0] for p in points], dtype=np.float64)
    y = np.asarray([p[1] for p in points], dtype=np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def boundary_descriptor(layout: Optional[Dict[str, Any]]) -> Tuple[List[float], List[str]]:
    """Extract simple external-boundary features from a layout JSON object."""
    names = [
        "boundary_width",
        "boundary_height",
        "boundary_aspect_ratio",
        "boundary_area_ratio",
        "boundary_num_vertices",
    ]

    if layout is None or not layout.get("boundary"):
        return [0.0, 0.0, 0.0, 0.0, 0.0], names

    points = [[float(p[0]), float(p[1])] for p in layout["boundary"] if len(p) >= 2]
    if len(points) < 3:
        return [0.0, 0.0, 0.0, 0.0, float(len(points))], names

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    aspect = width / max(height, 1e-8)
    area_ratio = polygon_area(points) / (256.0 * 256.0)
    return [float(width), float(height), float(aspect), float(area_ratio), float(len(points))], names


def build_graph_feature_vector(
    graph: Dict[str, Any],
    layout: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a layout graph into an interpretable structural feature vector.

    The feature vector intentionally combines semantic, topological, geometric,
    and boundary information. This is not a learned GNN embedding yet; it is a
    strong transparent baseline used before moving to trainable graph encoders.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    features: List[float] = []
    names: List[str] = []

    num_rooms = len(nodes)
    num_edges = len(edges)
    max_possible_edges = max(1, num_rooms * (num_rooms - 1) / 2.0)

    features.extend([float(num_rooms), float(num_edges), float(num_edges / max_possible_edges)])
    names.extend(["num_rooms", "num_edges", "edge_density"])

    # Room type histogram and normalized room type distribution.
    type_counter = Counter(room_type_id(node) for node in nodes)
    for type_id in ROOM_TYPE_IDS:
        features.append(float(type_counter.get(type_id, 0)))
        names.append(f"room_type_count_{type_id}_{ROOM_TYPE_ID_TO_NAME[type_id]}")
    for type_id in ROOM_TYPE_IDS:
        features.append(float(type_counter.get(type_id, 0) / max(1, num_rooms)))
        names.append(f"room_type_ratio_{type_id}_{ROOM_TYPE_ID_TO_NAME[type_id]}")

    # Relation histogram and normalized relation distribution.
    relation_counter = Counter(int(edge.get("relation_id", -1)) for edge in edges)
    for relation_id in RELATION_IDS:
        features.append(float(relation_counter.get(relation_id, 0)))
        names.append(f"relation_count_{relation_id}_{RELATION_ID_TO_NAME[relation_id]}")
    for relation_id in RELATION_IDS:
        features.append(float(relation_counter.get(relation_id, 0) / max(1, num_edges)))
        names.append(f"relation_ratio_{relation_id}_{RELATION_ID_TO_NAME[relation_id]}")

    # Area distribution.
    areas = np.asarray([float(node.get("area_ratio", 0.0)) for node in nodes], dtype=np.float32)
    if len(areas) > 0:
        area_stats = [float(areas.mean()), float(areas.std()), float(areas.min()), float(areas.max())]
    else:
        area_stats = [0.0, 0.0, 0.0, 0.0]
    features.extend(area_stats)
    names.extend(["area_mean", "area_std", "area_min", "area_max"])

    # Area by room type.
    area_by_type: Dict[int, List[float]] = defaultdict(list)
    for node in nodes:
        area_by_type[room_type_id(node)].append(float(node.get("area_ratio", 0.0)))
    for type_id in ROOM_TYPE_IDS:
        values = area_by_type.get(type_id, [])
        features.append(float(np.sum(values)) if values else 0.0)
        names.append(f"area_sum_{type_id}_{ROOM_TYPE_ID_TO_NAME[type_id]}")
        features.append(float(np.mean(values)) if values else 0.0)
        names.append(f"area_mean_{type_id}_{ROOM_TYPE_ID_TO_NAME[type_id]}")

    # Centroid distribution in image coordinates: x grows right, y grows down.
    centroids = np.asarray([node.get("centroid", [0.0, 0.0]) for node in nodes], dtype=np.float32)
    if len(centroids) > 0:
        centroid_stats = [
            float(centroids[:, 0].mean() / 256.0),
            float(centroids[:, 1].mean() / 256.0),
            float(centroids[:, 0].std() / 256.0),
            float(centroids[:, 1].std() / 256.0),
        ]
    else:
        centroid_stats = [0.0, 0.0, 0.0, 0.0]
    features.extend(centroid_stats)
    names.extend(["centroid_x_mean", "centroid_y_mean", "centroid_x_std", "centroid_y_std"])

    # Bounding-box shape statistics.
    widths = []
    heights = []
    aspect_ratios = []
    for node in nodes:
        bbox = node.get("bbox", [0, 0, 0, 0])
        w = max(0.0, float(bbox[2]) - float(bbox[0]))
        h = max(0.0, float(bbox[3]) - float(bbox[1]))
        widths.append(w / 256.0)
        heights.append(h / 256.0)
        aspect_ratios.append(w / max(h, 1e-8))

    for values, prefix in [(widths, "bbox_width"), (heights, "bbox_height"), (aspect_ratios, "bbox_aspect")]:
        arr = np.asarray(values, dtype=np.float32)
        if len(arr) > 0:
            features.extend([float(arr.mean()), float(arr.std()), float(arr.max())])
        else:
            features.extend([0.0, 0.0, 0.0])
        names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_max"])

    # Graph degree statistics.
    degree_counter = Counter()
    for edge in edges:
        degree_counter[int(edge.get("source", -1))] += 1
        degree_counter[int(edge.get("target", -1))] += 1
    degrees = np.asarray([degree_counter[i] for i in range(num_rooms)], dtype=np.float32)
    if len(degrees) > 0:
        degree_stats = [float(degrees.mean()), float(degrees.std()), float(degrees.max())]
    else:
        degree_stats = [0.0, 0.0, 0.0]
    features.extend(degree_stats)
    names.extend(["degree_mean", "degree_std", "degree_max"])

    # Boundary descriptor from the associated layout file.
    boundary_features, boundary_names = boundary_descriptor(layout)
    features.extend(boundary_features)
    names.extend(boundary_names)

    return np.asarray(features, dtype=np.float32), names


def build_graph_pipeline(
    paths: PipelinePaths,
    max_items: Optional[int],
    metric: str,
    index_type: str,
) -> Dict[str, Any]:
    """Build graph descriptors, normalize them, and store a FAISS index."""
    graph_paths = sorted(paths.graphs_dir.glob("*.json"))
    graph_paths = maybe_limit_paths(graph_paths, max_items)
    if not graph_paths:
        raise RuntimeError(f"No graph JSON files found in {paths.graphs_dir}")

    ids: List[str] = []
    vectors: List[np.ndarray] = []
    feature_names_ref: Optional[List[str]] = None

    for graph_path in tqdm(graph_paths, desc="Building graph descriptors"):
        graph = load_json(graph_path)
        layout_path = paths.layouts_dir / f"{graph_path.stem}.json"
        layout = load_json(layout_path) if layout_path.exists() else None

        vector, feature_names = build_graph_feature_vector(graph, layout)
        if feature_names_ref is None:
            feature_names_ref = feature_names
        elif feature_names != feature_names_ref:
            raise RuntimeError("Graph feature names changed across samples.")

        ids.append(str(graph.get("id", graph_path.stem)))
        vectors.append(vector)

    X_raw = np.vstack(vectors).astype(np.float32)
    X_scaled, scaler = standardize_features(X_raw)

    index, X_index, faiss_metric = build_faiss_index(X_scaled, metric=metric, index_type=index_type)

    artifact_dir = paths.graph_artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "features_raw.npy", X_raw)
    np.save(artifact_dir / "features_scaled.npy", X_scaled)
    np.save(artifact_dir / "features_index.npy", X_index)
    save_json(ids, artifact_dir / "ids.json")
    save_json(feature_names_ref or [], artifact_dir / "feature_names.json")
    save_json(scaler, artifact_dir / "scaler.json")
    write_faiss_index(index, artifact_dir / "faiss.index")

    metadata = {
        "pipeline": "graph",
        "num_items": len(ids),
        "feature_dim": int(X_raw.shape[1]),
        "metric": metric,
        "faiss_metric": faiss_metric,
        "faiss_index_type": index_type,
        "artifact_dir": str(artifact_dir),
        "created_at_unix": time.time(),
    }
    save_json(metadata, artifact_dir / "metadata.json")
    return metadata


# -----------------------------------------------------------------------------
# Vision feature extraction
# -----------------------------------------------------------------------------

def collect_vision_items(paths: PipelinePaths, max_items: Optional[int]) -> List[Tuple[str, Path]]:
    """Collect snapshot images whose IDs also have preprocessed layout files."""
    if not paths.snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {paths.snapshot_dir}")

    image_paths = sorted(paths.snapshot_dir.glob("*.png"))
    items: List[Tuple[str, Path]] = []
    for image_path in image_paths:
        sample_id = image_path.stem
        # Keep the vision branch aligned with the preprocessed layout/graph IDs.
        if (paths.layouts_dir / f"{sample_id}.json").exists():
            items.append((sample_id, image_path))

    if max_items is not None:
        items = items[:max_items]
    if not items:
        raise RuntimeError("No usable vision items were found.")
    return items


def extract_colorhist_features(
    items: Sequence[Tuple[str, Path]],
    image_size: int,
    bins: int = 8,
    grid_size: int = 8,
) -> np.ndarray:
    """
    Extract a spatial RGB histogram descriptor for CPU-only vision retrieval.

    A single global color histogram is too weak for RPLAN snapshots because many
    floorplans share the same palette and similar room-color proportions. That
    makes cosine similarities collapse close to 1.0 for many unrelated items.

    This descriptor keeps the same CPU-only spirit, but encodes where colors are
    located by computing RGB histograms inside an image grid. The output captures
    coarse spatial layout information without requiring PyTorch, CUDA, ResNet,
    DINO, or timm.
    """
    vectors: List[np.ndarray] = []

    for _, image_path in tqdm(items, desc="Extracting spatial color-histogram features"):
        img = Image.open(image_path).convert("RGB")

        # Use a fixed canvas so every descriptor has the same dimensionality.
        # Padding preserves the original aspect ratio better than direct resize.
        img = ImageOps.pad(
            img,
            (image_size, image_size),
            method=Image.Resampling.BILINEAR,
            color=(255, 255, 255),
        )

        arr = np.asarray(img, dtype=np.float32) / 255.0
        cell_h = image_size // grid_size
        cell_w = image_size // grid_size

        parts: List[np.ndarray] = []

        # 1) Global color statistics.
        for channel in range(3):
            hist, _ = np.histogram(
                arr[:, :, channel],
                bins=bins,
                range=(0.0, 1.0),
                density=False,
            )
            hist = hist.astype(np.float32)
            hist = hist / max(float(hist.sum()), 1.0)
            parts.append(hist)

        # 2) Spatial color histograms over a regular grid.
        for gy in range(grid_size):
            y0 = gy * cell_h
            y1 = image_size if gy == grid_size - 1 else (gy + 1) * cell_h

            for gx in range(grid_size):
                x0 = gx * cell_w
                x1 = image_size if gx == grid_size - 1 else (gx + 1) * cell_w
                patch = arr[y0:y1, x0:x1, :]

                for channel in range(3):
                    hist, _ = np.histogram(
                        patch[:, :, channel],
                        bins=bins,
                        range=(0.0, 1.0),
                        density=False,
                    )
                    hist = hist.astype(np.float32)
                    hist = hist / max(float(hist.sum()), 1.0)
                    parts.append(hist)

        # 3) Simple foreground occupancy grid. This reduces the dominance of the
        # white background and helps distinguish compact vs elongated layouts.
        gray = arr.mean(axis=2)
        foreground = (gray < 0.97).astype(np.float32)
        occupancy: List[float] = []
        for gy in range(grid_size):
            y0 = gy * cell_h
            y1 = image_size if gy == grid_size - 1 else (gy + 1) * cell_h
            for gx in range(grid_size):
                x0 = gx * cell_w
                x1 = image_size if gx == grid_size - 1 else (gx + 1) * cell_w
                occupancy.append(float(foreground[y0:y1, x0:x1].mean()))
        parts.append(np.asarray(occupancy, dtype=np.float32))

        vector = np.concatenate(parts).astype(np.float32)
        vectors.append(vector)

    return np.vstack(vectors).astype(np.float32)


def make_torchvision_encoder(backbone: str, weights_policy: str, device: str, image_size: int):
    """Create a torchvision image encoder and its preprocessing transform."""
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import (
        ResNet18_Weights,
        ResNet50_Weights,
        resnet18,
        resnet50,
    )

    if backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT if weights_policy == "imagenet" else None
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
    elif backbone == "resnet50":
        weights = ResNet50_Weights.DEFAULT if weights_policy == "imagenet" else None
        model = resnet50(weights=weights)
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported torchvision backbone: {backbone}")

    if weights_policy == "imagenet" and weights is not None:
        transform = weights.transforms()
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    model.eval().to(device)
    return model, transform


def make_dinov2_encoder(backbone: str, device: str, image_size: int):
    """Create a DINOv2 encoder through torch.hub."""
    import torch
    from torchvision import transforms

    allowed = {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"}
    if backbone not in allowed:
        raise ValueError(f"Unsupported DINOv2 backbone: {backbone}")

    model = torch.hub.load(
        "facebookresearch/dinov2",
        backbone,
        trust_repo=True,
        skip_validation=True,
    )
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def make_timm_encoder(backbone: str, weights_policy: str, device: str, image_size: int):
    """Create a timm encoder. The backbone string must be formatted as timm:<model>."""
    import timm
    from timm.data import create_transform, resolve_model_data_config

    model_name = backbone.split(":", 1)[1]
    pretrained = weights_policy == "imagenet"
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    model.eval().to(device)

    if pretrained:
        data_config = resolve_model_data_config(model)
        transform = create_transform(**data_config, is_training=False)
    else:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return model, transform


def extract_torch_features(
    items: Sequence[Tuple[str, Path]],
    backbone: str,
    weights_policy: str,
    batch_size: int,
    device: str,
    image_size: int,
) -> np.ndarray:
    """Extract image embeddings using a torch-based backbone."""
    import torch

    if backbone in {"resnet18", "resnet50"}:
        model, transform = make_torchvision_encoder(backbone, weights_policy, device, image_size)
    elif backbone.startswith("dinov2_"):
        model, transform = make_dinov2_encoder(backbone, device, image_size)
    elif backbone.startswith("timm:"):
        model, transform = make_timm_encoder(backbone, weights_policy, device, image_size)
    else:
        raise ValueError(f"Unsupported vision backbone: {backbone}")

    vectors: List[np.ndarray] = []
    batch_tensors = []

    with torch.no_grad():
        for _, image_path in tqdm(items, desc=f"Extracting vision features ({backbone})"):
            img = Image.open(image_path).convert("RGB")
            tensor = transform(img)
            batch_tensors.append(tensor)

            if len(batch_tensors) >= batch_size:
                batch = torch.stack(batch_tensors, dim=0).to(device)
                feats = model(batch)
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
                feats = feats.detach().cpu().float().numpy()
                vectors.append(feats)
                batch_tensors = []

        if batch_tensors:
            batch = torch.stack(batch_tensors, dim=0).to(device)
            feats = model(batch)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats.detach().cpu().float().numpy()
            vectors.append(feats)

    return np.vstack(vectors).astype(np.float32)



def resolve_torch_device(requested_device: str, require_torch: bool) -> str:
    """Resolve the requested device string into a concrete torch device name.

    Parameters
    ----------
    requested_device:
        One of ``auto``, ``cuda`` or ``cpu``. ``auto`` uses CUDA when PyTorch
        can see a CUDA-capable GPU, otherwise it falls back to CPU.
    require_torch:
        Whether the selected vision encoder requires PyTorch. The CPU-only
        spatial color histogram does not require PyTorch.
    """
    requested_device = requested_device.lower()
    if requested_device not in {"auto", "cuda", "cpu"}:
        raise ValueError("--device must be one of: auto, cuda, cpu")

    if not require_torch:
        return "cpu"

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The selected vision backbone requires PyTorch. Install torch and "
            "torchvision, or use --vision-backbone colorhist for the CPU-only "
            "descriptor."
        ) from exc

    if requested_device == "cpu":
        return "cpu"

    cuda_available = torch.cuda.is_available()
    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was explicitly requested with --device cuda, but "
            "torch.cuda.is_available() is False. Use --device cpu or fix the "
            "PyTorch/CUDA installation."
        )

    if requested_device == "auto":
        return "cuda" if cuda_available else "cpu"

    return "cuda"

def build_vision_pipeline(
    paths: PipelinePaths,
    max_items: Optional[int],
    metric: str,
    index_type: str,
    backbone: str,
    weights_policy: str,
    batch_size: int,
    device: str,
    image_size: int,
) -> Dict[str, Any]:
    """Build image descriptors and store a FAISS index."""
    items = collect_vision_items(paths, max_items)
    ids = [item_id for item_id, _ in items]

    if backbone == "colorhist":
        X_raw = extract_colorhist_features(items, image_size=image_size)
        encoder_info = {"backbone": backbone, "weights": "none", "descriptor": "spatial_color_histogram", "color_bins": 8, "grid_size": 8}
    else:
        X_raw = extract_torch_features(
            items=items,
            backbone=backbone,
            weights_policy=weights_policy,
            batch_size=batch_size,
            device=device,
            image_size=image_size,
        )
        encoder_info = {"backbone": backbone, "weights": weights_policy, "device": device}

    # Most deep descriptors are already meaningful in their native space. We do
    # not standardize vision embeddings by default; cosine FAISS uses L2-normalized
    # descriptors at the indexing stage.
    index, X_index, faiss_metric = build_faiss_index(X_raw, metric=metric, index_type=index_type)

    artifact_dir = paths.vision_artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "features_raw.npy", X_raw)
    np.save(artifact_dir / "features_index.npy", X_index)
    save_json(ids, artifact_dir / "ids.json")
    write_faiss_index(index, artifact_dir / "faiss.index")

    metadata = {
        "pipeline": "vision",
        "num_items": len(ids),
        "feature_dim": int(X_raw.shape[1]),
        "metric": metric,
        "faiss_metric": faiss_metric,
        "faiss_index_type": index_type,
        "encoder": encoder_info,
        "image_size": int(image_size),
        "artifact_dir": str(artifact_dir),
        "created_at_unix": time.time(),
    }
    save_json(metadata, artifact_dir / "metadata.json")
    return metadata


# -----------------------------------------------------------------------------
# Retrieval and visualization
# -----------------------------------------------------------------------------

def load_pipeline_artifacts(artifact_dir: Path) -> Tuple[Any, np.ndarray, List[str], Dict[str, Any]]:
    """Load the FAISS index, search vectors, IDs, and metadata for a pipeline."""
    index = read_faiss_index(artifact_dir / "faiss.index")
    vectors = np.load(artifact_dir / "features_index.npy")
    ids = load_json(artifact_dir / "ids.json")
    metadata = load_json(artifact_dir / "metadata.json")
    return index, vectors, ids, metadata


def retrieve_from_pipeline(
    artifact_dir: Path,
    query_id: str,
    top_k: int,
) -> Dict[str, Any]:
    """Run top-k retrieval for a query ID using one built FAISS pipeline."""
    index, vectors, ids, metadata = load_pipeline_artifacts(artifact_dir)
    query_index = find_id_index(ids, query_id)
    results = faiss_search_existing_vector(
        index=index,
        vectors_for_search=vectors,
        query_index=query_index,
        ids=ids,
        top_k=top_k,
        metric=metadata["metric"],
    )

    return {
        "query_id": str(query_id),
        "pipeline": metadata["pipeline"],
        "metric": metadata["metric"],
        "faiss_index_type": metadata["faiss_index_type"],
        "results": [
            {
                "rank": rank + 1,
                "id": item["id"],
                "similarity": item["similarity"],
                "distance": item["distance"],
            }
            for rank, item in enumerate(results)
        ],
    }


def snapshot_path(paths: PipelinePaths, sample_id: str) -> Path:
    """Return the expected original snapshot path for a floorplan ID."""
    return paths.snapshot_dir / f"{sample_id}.png"


def draw_retrieval_grid(
    paths: PipelinePaths,
    result_obj: Dict[str, Any],
    output_path: Path,
    max_cols: int = 4,
) -> None:
    """Draw a query + top-k retrieval grid using original snapshot images."""
    query_id = result_obj["query_id"]
    pipeline = result_obj["pipeline"]

    items: List[Dict[str, Any]] = [{
        "id": query_id,
        "title": f"Query\nID: {query_id}",
    }]
    for result in result_obj["results"]:
        items.append({
            "id": result["id"],
            "title": f"Top {result['rank']}\nID: {result['id']}\nscore={result['similarity']:.6f}",
        })

    cols = min(max_cols, len(items))
    rows = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, item in zip(axes, items):
        img_path = snapshot_path(paths, item["id"])
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Missing image\n{item['id']}", ha="center", va="center")
        ax.set_title(item["title"], fontsize=9)
        ax.axis("off")

    for ax in axes[len(items):]:
        ax.axis("off")

    fig.suptitle(f"{pipeline.capitalize()} retrieval", fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_pipeline_comparison(
    paths: PipelinePaths,
    graph_result: Optional[Dict[str, Any]],
    vision_result: Optional[Dict[str, Any]],
    output_path: Path,
    top_k_to_show: int = 5,
) -> None:
    """Draw a compact side-by-side comparison of graph and vision retrieval."""
    available = [("graph", graph_result), ("vision", vision_result)]
    available = [(name, obj) for name, obj in available if obj is not None]
    if not available:
        return

    query_id = available[0][1]["query_id"]
    rows = len(available)
    cols = top_k_to_show + 1
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    for row, (pipeline_name, result_obj) in enumerate(available):
        ids_to_show = [query_id] + [r["id"] for r in result_obj["results"][:top_k_to_show]]
        titles = [f"{pipeline_name}\nQuery\n{query_id}"] + [
            f"Top {r['rank']}\n{r['id']}\n{r['similarity']:.6f}"
            for r in result_obj["results"][:top_k_to_show]
        ]

        for col, (sample_id, title) in enumerate(zip(ids_to_show, titles)):
            ax = axes[row, col]
            img_path = snapshot_path(paths, sample_id)
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"Missing\n{sample_id}", ha="center", va="center")
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def run_query(
    paths: PipelinePaths,
    pipeline: str,
    query_id: str,
    top_k: int,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Run retrieval for one query in the selected pipeline(s)."""
    results: Dict[str, Optional[Dict[str, Any]]] = {"graph": None, "vision": None}

    if pipeline in {"graph", "both"}:
        graph_result = retrieve_from_pipeline(paths.graph_artifact_dir, query_id, top_k)
        save_json(graph_result, paths.graph_output_dir / f"{query_id}_top{top_k}.json")
        draw_retrieval_grid(paths, graph_result, paths.graph_output_dir / f"{query_id}_top{top_k}.png")
        results["graph"] = graph_result

    if pipeline in {"vision", "both"}:
        vision_result = retrieve_from_pipeline(paths.vision_artifact_dir, query_id, top_k)
        save_json(vision_result, paths.vision_output_dir / f"{query_id}_top{top_k}.json")
        draw_retrieval_grid(paths, vision_result, paths.vision_output_dir / f"{query_id}_top{top_k}.png")
        results["vision"] = vision_result

    if pipeline == "both" and (results["graph"] is not None or results["vision"] is not None):
        draw_pipeline_comparison(
            paths=paths,
            graph_result=results["graph"],
            vision_result=results["vision"],
            output_path=paths.comparison_output_dir / f"{query_id}_comparison.png",
        )

    return results


def available_query_ids(paths: PipelinePaths, pipeline: str) -> List[str]:
    """Return IDs available for the requested pipeline(s)."""
    id_sets: List[set] = []

    if pipeline in {"graph", "both"}:
        graph_ids = set(load_json(paths.graph_artifact_dir / "ids.json"))
        id_sets.append(graph_ids)
    if pipeline in {"vision", "both"}:
        vision_ids = set(load_json(paths.vision_artifact_dir / "ids.json"))
        id_sets.append(vision_ids)

    if not id_sets:
        return []
    ids = sorted(set.intersection(*id_sets)) if len(id_sets) > 1 else sorted(id_sets[0])
    return ids


# -----------------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------------

def write_report(paths: PipelinePaths, args: argparse.Namespace, build_metadata: Dict[str, Any]) -> None:
    """Write a standalone Markdown report for the retrieval run."""
    lines = []
    lines.append("# RPLAN Post-Preprocessing Retrieval Report")
    lines.append("")
    lines.append("This report summarizes graph and vision retrieval artifacts generated from the preprocessed RPLAN / Graph2Plan data.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Dataset root: `{paths.dataset_root}`")
    lines.append(f"- Processed root: `{paths.processed_root}`")
    lines.append(f"- Outputs root: `{paths.outputs_root}`")
    lines.append(f"- Pipeline: `{args.pipeline}`")
    lines.append(f"- FAISS metric: `{args.metric}`")
    lines.append(f"- FAISS index type: `{args.faiss_index}`")
    lines.append(f"- Top-k: `{args.top_k}`")
    lines.append("")

    if "graph" in build_metadata:
        m = build_metadata["graph"]
        lines.append("## Graph Retrieval Pipeline")
        lines.append("")
        lines.append(f"- Items: `{m['num_items']}`")
        lines.append(f"- Feature dimension: `{m['feature_dim']}`")
        lines.append(f"- Artifact directory: `{m['artifact_dir']}`")
        lines.append("- Input representation: room nodes, typed edges, area statistics, centroid distribution, boundary descriptors.")
        lines.append("")

    if "vision" in build_metadata:
        m = build_metadata["vision"]
        lines.append("## Vision Retrieval Pipeline")
        lines.append("")
        lines.append(f"- Items: `{m['num_items']}`")
        lines.append(f"- Feature dimension: `{m['feature_dim']}`")
        lines.append(f"- Encoder: `{m['encoder']}`")
        lines.append(f"- Artifact directory: `{m['artifact_dir']}`")
        lines.append("- Input representation: original rendered floorplan snapshots.")
        lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- Graph retrieval examples: `{paths.graph_output_dir}`")
    lines.append(f"- Vision retrieval examples: `{paths.vision_output_dir}`")
    lines.append(f"- Pipeline comparisons: `{paths.comparison_output_dir}`")
    lines.append("")
    lines.append("Each query produces a JSON file with ranking scores and a PNG grid with the query floorplan and its nearest neighbors.")
    lines.append("")

    report_path = paths.outputs_root / "retrieval_pipelines_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query graph and vision retrieval pipelines after RPLAN preprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/rplan/extracted"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/rplan"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/retrieval"))

    parser.add_argument("--pipeline", choices=["graph", "vision", "both"], default="both")
    parser.add_argument("--build", action="store_true", help="Build feature files and vector indexes before querying.")
    parser.add_argument("--clean", action="store_true", help="Remove artifacts generated by this script before building/querying.")
    parser.add_argument("--max-items", type=int, default=None, help="Limit the number of items for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--faiss-index", choices=["flat", "hnsw"], default="flat")
    parser.add_argument("--hnsw-m", type=int, default=32)

    parser.add_argument("--vision-backbone", type=str, default="resnet50",
                        help="Vision encoder: colorhist (CPU spatial color histogram), resnet18, resnet50, dinov2_vits14, dinov2_vitb14, dinov2_vitl14, or timm:<model>.")
    parser.add_argument("--vision-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Torch device for vision extraction. auto uses CUDA when available.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--query-id", type=str, default=None, help="Run retrieval for a specific layout ID.")
    parser.add_argument("--random-queries", type=int, default=0, help="Number of random query examples to generate.")
    parser.add_argument("--top-k", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    paths = PipelinePaths(
        dataset_root=args.dataset_root,
        processed_root=args.processed_root,
        outputs_root=args.outputs_root,
    )
    paths.ensure_dirs()
    if args.clean:
        paths.clean_pipeline_outputs()

    build_metadata: Dict[str, Any] = {}

    if args.build:
        if args.pipeline in {"graph", "both"}:
            graph_metadata = build_graph_pipeline(
                paths=paths,
                max_items=args.max_items,
                metric=args.metric,
                index_type=args.faiss_index,
            )
            build_metadata["graph"] = graph_metadata

        if args.pipeline in {"vision", "both"}:
            resolved_device = resolve_torch_device(
                requested_device=args.device,
                require_torch=args.vision_backbone != "colorhist",
            )
            if args.vision_backbone != "colorhist":
                print(f"Vision backbone '{args.vision_backbone}' will run on device: {resolved_device}")

            vision_metadata = build_vision_pipeline(
                paths=paths,
                max_items=args.max_items,
                metric=args.metric,
                index_type=args.faiss_index,
                backbone=args.vision_backbone,
                weights_policy=args.vision_weights,
                batch_size=args.batch_size,
                device=resolved_device,
                image_size=args.image_size,
            )
            build_metadata["vision"] = vision_metadata

    # If the user only queries an existing index, load metadata for reporting.
    if not build_metadata:
        if args.pipeline in {"graph", "both"} and (paths.graph_artifact_dir / "metadata.json").exists():
            build_metadata["graph"] = load_json(paths.graph_artifact_dir / "metadata.json")
        if args.pipeline in {"vision", "both"} and (paths.vision_artifact_dir / "metadata.json").exists():
            build_metadata["vision"] = load_json(paths.vision_artifact_dir / "metadata.json")

    if args.query_id is not None:
        run_query(paths, args.pipeline, args.query_id, args.top_k)

    if args.random_queries > 0:
        ids = available_query_ids(paths, args.pipeline)
        if not ids:
            raise RuntimeError("No query IDs are available. Build the selected pipeline(s) first.")
        selected = random.sample(ids, min(args.random_queries, len(ids)))
        random_rows = []
        for query_id in selected:
            run_query(paths, args.pipeline, query_id, args.top_k)
            random_rows.append({"query_id": query_id})
        save_csv(random_rows, paths.outputs_root / "random_queries.csv")

    write_report(paths, args, build_metadata)

    print("Done.")
    print(f"Retrieval artifacts: {paths.retrieval_root}")
    print(f"Retrieval outputs:   {paths.outputs_root}")


if __name__ == "__main__":
    main()
