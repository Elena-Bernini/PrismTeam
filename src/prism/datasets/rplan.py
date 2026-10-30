"""RPLAN adapter (Graph2Plan pre-processed pickle -> unified layout JSON).

Input: ``Interface/static/Data/data_train_converted.pkl`` from the Graph2Plan
release. ``pickle.load`` returns ``{"data": [sample, ...], ...}`` where each
sample exposes (as attributes):

- ``name``       floor-plan id
- ``boundary``   (N, 4) exterior outline; rows 0-1 are the FRONT DOOR endpoints
- ``box``        (M, 5) room boxes ``[x0, y0, x1, y1, room_type_id]``
- ``edge``       (K, 3) ``[u, v, value]`` adjacency list from Graph2Plan
- ``rBoundary``  list of M room polygons
- ``order``      drawing order of rooms

This module keeps Pier's conversion (same field names) and adds: the corrected
room taxonomy, unified types, true polygon areas, polygon/box alignment checks
and the front door.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from src.prism.datasets.schema import SCHEMA_VERSION, Taxonomy
from src.prism.geometry.front_door import describe_front_door, door_segment_from_g2p_boundary
from src.prism.geometry.polygon import (as_points, bbox_iou, bbox_to_polygon, polygon_area,
                                    polygon_bbox)


def load_g2p_pickle(path: Path) -> Dict[str, Any]:
    """Load the Graph2Plan pickle, retrying with ``latin1`` for Python-2 era pickles."""
    if not path.exists():
        raise FileNotFoundError(
            f"RPLAN pickle not found: {path}\n"
            "Place data_train_converted.pkl under data/raw/rplan/Interface/static/Data/ "
            "(see README, section 'Where to put the data')."
        )
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def samples_from_pickle(obj: Any) -> List[Any]:
    """Return the list of samples from the loaded pickle (``obj['data']`` or the object itself)."""
    if isinstance(obj, dict) and "data" in obj:
        return list(obj["data"])
    if isinstance(obj, (list, tuple)):
        return list(obj)
    raise ValueError("Unrecognised Graph2Plan pickle structure (expected a dict with key 'data').")


def sample_id(sample: Any) -> str:
    """Floor-plan id of a sample as a string."""
    name = getattr(sample, "name", None)
    if isinstance(name, np.ndarray):
        name = name.item() if name.size == 1 else str(name)
    return str(name)


def grid_location(cx: float, cy: float, canvas: int = 256, grid_size: int = 5) -> List[int]:
    """Graph2Plan-style 5x5 grid cell of a centroid (image coordinates)."""
    gx = min(grid_size - 1, max(0, int(cx / canvas * grid_size)))
    gy = min(grid_size - 1, max(0, int(cy / canvas * grid_size)))
    return [gx, gy]


def align_polygons_to_boxes(boxes: np.ndarray, raw_polygons: Sequence[Any]) -> Tuple[List[List[List[float]]], str, float]:
    """Associate each room box with its polygon.

    Pier's code assumed ``rBoundary[i]`` belongs to ``box[i]``. We verify that
    with the IoU between the polygon bounding box and the room box. If the mean
    IoU is low, polygons are re-assigned greedily by IoU. Rooms that still have
    no plausible polygon fall back to their box rectangle.

    Returns ``(polygons, method, mean_iou)`` where ``method`` is ``index``,
    ``iou_matched`` or ``bbox_fallback``.
    """
    m = len(boxes)
    polys = [as_points(p) for p in raw_polygons] if raw_polygons is not None else []
    if not polys:
        return [bbox_to_polygon(b) for b in boxes], "bbox_fallback", 0.0

    def iou_of(i: int, j: int) -> float:
        """IoU between room box ``i`` and the bounding box of polygon ``j``."""
        bb = polygon_bbox(polys[j])
        return bbox_iou(boxes[i][:4], bb) if bb else 0.0

    direct = [iou_of(i, i) if i < len(polys) else 0.0 for i in range(m)]
    if direct and float(np.mean(direct)) >= 0.5:
        out = [polys[i] if (i < len(polys) and direct[i] >= 0.3) else bbox_to_polygon(boxes[i]) for i in range(m)]
        return out, "index", float(np.mean(direct))

    # Greedy IoU matching.
    scores = sorted(((iou_of(i, j), i, j) for i in range(m) for j in range(len(polys))), reverse=True)
    used_i, used_j, assign = set(), set(), {}
    for s, i, j in scores:
        if s < 0.3:
            break
        if i in used_i or j in used_j:
            continue
        assign[i] = (j, s)
        used_i.add(i)
        used_j.add(j)
    out, ious = [], []
    for i in range(m):
        if i in assign:
            out.append(polys[assign[i][0]])
            ious.append(assign[i][1])
        else:
            out.append(bbox_to_polygon(boxes[i]))
            ious.append(0.0)
    return out, "iou_matched", float(np.mean(ious)) if ious else 0.0


def parse_g2p_edges(raw: Any) -> np.ndarray:
    """Return the Graph2Plan edge list as a ``(K, 3)`` array ``[u, v, value]``.

    A missing third column is filled with ``-1``.
    """
    if raw is None:
        return np.zeros((0, 3))
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3))
    arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(1, -1)
    if arr.shape[1] < 3:
        arr = np.concatenate([arr[:, :2], -np.ones((len(arr), 1))], axis=1)
    return arr[:, :3]


def convert_sample(sample: Any, taxonomy: Taxonomy, cfg: Dict[str, Any], pkl_path: Optional[Path] = None) -> Dict[str, Any]:
    """Convert one Graph2Plan sample into the unified layout dictionary."""
    canvas = int(cfg["rplan"]["canvas"])
    canvas_area = float(canvas * canvas)
    sid = sample_id(sample)

    boundary_raw = np.asarray(getattr(sample, "boundary"), dtype=np.float64)
    boxes = np.asarray(getattr(sample, "box"), dtype=np.float64).reshape(-1, 5)
    edges = parse_g2p_edges(getattr(sample, "edge", None))
    raw_polys = getattr(sample, "rBoundary", None)

    polygons, poly_method, poly_iou = align_polygons_to_boxes(boxes, raw_polys)
    boundary = as_points(boundary_raw)

    rooms: List[Dict[str, Any]] = []
    for room_id, (box, poly) in enumerate(zip(boxes, polygons)):
        x_min, y_min, x_max, y_max, type_id = [int(round(v)) for v in box]
        w, h = max(0, x_max - x_min), max(0, y_max - y_min)
        cx, cy = x_min + w / 2.0, y_min + h / 2.0
        rooms.append({
            "id": room_id,                                   # local node index
            "room_type_id": type_id,                         # official Graph2Plan/RPLAN id
            "room_type_name": taxonomy.name(type_id),
            "unified_type": taxonomy.unified(type_id),
            "bbox": [x_min, y_min, x_max, y_max],
            "width": w,
            "height": h,
            "area_pixels": w * h,
            "area_ratio": (w * h) / canvas_area,             # bbox-based, as in Pier's version
            "polygon_area_ratio": polygon_area(poly) / canvas_area,
            "centroid": [cx, cy],                            # bbox centre, as in Graph2Plan
            "grid_location_5x5": grid_location(cx, cy, canvas),
            "polygon": poly,
            "is_entrance_room": False,
        })

    door_segment = door_segment_from_g2p_boundary(boundary_raw)
    front_door = describe_front_door(
        door_segment, boundary, [r["polygon"] for r in rooms], [r["room_type_name"] for r in rooms],
        max_dist_px=float(cfg["geometry"]["entrance_max_dist_px"]),
        min_width_px=float(cfg["front_door"]["min_width_px"]),
        source="g2p_boundary_rows_0_1",
    )
    if front_door and front_door["entrance_room_id"] is not None:
        rooms[front_door["entrance_room_id"]]["is_entrance_room"] = True

    door_flag_col = None
    if boundary_raw.ndim == 2 and boundary_raw.shape[1] >= 4:
        door_flag_col = [int(v) for v in boundary_raw[:, 3].tolist()]

    return {
        "schema_version": SCHEMA_VERSION,
        "id": sid,
        "dataset": "rplan",
        "coordinate_system": {"width": canvas, "height": canvas, "origin": "top_left",
                              "x_axis": "right", "y_axis": "down", "units": "px"},
        "boundary": boundary,
        "boundary_area_ratio": polygon_area(boundary) / canvas_area,
        "front_door": front_door,
        "rooms": rooms,
        "source_edges": [{"source": int(u), "target": int(v), "raw_value": int(r)} for u, v, r in edges],
        "source": {"dataset": "RPLAN (Graph2Plan data_train_converted.pkl)",
                   "pkl_file": str(pkl_path) if pkl_path else None},
        "raw": {
            "boundary_columns_2_3": boundary_raw[:, 2:4].astype(int).tolist() if boundary_raw.shape[1] >= 4 else None,
            "boundary_door_flag_column": door_flag_col,
            "order": np.asarray(getattr(sample, "order")).astype(int).tolist() if hasattr(sample, "order") else None,
            "polygon_alignment": {"method": poly_method, "mean_iou": poly_iou},
        },
    }


def iter_rplan_samples(pkl_path: Path, ids: Optional[Sequence[str]] = None) -> Iterator[Any]:
    """Yield samples from the pickle, optionally restricted to a set of ids."""
    samples = samples_from_pickle(load_g2p_pickle(pkl_path))
    if ids is None:
        yield from samples
        return
    wanted = set(map(str, ids))
    for s in samples:
        if sample_id(s) in wanted:
            yield s
