"""ResPlan adapter (EXPERIMENTAL - written from the paper, not yet run on the real files).

ResPlan (Sec. 4 of the paper) stores each plan as a dictionary whose keys are
semantic categories (``bedroom``, ``living``, ``kitchen``, ``bathroom``,
``balcony``, ``storage``, ``wall``, ``door``, ``window``, ``front_door``,
``inner``, ``land`` ...) and whose values are shapely (Multi)Polygons in
metres, plus metadata (``wall_depth``, ``unitType``, ``area``, ``id``) and a
networkx room graph whose edges are typed ``door`` / ``arch`` / ``wall``.

This adapter maps a record onto the SAME layout schema used for RPLAN:

- coordinates are scaled into a 256x256 canvas (aspect ratio preserved,
  4% margin); the scale factor is stored in ``coordinate_system``;
- each polygon of a room category becomes one room, in the stored order;
- ``inner`` (or the union of rooms' bounding box) becomes the boundary;
- ``front_door`` becomes the front-door segment (long axis of the polygon);
- the native graph edges are kept in ``source_edges`` with their connection type.

When the real files arrive: put them in ``data/raw/resplan/``, adapt
``load_resplan_records`` to the actual file format, and run phase 01/02 with
``--dataset resplan``. Everything downstream works unchanged.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np

from src.prism.datasets.schema import SCHEMA_VERSION, Taxonomy
from src.prism.geometry.front_door import describe_front_door, segment_from_door_polygon
from src.prism.geometry.polygon import polygon_area, polygon_bbox

CANVAS = 256


def geometry_to_polygons(geom: Any) -> List[List[List[float]]]:
    """Extract exterior rings from a shapely-like (Multi)Polygon without importing shapely."""
    if geom is None:
        return []
    if getattr(geom, "is_empty", False):
        return []
    if hasattr(geom, "geoms"):
        out: List[List[List[float]]] = []
        for g in geom.geoms:
            out.extend(geometry_to_polygons(g))
        return out
    if hasattr(geom, "exterior"):
        return [[[float(x), float(y)] for x, y in list(geom.exterior.coords)[:-1]]]
    if isinstance(geom, (list, tuple)) and geom and isinstance(geom[0], (list, tuple)):
        return [[[float(p[0]), float(p[1])] for p in geom]]
    return []


def load_resplan_records(root: Path) -> Iterator[Dict[str, Any]]:
    """Yield raw ResPlan records from ``root`` (``*.pkl`` holding a list or dict of records)."""
    files = sorted(root.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No ResPlan .pkl files found in {root}")
    for path in files:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        records = obj.values() if isinstance(obj, dict) else obj
        for rec in records:
            yield rec


def convert_record(rec: Dict[str, Any], taxonomy: Taxonomy, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one ResPlan record into the unified layout dictionary."""
    room_keys = [taxonomy.fine[i]["name"] for i in taxonomy.fine_ids]
    raw_rooms = []
    for key in room_keys:
        for poly in geometry_to_polygons(rec.get(key)):
            raw_rooms.append((key, poly))
    inner = geometry_to_polygons(rec.get("inner"))
    all_pts = np.asarray([p for _, poly in raw_rooms for p in poly] + [p for poly in inner for p in poly], dtype=np.float64)
    if len(all_pts) == 0:
        raise ValueError("ResPlan record without room geometry")
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    span = float(max(hi - lo))
    scale = CANVAS * 0.92 / max(span, 1e-9)
    offset = (CANVAS - (hi - lo) * scale) / 2.0

    def tr(poly: Sequence[Sequence[float]]) -> List[List[float]]:
        """Map metric coordinates into the 256 canvas (y flipped to image convention)."""
        out = []
        for x, y in poly:
            px = (x - lo[0]) * scale + offset[0]
            py = CANVAS - ((y - lo[1]) * scale + offset[1])
            out.append([float(px), float(py)])
        return out

    rooms = []
    for rid, (key, poly) in enumerate(raw_rooms):
        p = tr(poly)
        x0, y0, x1, y1 = polygon_bbox(p)
        type_id = taxonomy.name_to_id[key]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rooms.append({
            "id": rid, "room_type_id": type_id, "room_type_name": key, "unified_type": taxonomy.unified(type_id),
            "bbox": [x0, y0, x1, y1], "width": x1 - x0, "height": y1 - y0,
            "area_pixels": (x1 - x0) * (y1 - y0), "area_ratio": (x1 - x0) * (y1 - y0) / CANVAS ** 2,
            "polygon_area_ratio": polygon_area(p) / CANVAS ** 2, "centroid": [cx, cy],
            "grid_location_5x5": [min(4, int(cx / CANVAS * 5)), min(4, int(cy / CANVAS * 5))],
            "polygon": p, "is_entrance_room": False,
        })
    boundary = tr(inner[0]) if inner else [[r["bbox"][0], r["bbox"][1]] for r in rooms[:1]]
    fd_polys = geometry_to_polygons(rec.get("front_door"))
    segment = segment_from_door_polygon(tr(fd_polys[0])) if fd_polys else None
    front_door = describe_front_door(segment, boundary, [r["polygon"] for r in rooms],
                                     [r["unified_type"] for r in rooms],
                                     max_dist_px=float(cfg["geometry"]["entrance_max_dist_px"]),
                                     source="resplan_front_door_polygon")
    if front_door and front_door["entrance_room_id"] is not None:
        rooms[front_door["entrance_room_id"]]["is_entrance_room"] = True

    source_edges = []
    graph = rec.get("graph")
    if graph is not None and hasattr(graph, "edges"):
        for u, v, data in graph.edges(data=True):
            source_edges.append({"source": int(u), "target": int(v), "raw_value": -1,
                                 "connection": str(data.get("type", data.get("connection", "unknown")))})
    return {
        "schema_version": SCHEMA_VERSION, "id": str(rec.get("id", rec.get("planID", "unknown"))), "dataset": "resplan",
        "coordinate_system": {"width": CANVAS, "height": CANVAS, "origin": "top_left", "x_axis": "right",
                              "y_axis": "down", "units": "px", "metres_per_px": 1.0 / scale},
        "boundary": boundary, "boundary_area_ratio": polygon_area(boundary) / CANVAS ** 2,
        "front_door": front_door, "rooms": rooms, "source_edges": source_edges,
        "source": {"dataset": "ResPlan"},
        "raw": {"unitType": rec.get("unitType"), "area_m2": rec.get("area"), "wall_depth": rec.get("wall_depth")},
    }
