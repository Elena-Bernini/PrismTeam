"""ResPlan adapter (``ResPlan.pkl`` -> unified layout JSON).

Source: https://github.com/m-agour/ResPlan (``ResPlan.zip`` -> ``ResPlan.pkl``,
``split.json``) or the Kaggle release (same geometry plus ``plan["graph"]``).

``pickle.load`` returns a list of plan dictionaries. Geometry values are shapely
(Multi)Polygons, so **shapely must be installed** to unpickle the file. Keys
used here (phase 01 with ``--dataset resplan`` lists every key in the real file):

- room categories: ``living``, ``bedroom``, ``bathroom``, ``kitchen``,
  ``balcony`` (misspelled ``balacony`` in some files), ``storage``, ``stair`` ...
  (``configs/room_types.yaml -> resplan``)
- ``wall``, ``door``, ``window``, ``front_door``: building elements
- ``inner``: interior outline of the apartment (used as boundary)
- ``wall_depth``: wall thickness (same units as the coordinates)
- metadata: ``id``, ``unitType``, ``area``, ``net_area``; optional ``graph``

Checked against the 2026-07-28 release of ``ResPlan.pkl`` (17,000 plans, every
plan has a networkx ``graph`` whose nodes are ``<category>_<k>`` plus
``front_door_0``, and whose edges are typed ``via_door`` / ``adjacency`` /
``direct`` / ``via_window``; ``direct`` only links the front door to a room).

Alignment with RPLAN (so both datasets can share one gallery):

- coordinates are fitted into the same 256x256 image canvas as RPLAN: the
  bounding box of the interior outline plus every indoor room (balconies lie
  outside ``inner``) gets a longest side of ``resplan.fit_ratio`` x 256, centred,
  y flipped to image convention (y down). ``source_units_per_px`` keeps the scale;
- room types keep the ResPlan name as fine type and map to the unified taxonomy;
- every living polygon forms ONE living room (as in ResPlan's own
  ``plan_to_graph``, node ``living_0``); other categories give one room per
  polygon (``bedroom_0``, ``bedroom_1`` ...), stored in ``source_key``;
- walls, doors and windows are kept so phase 03 can label each adjacency as
  door / arch (open passage) / window / wall;
- the wall thickness in canvas pixels is stored in ``geometry_hints`` and used
  by phase 03 instead of the global dilation radius;
- ``front_door`` is collapsed into a segment exactly like the RPLAN door.

ResPlan's own graph (``plan["graph"]`` in the Kaggle release) is read by
:func:`read_resplan_graph` into ``source_edges``. With
``resplan.adjacency_source: source`` (default) phase 03 uses those edges and
their types (``via_door`` -> door, ``adjacency`` -> wall, ``via_window`` ->
window, ``via_opening`` -> arch); its front-door edge also decides the entrance
room. Plans without a graph fall back to our geometric adjacency.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from prism.datasets.schema import SCHEMA_VERSION, Taxonomy, make_id
from prism.geometry.front_door import describe_front_door, segment_from_door_polygon
from prism.geometry.polygon import point_in_polygon, room_shapes, segment_polygon_distance
from prism.geometry.polygon import polygon_area

# ResPlan graph edge types -> our connection types.
RESPLAN_EDGE_TO_CONNECTION = {
    "via_door": "door", "via_opening": "arch", "adjacency": "wall", "via_window": "window",
    "direct": "entrance", "fallback": "unknown",
}
KEY_ALIASES = {"balacony": "balcony"}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_resplan_pickle(path: Path) -> List[Dict[str, Any]]:
    """Load ``ResPlan.pkl`` into a list of plan dictionaries (keys normalised)."""
    if not path.exists():
        raise FileNotFoundError(
            f"ResPlan pickle not found: {path}\n"
            "Unzip ResPlan.zip into data/raw/resplan/ (see README, 'Where to put the data').")
    with open(path, "rb") as f:
        try:
            obj = pickle.load(f)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"{exc}. ResPlan stores shapely geometries: run `pip install shapely` "
                "(or `pip install -r requirements/base.txt`).") from exc
    if isinstance(obj, dict):
        obj = list(obj.values()) if all(isinstance(v, dict) for v in obj.values()) else [obj]
    return [normalize_keys(p) for p in obj]


def normalize_keys(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Rename misspelled keys (``balacony`` -> ``balcony``), as ResPlan's own utils do."""
    out = dict(plan)
    for wrong, right in KEY_ALIASES.items():
        if wrong in out and right not in out:
            out[right] = out.pop(wrong)
    return out


def plan_source_id(plan: Dict[str, Any], index: int) -> str:
    """Original id of a plan (``id`` / ``planID`` key, else its position in the list)."""
    for key in ("id", "planID", "plan_id"):
        if plan.get(key) is not None:
            return str(plan[key])
    return str(index)


def load_resplan_splits(path: Path) -> Optional[Dict[str, List[str]]]:
    """Official ``split.json`` as ``{"train": [...], "val": [...], "test": [...], "augmented": [...]}`` (or None)."""
    if not path.exists():
        return None
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: [str(v) for v in vals] for k, vals in data.items()}


# ----------------------------------------------------------------------------
# Geometry extraction (duck-typed on shapely, no shapely call needed)
# ----------------------------------------------------------------------------
def iter_polygons(geom: Any) -> List[Tuple[List[List[float]], List[List[List[float]]]]]:
    """Return ``[(exterior, holes), ...]`` for a (Multi)Polygon / GeometryCollection.

    Lines and points are ignored. Closing points are dropped.
    """
    if geom is None or getattr(geom, "is_empty", False):
        return []
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(iter_polygons(g))
        return out
    if hasattr(geom, "exterior"):
        ext = [[float(x), float(y)] for x, y, *_ in list(geom.exterior.coords)[:-1]]
        holes = [[[float(x), float(y)] for x, y, *_ in list(r.coords)[:-1]] for r in getattr(geom, "interiors", [])]
        return [(ext, holes)] if len(ext) >= 3 else []
    if isinstance(geom, (list, tuple)) and geom and isinstance(geom[0], (list, tuple)):
        pts = [[float(p[0]), float(p[1])] for p in geom]
        return [(pts, [])] if len(pts) >= 3 else []
    return []


def geometry_points(geom: Any) -> List[List[float]]:
    """Every coordinate of a geometry (polygons, lines or points) - used for bounds and doors."""
    if geom is None or getattr(geom, "is_empty", False):
        return []
    if hasattr(geom, "geoms"):
        pts: List[List[float]] = []
        for g in geom.geoms:
            pts.extend(geometry_points(g))
        return pts
    if hasattr(geom, "exterior"):
        return [[float(x), float(y)] for x, y, *_ in geom.exterior.coords]
    if hasattr(geom, "coords"):
        return [[float(x), float(y)] for x, y, *_ in geom.coords]
    return []


class CanvasTransform:
    """Map plan coordinates into the shared 256x256 image canvas (y flipped)."""

    def __init__(self, points: np.ndarray, canvas: int, fit_ratio: float) -> None:
        """Fit the bounding box of ``points`` into ``fit_ratio * canvas``, centred."""
        self.lo = points.min(axis=0)
        hi = points.max(axis=0)
        span = float(max(hi - self.lo))
        self.scale = canvas * fit_ratio / max(span, 1e-9)
        self.offset = (canvas - (hi - self.lo) * self.scale) / 2.0
        self.canvas = canvas

    def __call__(self, pts: Sequence[Sequence[float]]) -> List[List[float]]:
        """Transform a list of points."""
        out = []
        for x, y in pts:
            px = (x - self.lo[0]) * self.scale + self.offset[0]
            py = self.canvas - ((y - self.lo[1]) * self.scale + self.offset[1])
            out.append([round(float(px), 2), round(float(py), 2)])
        return out


# ----------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------
def room_categories(plan: Dict[str, Any], taxonomy: Taxonomy) -> List[str]:
    """Room categories present in a plan, in the order of ``room_types.yaml``."""
    return [c for c in taxonomy.resplan if plan.get(c) is not None and iter_polygons(plan.get(c))]


def unknown_keys(plan: Dict[str, Any], taxonomy: Taxonomy) -> List[str]:
    """Keys that are neither known room categories nor known non-room keys (shown by phase 01)."""
    known = set(taxonomy.resplan) | set(taxonomy.resplan_non_room) | set(KEY_ALIASES)
    return sorted(k for k in plan.keys() if k not in known)


def _room_record(rid: int, category: str, parts: List[List[List[float]]], taxonomy: Taxonomy,
                 type_index: Dict[str, int], canvas: int, area_px: Optional[float] = None) -> Dict[str, Any]:
    """Layout room from one or more transformed polygons (``area_px`` overrides the polygon area)."""
    parts = sorted(parts, key=polygon_area, reverse=True)
    allp = np.asarray([p for poly in parts for p in poly], dtype=np.float64)
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0)
    w, h = float(x1 - x0), float(y1 - y0)
    cx, cy = float(x0 + w / 2.0), float(y0 + h / 2.0)
    area = canvas * canvas
    poly_area = float(area_px) if area_px is not None else sum(polygon_area(p) for p in parts)
    return {
        "id": rid,
        "room_type_id": type_index[category],
        "room_type_name": category,
        "unified_type": taxonomy.resplan_unified(category),
        "bbox": [round(float(x0), 2), round(float(y0), 2), round(float(x1), 2), round(float(y1), 2)],
        "width": round(w, 2), "height": round(h, 2),
        "area_pixels": round(w * h, 2), "area_ratio": (w * h) / area,
        "polygon_area_ratio": poly_area / area,
        "centroid": [round(cx, 2), round(cy, 2)],
        "grid_location_5x5": [min(4, max(0, int(cx / canvas * 5))), min(4, max(0, int(cy / canvas * 5)))],
        "polygon": parts[0],
        "extra_polygons": parts[1:],
        "is_entrance_room": False,
    }


def room_groups(plan: Dict[str, Any], taxonomy: Taxonomy, cfg: Dict[str, Any]) -> List[Tuple[str, str, List[Any]]]:
    """Rooms of a plan as ``(source_key, category, [(exterior, holes), ...])``.

    Every living polygon forms ONE room (``living_0``, as in ResPlan's own graph);
    other categories give one room per polygon (``bedroom_0``, ``bedroom_1`` ...,
    numbered like the nodes of ``plan["graph"]``). Outdoor categories (garden,
    parking, pool) are skipped unless ``resplan.include_outdoor`` is true: RPLAN has
    no outdoor areas and they would sit outside the building outline.
    """
    rcfg = cfg["resplan"]
    outdoor = set(rcfg.get("outdoor_categories", [])) if not rcfg.get("include_outdoor", False) else set()
    groups: List[Tuple[str, str, List[Any]]] = []
    for cat in room_categories(plan, taxonomy):
        if cat in outdoor:
            continue
        polys = iter_polygons(plan[cat])
        if cat == "living":
            groups.append(("living_0", cat, polys))
        else:
            groups.extend((f"{cat}_{i}", cat, [p]) for i, p in enumerate(polys))
    return groups


def _wall_depth(plan: Dict[str, Any], groups: List[Tuple[str, str, List[Any]]]) -> float:
    """``wall_depth`` of the plan in source units (fallback: 2% of the plan's longest side)."""
    try:
        wd = float(plan.get("wall_depth"))
        if np.isfinite(wd) and wd > 0:
            return wd
    except (TypeError, ValueError):
        pass
    pts = np.asarray([p for _, _, polys in groups for ext, _ in polys for p in ext], dtype=np.float64)
    return 0.02 * float((pts.max(axis=0) - pts.min(axis=0)).max())


def convert_plan(plan: Dict[str, Any], index: int, taxonomy: Taxonomy, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one ResPlan plan dictionary into the unified layout dictionary.

    With ``resplan.room_partition`` (default) the wall-separated rooms are first
    turned into an RPLAN-style partition (see :mod:`prism.datasets.resplan_geometry`):
    room polygons meet on the wall centre lines and tile the building outline, so
    renders, areas and adjacency follow the RPLAN conventions. The original room
    polygons are kept in ``source_polygons`` for the door / arch / window labels.
    """
    rcfg = cfg["resplan"]
    canvas = int(rcfg["canvas"])
    sid = plan_source_id(plan, index)
    groups = room_groups(plan, taxonomy, cfg)
    if not groups:
        raise ValueError("plan without room geometry")
    wall_depth = _wall_depth(plan, groups)
    partition = bool(rcfg.get("room_partition", True))
    hints: Dict[str, Any] = {"room_partition": partition}
    boundary_holes: List[List[List[float]]] = []

    if partition:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        from prism.datasets.resplan_geometry import outline_holes, outline_ring, partition_rooms, polygon_rings

        geoms = [unary_union([Polygon(ext, holes).buffer(0) for ext, holes in polys]) for _, _, polys in groups]
        pcfg = rcfg.get("partition", {})
        cells, footprint, pstats = partition_rooms(
            geoms, wall_depth, growth=float(pcfg.get("growth", 0.55)), steps=int(pcfg.get("steps", 6)),
            closing=float(pcfg.get("closing", 0.75)), regularize=float(pcfg.get("regularize", 0.4)),
            snap=float(pcfg.get("snap", 1.5)), fill_holes=float(pcfg.get("fill_holes", 0.05)),
            simplify=float(pcfg.get("simplify", 0.25)))
        hints["partition"] = {k: round(float(v), 4) for k, v in pstats.items()}
        outline = outline_ring(footprint)
        bound_pts = [p for cell in cells for ring in polygon_rings(cell) for p in ring]
        tr = CanvasTransform(np.asarray(bound_pts, dtype=np.float64), canvas, float(rcfg["fit_ratio"]))
        boundary = tr(outline)
        boundary_holes = [tr(h) for h in outline_holes(footprint)]
        room_parts = [[tr(ring) for ring in polygon_rings(cell)] for cell in cells]
        room_areas = [float(cell.area) * tr.scale ** 2 for cell in cells]
    else:
        # Bounds: interior outline + every indoor room polygon (balconies lie outside ``inner``).
        inner = iter_polygons(plan.get("inner"))
        bound_pts = ([p for ext, _ in inner for p in ext]
                     + [p for _, _, polys in groups for ext, _ in polys for p in ext])
        tr = CanvasTransform(np.asarray(bound_pts, dtype=np.float64), canvas, float(rcfg["fit_ratio"]))
        room_parts = [[tr(ext) for ext, _ in polys] for _, _, polys in groups]
        room_areas = [None] * len(groups)
        boundary = tr(max(inner, key=lambda eh: polygon_area(eh[0]))[0]) if inner else []

    type_index = {name: i for i, name in enumerate(taxonomy.resplan)}
    rooms: List[Dict[str, Any]] = []
    for (key, cat, polys), parts, area_px in zip(groups, room_parts, room_areas):
        if not parts:
            continue
        room = _room_record(len(rooms), cat, parts, taxonomy, type_index, canvas, area_px)
        room["source_key"] = key
        if partition:
            room["source_polygons"] = [tr(ext) for ext, _ in polys]
        rooms.append(room)
    if not rooms:
        raise ValueError("plan without room geometry after partition")
    if len(boundary) < 3:
        x0 = min(r["bbox"][0] for r in rooms)
        y0 = min(r["bbox"][1] for r in rooms)
        x1 = max(r["bbox"][2] for r in rooms)
        y1 = max(r["bbox"][3] for r in rooms)
        boundary = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    walls = [{"exterior": tr(ext), "holes": [tr(h) for h in holes]} for ext, holes in iter_polygons(plan.get("wall"))]
    doors = [tr(ext) for ext, _ in iter_polygons(plan.get("door"))]
    windows = [tr(ext) for ext, _ in iter_polygons(plan.get("window"))]
    wall_px = max(float(rcfg["wall_px_min"]), wall_depth * tr.scale)
    hints["wall_thickness_px"] = wall_px

    # Front door: largest component, collapsed to a segment along its long axis.
    fd_polys = iter_polygons(plan.get("front_door"))
    if fd_polys:
        fd_pts = tr(max(fd_polys, key=lambda eh: polygon_area(eh[0]))[0])
    else:
        fd_pts = tr(geometry_points(plan.get("front_door"))) if geometry_points(plan.get("front_door")) else []
    segment = segment_from_door_polygon(fd_pts) if fd_pts else None
    front_door = describe_front_door(
        segment, boundary, [r["polygon"] for r in rooms], [r["room_type_name"] for r in rooms],
        max_dist_px=max(float(cfg["geometry"]["entrance_max_dist_px"]), 3 * wall_px),
        min_width_px=float(cfg["front_door"]["min_width_px"]),
        source="resplan_front_door_polygon",
    )
    # ResPlan's own graph: Kaggle plan["graph"], or the strict graph from resplan_utils on request.
    graph = plan.get("graph")
    graph_origin = "plan['graph']" if graph is not None else None
    if rcfg.get("graph_from_utils") and (graph is None or rcfg.get("graph_from_utils") == "always"):
        from prism.common.config import resolve_path
        graph = graph_with_resplan_utils(plan, resolve_path(cfg["paths"]["data_root"]) / str(rcfg["utils_dir"]))
        graph_origin = "resplan_utils.plan_to_graph"
    source_edges, door_rooms, graph_info = read_resplan_graph(graph, rooms, tr, bool(rcfg.get("keep_fallback_edges")))
    graph_info["origin"] = graph_origin

    # Entrance room: ResPlan's front-door ("direct") edge when available, else the closest room.
    if front_door is not None:
        front_door["entrance_source"] = "closest_room"
        if door_rooms:
            closest = front_door["entrance_room_id"]
            pick = closest if closest in door_rooms else min(
                door_rooms, key=lambda rid: segment_polygon_distance(segment[0], segment[1], rooms[rid]["polygon"]))
            front_door["entrance_room_id"] = int(pick)
            front_door["entrance_source"] = "resplan_graph"
    if front_door and front_door["entrance_room_id"] is not None:
        rooms[front_door["entrance_room_id"]]["is_entrance_room"] = True

    meta = {k: plan.get(k) for k in ("unitType", "area", "net_area", "wall_depth") if k in plan}
    return {
        "schema_version": SCHEMA_VERSION,
        "id": make_id("resplan", sid),
        "source_id": sid,
        "dataset": "resplan",
        "coordinate_system": {"width": canvas, "height": canvas, "origin": "top_left", "x_axis": "right",
                              "y_axis": "down", "units": "px", "source_units_per_px": 1.0 / tr.scale},
        "boundary": boundary,
        "boundary_area_ratio": (polygon_area(boundary) - sum(polygon_area(h) for h in boundary_holes)) / canvas ** 2,
        "boundary_holes": boundary_holes,             # light wells / courtyards (drawn as outside)
        "front_door": front_door,
        "rooms": rooms,
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "geometry_hints": hints,
        "source_edges": source_edges,
        "source_graph_info": graph_info,
        "source": {"dataset": "ResPlan", "index_in_pickle": index},
        "raw": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in meta.items()},
    }


EDGE_TYPE_KEYS = ("type", "edge_type", "kind", "label", "connection")
# When ResPlan's graph lists the same room pair twice, keep the most specific type.
EDGE_TYPE_PRIORITY = {"via_door": 0, "via_opening": 1, "via_window": 2, "adjacency": 3, "fallback": 4}


def edge_type_of(data: Any) -> str:
    """Edge type stored in a ResPlan graph edge (the attribute name is checked among common keys)."""
    for key in EDGE_TYPE_KEYS:
        value = (data or {}).get(key) if isinstance(data, dict) else None
        if isinstance(value, str) and value:
            return value
    return "unknown"


def is_front_door_node(node: Any, data: Any) -> bool:
    """True for ResPlan's front-door nodes (``front_door_<i>`` or ``type == "front_door"``)."""
    return str(node).startswith("front_door") or (isinstance(data, dict) and data.get("type") == "front_door")


def node_to_room(node: Any, data: Any, key_to_room: Dict[str, int], rooms: List[Dict[str, Any]],
                 tr: "CanvasTransform") -> Tuple[Optional[int], str]:
    """Match a ResPlan graph node to one of our rooms.

    First by name (``living_0``, ``bedroom_1`` ... = our ``source_key``), then by
    geometry: the centre of the node's ``geometry`` attribute, mapped into the
    canvas, must fall inside one of our room polygons. Returns
    ``(room_id or None, "name" | "geometry" | "unmatched")``.
    """
    name = str(node)
    if name in key_to_room:
        return key_to_room[name], "name"
    pts = geometry_points(data.get("geometry")) if isinstance(data, dict) else []
    if pts:
        cx, cy = np.asarray(pts, dtype=np.float64).mean(axis=0)
        (px, py), = tr([[cx, cy]])
        for room in rooms:
            if any(point_in_polygon((px, py), poly) for poly in room_shapes(room)):
                return int(room["id"]), "geometry"
    return None, "unmatched"


def read_resplan_graph(graph: Any, rooms: List[Dict[str, Any]], tr: "CanvasTransform",
                       keep_fallback: bool = False) -> Tuple[List[Dict[str, Any]], List[int], Dict[str, Any]]:
    """Read ResPlan's own graph (Kaggle ``plan["graph"]`` or ``resplan_utils.plan_to_graph``).

    Returns ``(source_edges, front_door_rooms, info)``:

    - ``source_edges``: room-room edges mapped to our room ids, one per pair, with
      ``source_type`` (ResPlan's label) and ``connection`` (our label, see
      ``RESPLAN_EDGE_TO_CONNECTION``). ``fallback`` edges (links ResPlan adds to
      connect otherwise disconnected parts, without a shared wall) are dropped
      unless ``keep_fallback``;
    - ``front_door_rooms``: rooms linked to a front-door node (ResPlan ``direct`` edges);
    - ``info``: counters for the reports (how nodes were matched, what was dropped).
    """
    info = {"has_graph": False, "nodes": 0, "matched_by_name": 0, "matched_by_geometry": 0, "unmatched_nodes": 0,
            "edges": 0, "room_edges_kept": 0, "fallback_dropped": 0, "unmatched_edges": 0, "front_door_edges": 0}
    if graph is None or not hasattr(graph, "edges") or not hasattr(graph, "nodes"):
        return [], [], info
    info["has_graph"] = True
    key_to_room = {r["source_key"]: int(r["id"]) for r in rooms}
    node_room: Dict[Any, Optional[int]] = {}
    door_nodes = set()
    for node, data in graph.nodes(data=True):
        info["nodes"] += 1
        if is_front_door_node(node, data):
            door_nodes.add(node)
            continue
        rid, how = node_to_room(node, data, key_to_room, rooms, tr)
        node_room[node] = rid
        info["matched_by_name" if how == "name" else "matched_by_geometry" if how == "geometry" else "unmatched_nodes"] += 1

    best: Dict[Tuple[int, int], Dict[str, Any]] = {}
    front_door_rooms: List[int] = []
    for u, v, data in graph.edges(data=True):
        info["edges"] += 1
        kind = edge_type_of(data)
        if u in door_nodes or v in door_nodes:
            other = v if u in door_nodes else u
            if node_room.get(other) is not None:
                front_door_rooms.append(int(node_room[other]))
                info["front_door_edges"] += 1
            continue
        ru, rv = node_room.get(u), node_room.get(v)
        if ru is None or rv is None or ru == rv:
            info["unmatched_edges"] += 1
            continue
        if kind == "fallback" and not keep_fallback:
            info["fallback_dropped"] += 1
            continue
        key = (min(ru, rv), max(ru, rv))
        prev = best.get(key)
        if prev is None or EDGE_TYPE_PRIORITY.get(kind, 9) < EDGE_TYPE_PRIORITY.get(prev["source_type"], 9):
            best[key] = {"source": key[0], "target": key[1], "raw_value": -1, "source_type": kind,
                         "connection": RESPLAN_EDGE_TO_CONNECTION.get(kind, "unknown")}
    edges = [best[k] for k in sorted(best)]
    info["room_edges_kept"] = len(edges)
    return edges, sorted(set(front_door_rooms)), info


def graph_with_resplan_utils(plan: Dict[str, Any], utils_dir: Path) -> Any:
    """Build ResPlan's STRICT graph with the authors' ``resplan_utils.plan_to_graph``.

    Used only when ``resplan.graph_from_utils`` is true and the plan has no
    ``graph`` (GitHub release), or to get ``via_opening`` (open passages), which
    the Kaggle benchmark graph merges into ``adjacency``. ``resplan_utils.py``
    must be in ``utils_dir`` (it ships with the dataset repository).
    """
    import importlib
    import sys

    if not (utils_dir / "resplan_utils.py").exists():
        raise FileNotFoundError(f"resplan_utils.py not found in {utils_dir} (copy it from the ResPlan repository).")
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    return importlib.import_module("resplan_utils").plan_to_graph(plan)


def iter_selected_plans(plans: List[Dict[str, Any]], splits: Optional[Dict[str, List[str]]],
                        include_augmented: bool) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield ``(index, plan)`` skipping augmented copies unless requested."""
    augmented = set((splits or {}).get("augmented", [])) if not include_augmented else set()
    for i, plan in enumerate(plans):
        if augmented and plan_source_id(plan, i) in augmented:
            continue
        yield i, plan
