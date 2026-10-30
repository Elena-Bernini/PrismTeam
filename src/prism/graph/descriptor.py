"""Interpretable hand-crafted graph descriptor (Pier's baseline, extended).

The vector concatenates semantic, topological, geometric, boundary and
front-door statistics. It needs no training and is the transparent baseline
against which the GNN embeddings are compared. It also works on partial graphs
(see :func:`prism.graph.build.subgraph`).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.prism.datasets.schema import CONNECTIONS, RELATIONS, Taxonomy
from src.prism.geometry.polygon import polygon_area
from src.prism.graph.build import room_edges, room_nodes

FACINGS = ["north", "south", "east", "west", "unknown"]


def boundary_features(layout: Optional[Dict[str, Any]], canvas: float = 256.0) -> Tuple[List[float], List[str]]:
    """Width, height, aspect ratio, area ratio and vertex count of the exterior outline."""
    names = ["boundary_width", "boundary_height", "boundary_aspect", "boundary_area_ratio", "boundary_num_vertices"]
    pts = (layout or {}).get("boundary") or []
    if len(pts) < 3:
        return [0.0, 0.0, 0.0, 0.0, float(len(pts))], names
    p = np.asarray(pts, dtype=np.float64)
    w = float(p[:, 0].max() - p[:, 0].min())
    h = float(p[:, 1].max() - p[:, 1].min())
    return [w / canvas, h / canvas, w / max(h, 1e-8), polygon_area(pts) / canvas ** 2, float(len(pts))], names


def door_features(graph: Dict[str, Any], taxonomy: Taxonomy, canvas: float = 256.0) -> Tuple[List[float], List[str]]:
    """Front-door block: presence, position, width, facing, entrance room type, mean hop distance."""
    fd = graph.get("front_door")
    names = ["door_present", "door_cx", "door_cy", "door_width"]
    names += [f"door_facing_{f}" for f in FACINGS]
    names += [f"entrance_type_{taxonomy.name(t)}" for t in taxonomy.fine_ids]
    names += ["door_hops_mean", "door_hops_max", "door_reachable_ratio"]
    if not fd:
        return [0.0] * len(names), names
    vec = [1.0, fd["center"][0] / canvas, fd["center"][1] / canvas, float(fd.get("width_px") or 0) / canvas]
    vec += [1.0 if fd.get("facing", "unknown") == f else 0.0 for f in FACINGS]
    entrance_type = None
    if fd.get("entrance_room_id") is not None:
        for nd in room_nodes(graph):
            if int(nd["id"]) == int(fd["entrance_room_id"]):
                entrance_type = int(nd["room_type_id"])
    vec += [1.0 if entrance_type == t else 0.0 for t in taxonomy.fine_ids]
    hops = np.asarray([nd["door_hops"] for nd in room_nodes(graph)], dtype=np.float32)
    reach = hops[hops >= 0]
    vec += [float(reach.mean()) if len(reach) else 0.0, float(reach.max()) if len(reach) else 0.0,
            float(len(reach) / max(1, len(hops)))]
    return vec, names


def graph_descriptor(graph: Dict[str, Any], layout: Optional[Dict[str, Any]], taxonomy: Taxonomy) -> Tuple[np.ndarray, List[str]]:
    """Return ``(vector, feature_names)`` for one (possibly partial) graph."""
    canvas = float((layout or {}).get("coordinate_system", {}).get("width", 256))
    nodes = room_nodes(graph)
    edges = room_edges(graph)
    n = len(nodes)
    undirected = max(1, len(edges) // 2)
    feats: List[float] = []
    names: List[str] = []

    feats += [float(n), float(len(edges) // 2), (len(edges) / 2.0) / max(1.0, n * (n - 1) / 2.0)]
    names += ["num_rooms", "num_edges", "edge_density"]

    type_counts = Counter(int(nd["room_type_id"]) for nd in nodes)
    for t in taxonomy.fine_ids:
        feats.append(float(type_counts.get(t, 0)))
        names.append(f"type_count_{taxonomy.name(t)}")
    for t in taxonomy.fine_ids:
        feats.append(type_counts.get(t, 0) / max(1, n))
        names.append(f"type_ratio_{taxonomy.name(t)}")

    rel_counts = Counter(e["relation"] for e in edges)
    for r in RELATIONS:
        feats.append(float(rel_counts.get(r, 0)) / 2.0)
        names.append(f"relation_count_{r}")
    for r in RELATIONS:
        feats.append(rel_counts.get(r, 0) / max(1, len(edges)))
        names.append(f"relation_ratio_{r}")

    conn_counts = Counter(e.get("connection", "wall") for e in edges)
    for c in CONNECTIONS:
        if c == "entrance":
            continue
        feats.append(conn_counts.get(c, 0) / max(1, len(edges)))
        names.append(f"connection_ratio_{c}")

    areas = np.asarray([float(nd.get("polygon_area_ratio", nd["area_ratio"])) for nd in nodes], dtype=np.float32)
    feats += [float(areas.mean()), float(areas.std()), float(areas.min()), float(areas.max())] if n else [0.0] * 4
    names += ["area_mean", "area_std", "area_min", "area_max"]

    by_type: Dict[int, List[float]] = defaultdict(list)
    for nd in nodes:
        by_type[int(nd["room_type_id"])].append(float(nd.get("polygon_area_ratio", nd["area_ratio"])))
    for t in taxonomy.fine_ids:
        vals = by_type.get(t, [])
        feats += [float(np.sum(vals)) if vals else 0.0, float(np.mean(vals)) if vals else 0.0]
        names += [f"area_sum_{taxonomy.name(t)}", f"area_mean_{taxonomy.name(t)}"]

    cents = np.asarray([nd["centroid"] for nd in nodes], dtype=np.float32) / canvas if n else np.zeros((0, 2))
    feats += ([float(cents[:, 0].mean()), float(cents[:, 1].mean()), float(cents[:, 0].std()), float(cents[:, 1].std())]
              if n else [0.0] * 4)
    names += ["centroid_x_mean", "centroid_y_mean", "centroid_x_std", "centroid_y_std"]

    ws, hs, ars = [], [], []
    for nd in nodes:
        x0, y0, x1, y1 = [float(v) for v in nd["bbox"][:4]]
        w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
        ws.append(w / canvas)
        hs.append(h / canvas)
        ars.append(w / max(h, 1e-8))
    for vals, prefix in ((ws, "bbox_width"), (hs, "bbox_height"), (ars, "bbox_aspect")):
        a = np.asarray(vals, dtype=np.float32)
        feats += [float(a.mean()), float(a.std()), float(a.max())] if len(a) else [0.0] * 3
        names += [f"{prefix}_mean", f"{prefix}_std", f"{prefix}_max"]

    deg = Counter(int(e["source"]) for e in edges)
    d = np.asarray([deg.get(int(nd["id"]), 0) for nd in nodes], dtype=np.float32)
    feats += [float(d.mean()), float(d.std()), float(d.max())] if n else [0.0] * 3
    names += ["degree_mean", "degree_std", "degree_max"]

    # Typed adjacency histogram over unified types (compact, order-free).
    uni = taxonomy.vocab("unified")
    type_of = {int(nd["id"]): nd["unified_type"] for nd in nodes}
    pair_counts = Counter()
    for e in edges:
        a, b = type_of.get(int(e["source"])), type_of.get(int(e["target"]))
        if a and b and int(e["source"]) < int(e["target"]):
            pair_counts[tuple(sorted((a, b)))] += 1
    for i, a in enumerate(uni):
        for b in uni[i:]:
            feats.append(pair_counts.get((a, b), 0) / undirected)
            names.append(f"adj_{a}__{b}")

    bf, bn = boundary_features(layout, canvas)
    df, dn = door_features(graph, taxonomy, canvas)
    feats += bf + df
    names += bn + dn
    return np.asarray(feats, dtype=np.float32), names
