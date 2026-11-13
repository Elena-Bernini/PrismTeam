"""Build the graph JSON of a floor plan from its layout JSON.

Graph JSON (``graphs_json/<id>.json``)::

    {
      "id": "rplan_75342", "dataset": "rplan",
      "nodes": [{
          "id": 0, "kind": "room" | "front_door",
          "room_type_id": 7, "room_type_name": "SecondRoom", "unified_type": "bedroom",
          "area_ratio": 0.04, "polygon_area_ratio": 0.039,
          "bbox": [...], "centroid": [...], "grid_location_5x5": [gx, gy],
          "is_entrance_room": false,
          "door_hops": 2,                    # graph distance from the entrance room (-1 = unreachable/none)
          "door_distance": 0.31              # centroid -> door centre, divided by the canvas size
      }, ...],
      "edges": [{                            # every adjacency appears twice (u->v and v->u)
          "source": 0, "target": 1,
          "relation": "left_of", "relation_id": 0,     # source relative to target
          "connection": "wall",                        # wall | door | arch | window | entrance | unknown
          "geometric_connection": "wall",              # our own label (equal to connection unless the
                                                       # dataset's label is used: adjacency_source=source)
          "shared_wall_px": 38.0,
          "dx": -0.2, "dy": 0.01, "overlap_x": 0.0, "overlap_y": 0.93,
          "in_geometric": true, "in_source": true
      }, ...],
      "front_door": {"facing": "south", "orientation": "horizontal", "center": [...],
                     "width_px": 12.0, "entrance_room_id": 3, "node_id": 8},
      "adjacency": {"source": "geometric", "requested": "geometric", "wall_dilation_px": 3, "min_contact_px": 5},
      "comparison_with_source": {"geometric_pairs": 9, "source_pairs": 8, "common_pairs": 7}
    }
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from prism.datasets.schema import FRONT_DOOR_NODE_TYPE, RELATION_TO_ID
from prism.geometry.adjacency import RoomMasks, classify_connections, geometric_adjacency
from prism.geometry.polygon import bbox_to_polygon, room_shapes
from prism.geometry.relations import inverse_relation, relation_features, spatial_relation


def _room_polygon(room: Dict[str, Any]) -> List[List[float]]:
    """Main polygon of a room, falling back to its bounding box rectangle."""
    poly = room.get("polygon") or []
    return poly if len(poly) >= 3 else bbox_to_polygon(room["bbox"])


def _make_edge(u: int, v: int, box_u, box_v, rel: str, shared: float, connection: str,
               in_geo: bool, in_src: bool, canvas: float) -> Dict[str, Any]:
    """Assemble one directed edge record with its geometric attributes."""
    f = relation_features(box_u, box_v, canvas)
    return {
        "source": int(u), "target": int(v),
        "relation": rel, "relation_id": RELATION_TO_ID[rel],
        "connection": connection,
        "shared_wall_px": float(shared),
        "dx": f["dx"], "dy": f["dy"],
        "overlap_x": min(1.0, f["overlap_x"]), "overlap_y": min(1.0, f["overlap_y"]),
        "in_geometric": bool(in_geo), "in_source": bool(in_src),
    }


def source_pairs(layout: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """Undirected room pairs listed in the edges shipped with the dataset.

    RPLAN: Graph2Plan's ``sample.edge``; ResPlan: ``plan["graph"]`` (Kaggle release).
    """
    n = len(layout["rooms"])
    out: Set[Tuple[int, int]] = set()
    for e in layout.get("source_edges", []):
        u, v = int(e["source"]), int(e["target"])
        if 0 <= u < n and 0 <= v < n and u != v:
            out.add((min(u, v), max(u, v)))
    return out


# Backwards-compatible name used by older scripts.
g2p_pairs = source_pairs


def wall_radius(layout: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    """Dilation radius that bridges the plan's real walls (ResPlan wall thickness), else the config value."""
    hint = (layout.get("geometry_hints") or {}).get("wall_thickness_px")
    if hint:
        return max(1, int(np.ceil(float(hint) / 2.0)) + 1)
    return int(cfg["geometry"]["wall_dilation_px"])


def dilation_radius(layout: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    """Dilation radius for adjacency.

    RPLAN and partitioned ResPlan layouts (rooms meeting on wall centre lines) use
    ``geometry.wall_dilation_px``, so both datasets follow the same rule; ResPlan
    layouts with wall-separated rooms (``resplan.room_partition: false``) use their
    own wall thickness.
    """
    hints = layout.get("geometry_hints") or {}
    if hints.get("room_partition"):
        return int(cfg["geometry"]["wall_dilation_px"])
    return wall_radius(layout, cfg)


def connection_masks(layout: Dict[str, Any], cfg: Dict[str, Any], masks: RoomMasks) -> RoomMasks:
    """Room masks used to label door / arch / window / wall.

    The labels look at the strip between two rooms where their shared wall is, so
    they need the ORIGINAL wall-separated room polygons (``source_polygons`` of
    partitioned ResPlan layouts) and a radius that bridges the real wall.
    """
    rooms = layout["rooms"]
    if not any(r.get("source_polygons") for r in rooms):
        return masks
    shapes = [r.get("source_polygons") or room_shapes(r) for r in rooms]
    return RoomMasks(shapes, masks.canvas, wall_radius(layout, cfg))


def adjacency_source_for(layout: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[str, str]:
    """Return ``(requested, used)`` adjacency source for one layout.

    ``requested`` comes from the dataset section (``rplan.adjacency_source``,
    ``resplan.adjacency_source``) or, if absent, ``geometry.adjacency_source``.
    ``used`` falls back to ``geometric`` when ``source`` is requested but this
    plan has no source edges (e.g. a ResPlan plan without ``graph``).
    """
    section = cfg.get(str(layout.get("dataset", "")), {}) or {}
    requested = str(section.get("adjacency_source") or cfg["geometry"]["adjacency_source"])
    if requested == "g2p":
        requested = "source"
    used = requested
    if requested == "source" and not layout.get("source_edges"):
        used = "geometric"
    return requested, used


def bfs_hops(n: int, undirected_pairs: Iterable[Tuple[int, int]], start: Optional[int]) -> List[int]:
    """Breadth-first hop distance from ``start`` to every node (-1 if unreachable or no start)."""
    hops = [-1] * n
    if start is None or not (0 <= start < n):
        return hops
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a, b in undirected_pairs:
        if a < n and b < n:
            adj[a].append(b)
            adj[b].append(a)
    hops[start] = 0
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in adj[cur]:
            if hops[nxt] < 0:
                hops[nxt] = hops[cur] + 1
                queue.append(nxt)
    return hops


def build_graph(layout: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute nodes, adjacency, connection types, spatial relations and the front-door node."""
    geo_cfg = cfg["geometry"]
    canvas = float(layout["coordinate_system"]["width"])
    rooms = layout["rooms"]
    n = len(rooms)
    boxes = [r["bbox"] for r in rooms]
    radius = dilation_radius(layout, cfg)
    masks = RoomMasks([room_shapes(r) for r in rooms], int(canvas), radius)

    geo = geometric_adjacency(masks, boxes, min_contact_px=int(geo_cfg["min_contact_px"]))
    src = source_pairs(layout)
    requested, source = adjacency_source_for(layout, cfg)
    if source == "geometric":
        pairs = dict(geo)
    elif source == "source":
        pairs = {p: geo.get(p, 0.0) for p in src}
    elif source == "union":
        pairs = dict(geo)
        for p in src:
            pairs.setdefault(p, 0.0)
    else:
        raise ValueError(f"Unknown adjacency_source '{source}'")

    # Connection types. Our own classification from wall/door/window geometry is
    # always computed (ResPlan: on the original wall-separated rooms, see
    # connection_masks; RPLAN gives "wall") and kept on every edge as
    # "geometric_connection". With adjacency_source=source the dataset's own
    # label (ResPlan graph) is the edge's "connection".
    hint = (layout.get("geometry_hints") or {}).get("wall_thickness_px", 2 * radius)
    rp = cfg.get("resplan", {})
    geo_conn = classify_connections(list(pairs), connection_masks(layout, cfg, masks), layout.get("walls"),
                                    layout.get("doors"), layout.get("windows"), float(hint),
                                    float(rp.get("open_passage_min_px", 6)), float(rp.get("door_min_px", 4)))
    conn = dict(geo_conn)
    if source in {"source", "union"}:
        for e in layout.get("source_edges", []):
            key = (min(int(e["source"]), int(e["target"])), max(int(e["source"]), int(e["target"])))
            if e.get("connection") and key in conn and (source == "source" or key not in geo):
                conn[key] = str(e["connection"])

    ct, ot = float(geo_cfg["containment_threshold"]), float(geo_cfg["axis_overlap_threshold"])
    edges: List[Dict[str, Any]] = []
    for (i, j), shared in sorted(pairs.items()):
        rel = spatial_relation(boxes[i], boxes[j], ct, ot)
        in_geo, in_s = (i, j) in geo, (i, j) in src
        c, gc = conn.get((i, j), "wall"), geo_conn.get((i, j), "wall")
        for e in (_make_edge(i, j, boxes[i], boxes[j], rel, shared, c, in_geo, in_s, canvas),
                  _make_edge(j, i, boxes[j], boxes[i], inverse_relation(rel), shared, c, in_geo, in_s, canvas)):
            e["geometric_connection"] = gc      # our label, kept for comparison when the source label is used
            edges.append(e)

    door = layout.get("front_door")
    entrance = door.get("entrance_room_id") if door else None
    hops = bfs_hops(n, pairs.keys(), entrance)

    nodes: List[Dict[str, Any]] = []
    for r in rooms:
        cx, cy = r["centroid"]
        dd = (float(np.hypot(cx - door["center"][0], cy - door["center"][1])) / canvas) if door else -1.0
        nodes.append({
            "id": int(r["id"]), "kind": "room",
            "room_type_id": int(r["room_type_id"]), "room_type_name": r["room_type_name"],
            "unified_type": r["unified_type"],
            "area_ratio": float(r["area_ratio"]),
            "polygon_area_ratio": float(r.get("polygon_area_ratio", r["area_ratio"])),
            "bbox": r["bbox"], "centroid": r["centroid"],
            "grid_location_5x5": r["grid_location_5x5"],
            "is_entrance_room": bool(r.get("is_entrance_room", False)),
            "door_hops": int(hops[int(r["id"])]),
            "door_distance": dd,
        })

    door_block = None
    if door:
        door_block = {k: door.get(k) for k in ("facing", "orientation", "center", "width_px", "entrance_room_id")}
        door_block["node_id"] = None
        if cfg["front_door"]["add_virtual_node"]:
            did = n
            (x1, y1), (x2, y2) = door["segment"]
            dbox = [min(x1, x2) - 1, min(y1, y2) - 1, max(x1, x2) + 1, max(y1, y2) + 1]
            nodes.append({
                "id": did, "kind": "front_door",
                "room_type_id": -1, "room_type_name": FRONT_DOOR_NODE_TYPE, "unified_type": "front_door",
                "area_ratio": 0.0, "polygon_area_ratio": 0.0,
                "bbox": dbox, "centroid": list(door["center"]),
                "grid_location_5x5": [min(4, max(0, int(door["center"][0] / canvas * 5))),
                                      min(4, max(0, int(door["center"][1] / canvas * 5)))],
                "is_entrance_room": False, "door_hops": -1, "door_distance": 0.0,
            })
            door_block["node_id"] = did
            if entrance is not None:
                rel = spatial_relation(dbox, boxes[entrance], ct, ot)
                w = float(door["width_px"])
                edges.append(_make_edge(did, entrance, dbox, boxes[entrance], rel, w, "entrance", False, False, canvas))
                edges.append(_make_edge(entrance, did, boxes[entrance], dbox, inverse_relation(rel), w,
                                        "entrance", False, False, canvas))

    for e in edges:                                   # front-door edges: same label in both fields
        e.setdefault("geometric_connection", e["connection"])
    return {
        "id": layout["id"],
        "dataset": layout.get("dataset", "rplan"),
        "nodes": nodes,
        "edges": edges,
        "front_door": door_block,
        "adjacency": {"source": source, "requested": requested, "wall_dilation_px": radius,
                      "min_contact_px": int(geo_cfg["min_contact_px"])},
        "comparison_with_source": {"geometric_pairs": len(geo), "source_pairs": len(src),
                                   "common_pairs": len(set(geo) & src),
                                   "source_pairs_without_shared_wall": len(src - set(geo))},
    }


def room_nodes(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Only the room nodes (excludes the virtual front-door node)."""
    return [nd for nd in graph["nodes"] if nd.get("kind", "room") == "room"]


def room_edges(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Only room-room edges (excludes edges to the front-door node)."""
    return [e for e in graph["edges"] if e.get("connection") != "entrance"]


def subgraph(graph: Dict[str, Any], visible_room_ids: Iterable[int]) -> Dict[str, Any]:
    """Partial graph that keeps only the visible rooms (and the front-door node).

    Original node ids are preserved. Door hops are recomputed on the visible
    graph so that the partial query does not leak information about hidden
    rooms. The door edge is kept only if the entrance room is visible.
    """
    visible = {int(i) for i in visible_room_ids}
    door_id = (graph.get("front_door") or {}).get("node_id")
    keep = visible | ({door_id} if door_id is not None else set())
    nodes = [dict(nd) for nd in graph["nodes"] if int(nd["id"]) in keep]
    edges = [dict(e) for e in graph["edges"] if int(e["source"]) in keep and int(e["target"]) in keep]

    entrance = (graph.get("front_door") or {}).get("entrance_room_id")
    entrance = entrance if entrance in visible else None
    ids = sorted(visible)
    pos = {nid: k for k, nid in enumerate(ids)}
    pairs = {(pos[int(e["source"])], pos[int(e["target"])]) for e in edges
             if e.get("connection") != "entrance" and int(e["source"]) in pos and int(e["target"]) in pos}
    hops = bfs_hops(len(ids), pairs, pos.get(entrance) if entrance is not None else None)
    for nd in nodes:
        if nd.get("kind", "room") == "room":
            nd["door_hops"] = int(hops[pos[int(nd["id"])]])
            nd["is_entrance_room"] = bool(entrance is not None and int(nd["id"]) == entrance)

    out = dict(graph)
    out["nodes"], out["edges"] = nodes, edges
    if graph.get("front_door"):
        fd = dict(graph["front_door"])
        fd["entrance_room_id"] = entrance
        out["front_door"] = fd
    out["partial"] = {"visible_room_ids": ids}
    return out


def sublayout(layout: Dict[str, Any], visible_room_ids: Iterable[int]) -> Dict[str, Any]:
    """Partial layout with only the visible rooms (boundary and front door are kept)."""
    visible = {int(i) for i in visible_room_ids}
    out = dict(layout)
    out["rooms"] = [r for r in layout["rooms"] if int(r["id"]) in visible]
    out["partial"] = {"visible_room_ids": sorted(visible),
                      "removed_room_ids": sorted({int(r["id"]) for r in layout["rooms"]} - visible)}
    return out
