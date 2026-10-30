"""Build the graph JSON of a floor plan from its layout JSON.

Graph JSON (``graphs_json/<id>.json``)::

    {
      "id": "75342", "dataset": "rplan",
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
          "connection": "wall",                        # wall | door | arch | entrance | unknown
          "shared_wall_px": 38.0,
          "dx": -0.2, "dy": 0.01, "overlap_x": 0.0, "overlap_y": 0.93,
          "in_geometric": true, "in_g2p": true
      }, ...],
      "front_door": {"facing": "south", "orientation": "horizontal", "center": [...],
                     "width_px": 12.0, "entrance_room_id": 3, "node_id": 8},
      "adjacency": {"source": "geometric", "wall_dilation_px": 3, "min_contact_px": 5},
      "comparison_with_g2p": {"geometric_pairs": 9, "g2p_pairs": 8, "common_pairs": 7}
    }
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from src.prism.datasets.schema import FRONT_DOOR_NODE_TYPE, RELATION_TO_ID
from src.prism.geometry.adjacency import geometric_adjacency
from src.prism.geometry.polygon import bbox_to_polygon
from src.prism.geometry.relations import inverse_relation, relation_features, spatial_relation


def _room_polygon(room: Dict[str, Any]) -> List[List[float]]:
    """Polygon of a room, falling back to its bounding box rectangle."""
    poly = room.get("polygon") or []
    return poly if len(poly) >= 3 else bbox_to_polygon(room["bbox"])


def _make_edge(u: int, v: int, box_u, box_v, rel: str, shared: float, connection: str,
               in_geo: bool, in_g2p: bool, canvas: float) -> Dict[str, Any]:
    """Assemble one directed edge record with its geometric attributes."""
    f = relation_features(box_u, box_v, canvas)
    return {
        "source": int(u), "target": int(v),
        "relation": rel, "relation_id": RELATION_TO_ID[rel],
        "connection": connection,
        "shared_wall_px": float(shared),
        "dx": f["dx"], "dy": f["dy"],
        "overlap_x": min(1.0, f["overlap_x"]), "overlap_y": min(1.0, f["overlap_y"]),
        "in_geometric": bool(in_geo), "in_g2p": bool(in_g2p),
    }


def g2p_pairs(layout: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """Undirected room pairs listed in the original Graph2Plan edge list."""
    n = len(layout["rooms"])
    out: Set[Tuple[int, int]] = set()
    for e in layout.get("source_edges", []):
        u, v = int(e["source"]), int(e["target"])
        if 0 <= u < n and 0 <= v < n and u != v:
            out.add((min(u, v), max(u, v)))
    return out


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
    """Compute nodes, adjacency, spatial relations and the front-door node for one layout."""
    geo_cfg = cfg["geometry"]
    canvas = float(layout["coordinate_system"]["width"])
    rooms = layout["rooms"]
    n = len(rooms)
    polys = [_room_polygon(r) for r in rooms]
    boxes = [r["bbox"] for r in rooms]

    geo = geometric_adjacency(polys, boxes, canvas=int(canvas),
                              wall_dilation_px=int(geo_cfg["wall_dilation_px"]),
                              min_contact_px=int(geo_cfg["min_contact_px"]))
    g2p = g2p_pairs(layout)
    source = geo_cfg["adjacency_source"]
    if source == "geometric":
        pairs = dict(geo)
    elif source == "g2p":
        pairs = {p: geo.get(p, 0.0) for p in g2p}
    elif source == "union":
        pairs = dict(geo)
        for p in g2p:
            pairs.setdefault(p, 0.0)
    else:
        raise ValueError(f"Unknown adjacency_source '{source}'")

    # Connection type from the source dataset when it provides one (ResPlan:
    # door / arch / wall). RPLAN does not, so its edges are "wall".
    conn_map: Dict[Tuple[int, int], str] = {}
    for e in layout.get("source_edges", []):
        if e.get("connection"):
            u, v = int(e["source"]), int(e["target"])
            conn_map[(min(u, v), max(u, v))] = str(e["connection"])

    ct, ot = float(geo_cfg["containment_threshold"]), float(geo_cfg["axis_overlap_threshold"])
    edges: List[Dict[str, Any]] = []
    for (i, j), shared in sorted(pairs.items()):
        rel = spatial_relation(boxes[i], boxes[j], ct, ot)
        in_geo, in_g = (i, j) in geo, (i, j) in g2p
        conn = conn_map.get((i, j), "wall")
        edges.append(_make_edge(i, j, boxes[i], boxes[j], rel, shared, conn, in_geo, in_g, canvas))
        edges.append(_make_edge(j, i, boxes[j], boxes[i], inverse_relation(rel), shared, conn, in_geo, in_g, canvas))

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

    return {
        "id": layout["id"],
        "dataset": layout.get("dataset", "rplan"),
        "nodes": nodes,
        "edges": edges,
        "front_door": door_block,
        "adjacency": {"source": source, "wall_dilation_px": int(geo_cfg["wall_dilation_px"]),
                      "min_contact_px": int(geo_cfg["min_contact_px"])},
        "comparison_with_g2p": {"geometric_pairs": len(geo), "g2p_pairs": len(g2p),
                                "common_pairs": len(set(geo) & g2p)},
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
