"""Graph JSON -> NumPy arrays (node features, edge index, edge attributes).

This module has no torch dependency, so it can be unit-tested anywhere; the
torch conversion happens in :mod:`prism.graph.data`.

Node features (per node)::

    one-hot node type   (fine or unified vocabulary, plus one slot for FrontDoor)
    area ratio, centroid x, centroid y, bbox width, bbox height   (normalised)
    is_entrance_room, door_hops / 10 (1.0 when unknown), door_distance

Edge features (per directed edge)::

    one-hot relation (10) | one-hot connection (5) | dx, dy, overlap_x, overlap_y, shared_wall / canvas
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.prism.datasets.schema import CONNECTION_TO_ID, CONNECTIONS, RELATION_TO_ID, RELATIONS, Taxonomy

NUM_GEOM_NODE_FEATS = 5
NUM_DOOR_NODE_FEATS = 3
NUM_GEOM_EDGE_FEATS = 5


def node_type_vocab(taxonomy: Taxonomy, kind: str) -> List[str]:
    """Node-type vocabulary (fine or unified) with the virtual ``FrontDoor`` type appended."""
    return taxonomy.vocab(kind) + ["FrontDoor"]


def feature_dims(taxonomy: Taxonomy, kind: str) -> Tuple[int, int]:
    """Return ``(node_feature_dim, edge_feature_dim)``."""
    node_dim = len(node_type_vocab(taxonomy, kind)) + NUM_GEOM_NODE_FEATS + NUM_DOOR_NODE_FEATS
    edge_dim = len(RELATIONS) + len(CONNECTIONS) + NUM_GEOM_EDGE_FEATS
    return node_dim, edge_dim


def graph_to_arrays(graph: Dict[str, Any], taxonomy: Taxonomy, kind: str = "fine", include_door: bool = True,
                    canvas: float = 256.0) -> Dict[str, np.ndarray]:
    """Convert a graph dictionary into ``x``, ``edge_index`` and ``edge_attr`` arrays.

    Node ids in the JSON may be non-contiguous (partial graphs keep original
    ids), so nodes are re-indexed by their order in ``graph['nodes']``.
    """
    vocab = node_type_vocab(taxonomy, kind)
    vocab_index = {name: i for i, name in enumerate(vocab)}
    nodes = [nd for nd in graph["nodes"] if include_door or nd.get("kind", "room") == "room"]
    pos = {int(nd["id"]): k for k, nd in enumerate(nodes)}
    node_dim, edge_dim = feature_dims(taxonomy, kind)

    x = np.zeros((len(nodes), node_dim), dtype=np.float32)
    for k, nd in enumerate(nodes):
        if nd.get("kind", "room") == "front_door":
            type_name = "FrontDoor"
        else:
            type_name = taxonomy.name(int(nd["room_type_id"])) if kind == "fine" else nd["unified_type"]
        x[k, vocab_index.get(type_name, vocab_index.get("other", 0))] = 1.0
        x0, y0, x1, y1 = [float(v) for v in nd["bbox"][:4]]
        base = len(vocab)
        x[k, base + 0] = float(nd.get("polygon_area_ratio", nd.get("area_ratio", 0.0)))
        x[k, base + 1] = float(nd["centroid"][0]) / canvas
        x[k, base + 2] = float(nd["centroid"][1]) / canvas
        x[k, base + 3] = max(0.0, x1 - x0) / canvas
        x[k, base + 4] = max(0.0, y1 - y0) / canvas
        hops = int(nd.get("door_hops", -1))
        x[k, base + 5] = 1.0 if nd.get("is_entrance_room") else 0.0
        x[k, base + 6] = min(hops, 10) / 10.0 if hops >= 0 else 1.0
        x[k, base + 7] = max(0.0, float(nd.get("door_distance", 0.0)))

    src, dst, attrs = [], [], []
    for e in graph["edges"]:
        u, v = int(e["source"]), int(e["target"])
        if u not in pos or v not in pos:
            continue
        a = np.zeros(edge_dim, dtype=np.float32)
        a[RELATION_TO_ID[e["relation"]]] = 1.0
        a[len(RELATIONS) + CONNECTION_TO_ID.get(e.get("connection", "unknown"), CONNECTION_TO_ID["unknown"])] = 1.0
        g = len(RELATIONS) + len(CONNECTIONS)
        a[g + 0] = float(e.get("dx", 0.0))
        a[g + 1] = float(e.get("dy", 0.0))
        a[g + 2] = float(e.get("overlap_x", 0.0))
        a[g + 3] = float(e.get("overlap_y", 0.0))
        a[g + 4] = float(e.get("shared_wall_px", 0.0)) / canvas
        src.append(pos[u])
        dst.append(pos[v])
        attrs.append(a)

    edge_index = np.asarray([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.stack(attrs).astype(np.float32) if attrs else np.zeros((0, edge_dim), dtype=np.float32)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
