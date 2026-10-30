"""Architectural metrics between two floor plans (Pier's evaluation, extended).

Each layout is first summarised once (:class:`LayoutSummary`), then pairs are
compared with generalised Jaccard similarities on counters:

- room types (fine and unified), room count, area by type
- typed adjacency (undirected room-type pairs)
- spatial relations and directed relation triplets (our recomputed relations)
- (room type, 5x5 grid cell) occupancy
- simple constraints (kitchen/bathroom count, bedrooms +-1, rooms +-1, living room)
- FRONT DOOR: entrance-room type match and door facing match

``architectural_score`` keeps Pier's weights for continuity; the door terms are
reported separately as ``door_score``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import numpy as np

from src.prism.datasets.schema import Taxonomy
from src.prism.graph.build import room_edges, room_nodes


@dataclass
class LayoutSummary:
    """Counters describing one (possibly partial) layout graph."""

    sid: str
    n_rooms: int
    type_counts: Counter = field(default_factory=Counter)
    type_id_counts: Counter = field(default_factory=Counter)
    unified_counts: Counter = field(default_factory=Counter)
    area_by_type: Counter = field(default_factory=Counter)
    adjacency: Counter = field(default_factory=Counter)
    relations: Counter = field(default_factory=Counter)
    triplets: Counter = field(default_factory=Counter)
    grid: Counter = field(default_factory=Counter)
    bedrooms: int = 0
    entrance_type: Optional[str] = None
    door_facing: Optional[str] = None

    @property
    def composition_signature(self) -> Tuple[Tuple[str, int], ...]:
        """Hashable unified room composition, used to bucket relevant layouts."""
        return tuple(sorted(self.unified_counts.items()))


def summarize(graph: Dict[str, Any], taxonomy: Taxonomy) -> LayoutSummary:
    """Build the :class:`LayoutSummary` of a graph dictionary."""
    nodes = room_nodes(graph)
    edges = room_edges(graph)
    s = LayoutSummary(sid=str(graph["id"]), n_rooms=len(nodes))
    name_of = {}
    for nd in nodes:
        name = taxonomy.name(int(nd["room_type_id"]))
        name_of[int(nd["id"])] = name
        s.type_counts[name] += 1
        s.type_id_counts[int(nd["room_type_id"])] += 1
        s.unified_counts[nd["unified_type"]] += 1
        s.area_by_type[name] += float(nd.get("polygon_area_ratio", nd["area_ratio"]))
        gx, gy = nd["grid_location_5x5"]
        s.grid[(name, int(gx), int(gy))] += 1
        if taxonomy.is_bedroom(int(nd["room_type_id"])):
            s.bedrooms += 1
    for e in edges:
        u, v = int(e["source"]), int(e["target"])
        if u not in name_of or v not in name_of:
            continue
        s.triplets[(name_of[u], e["relation"], name_of[v])] += 1
        if u < v:
            s.adjacency[tuple(sorted((name_of[u], name_of[v])))] += 1
            s.relations[e["relation"]] += 1
    fd = graph.get("front_door") or {}
    if fd.get("entrance_room_id") is not None:
        s.entrance_type = name_of.get(int(fd["entrance_room_id"]))
    s.door_facing = fd.get("facing")
    return s


def generalized_jaccard(a: Counter, b: Counter) -> float:
    """``sum min(a_k, b_k) / sum max(a_k, b_k)`` (1.0 when both are empty)."""
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    num = sum(min(float(a.get(k, 0)), float(b.get(k, 0))) for k in keys)
    den = sum(max(float(a.get(k, 0)), float(b.get(k, 0))) for k in keys)
    return num / den if den > 0 else 1.0


def set_jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    """Plain Jaccard similarity of two sets."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def constraint_metrics(q: LayoutSummary, r: LayoutSummary) -> Dict[str, Any]:
    """Simple functional constraints derived from the query (Pier's list)."""
    vals = {
        "constraint_same_kitchen_count": int(q.unified_counts.get("kitchen", 0) == r.unified_counts.get("kitchen", 0)),
        "constraint_same_bathroom_count": int(q.unified_counts.get("bathroom", 0) == r.unified_counts.get("bathroom", 0)),
        "constraint_bedroom_count_diff_le_1": int(abs(q.bedrooms - r.bedrooms) <= 1),
        "constraint_room_count_diff_le_1": int(abs(q.n_rooms - r.n_rooms) <= 1),
        "constraint_livingroom_covered": int(q.unified_counts.get("living", 0) == 0 or r.unified_counts.get("living", 0) >= 1),
    }
    scores = list(vals.values())
    vals["constraint_score"] = float(np.mean(scores))
    vals["hard_constraints_satisfied"] = int(all(scores))
    return vals


def door_metrics(q: LayoutSummary, r: LayoutSummary) -> Dict[str, float]:
    """Front-door agreement: same entrance-room type, same facing direction."""
    entrance = float(q.entrance_type is not None and q.entrance_type == r.entrance_type)
    facing = float(q.door_facing not in (None, "unknown") and q.door_facing == r.door_facing)
    return {"door_entrance_type_match": entrance, "door_facing_match": facing,
            "door_score": 0.5 * entrance + 0.5 * facing}


def evaluate_pair(q: LayoutSummary, r: LayoutSummary) -> Dict[str, Any]:
    """Every architectural metric for a (query, retrieved) pair."""
    count_sim = max(0.0, 1.0 - abs(q.n_rooms - r.n_rooms) / max(q.n_rooms, r.n_rooms, 1))
    m = {
        "query_num_rooms": q.n_rooms,
        "retrieved_num_rooms": r.n_rooms,
        "room_count_similarity": count_sim,
        "room_type_count_similarity": generalized_jaccard(q.type_counts, r.type_counts),
        "unified_type_count_similarity": generalized_jaccard(q.unified_counts, r.unified_counts),
        "room_type_set_jaccard": set_jaccard(q.type_counts.keys(), r.type_counts.keys()),
        "room_area_by_type_similarity": generalized_jaccard(q.area_by_type, r.area_by_type),
        "adjacency_type_similarity": generalized_jaccard(q.adjacency, r.adjacency),
        "relation_similarity": generalized_jaccard(q.relations, r.relations),
        "relation_triplet_similarity": generalized_jaccard(q.triplets, r.triplets),
        "grid_type_similarity": generalized_jaccard(q.grid, r.grid),
        "same_composition": int(q.composition_signature == r.composition_signature),
    }
    m.update(constraint_metrics(q, r))
    m.update(door_metrics(q, r))
    m["architectural_score"] = (0.25 * m["room_type_count_similarity"] + 0.15 * count_sim
                                + 0.15 * m["room_area_by_type_similarity"] + 0.20 * m["adjacency_type_similarity"]
                                + 0.10 * m["relation_similarity"] + 0.10 * m["grid_type_similarity"]
                                + 0.05 * m["constraint_score"])
    return m


def relevant_set(q: LayoutSummary, gallery: Dict[str, LayoutSummary], buckets: Dict[Tuple, Set[str]],
                 tau: float) -> Set[str]:
    """Gallery ids relevant to ``q``: same unified composition and adjacency similarity >= ``tau``.

    The query itself is excluded.
    """
    out = set()
    for sid in buckets.get(q.composition_signature, set()):
        if sid != q.sid and generalized_jaccard(q.adjacency, gallery[sid].adjacency) >= tau:
            out.add(sid)
    return out


def ranking_metrics(ranked_ids: Iterable[str], relevant: Set[str], k: int) -> Dict[str, float]:
    """Recall@K (normalised by ``min(K, |R|)``), Hit@K, Precision@K and reciprocal rank."""
    ranked = list(ranked_ids)[:k]
    hits = [1 if sid in relevant else 0 for sid in ranked]
    rr = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            rr = 1.0 / i
            break
    if not relevant:
        return {"recall_at_k": float("nan"), "hit_at_k": float("nan"), "precision_at_k": float("nan"),
                "mrr": float("nan"), "num_relevant": 0}
    return {"recall_at_k": sum(hits) / min(k, len(relevant)), "hit_at_k": float(any(hits)),
            "precision_at_k": sum(hits) / max(1, len(ranked)), "mrr": rr, "num_relevant": len(relevant)}


def graph_edit_distance(g1: Dict[str, Any], g2: Dict[str, Any], timeout: float = 2.0) -> Optional[float]:
    """Optional normalised graph edit distance with networkx (slow; use on few pairs).

    Node substitution costs 1 when the unified room types differ. The distance
    is divided by the size of the larger graph (nodes + undirected edges).
    """
    try:
        import networkx as nx
    except ImportError:
        return None

    def to_nx(g):
        """Undirected networkx graph of the room nodes."""
        G = nx.Graph()
        for nd in room_nodes(g):
            G.add_node(int(nd["id"]), t=nd["unified_type"])
        for e in room_edges(g):
            if int(e["source"]) < int(e["target"]):
                G.add_edge(int(e["source"]), int(e["target"]))
        return G

    a, b = to_nx(g1), to_nx(g2)
    ged = nx.graph_edit_distance(a, b, node_match=lambda x, y: x["t"] == y["t"], timeout=timeout)
    if ged is None:
        return None
    size = max(a.number_of_nodes() + a.number_of_edges(), b.number_of_nodes() + b.number_of_edges(), 1)
    return float(ged) / size
