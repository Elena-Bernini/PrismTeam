"""Compatibility of a complete candidate with a partial query (re-ranking stage).

A candidate is a plausible completion of the partial query when it CONTAINS
what the query shows. Hence the scores are coverages (how much of the query is
found in the candidate), not symmetric similarities:

- room coverage        known room types (with multiplicity) present in the candidate
- adjacency coverage   typed adjacencies of the query present in the candidate
- relation coverage    (type, relation, type) triplets present in the candidate
- grid coverage        (type, grid cell) pairs present in the candidate
- size compatibility   candidate must have >= visible rooms; extra rooms decay as exp(-extra/6)
- door compatibility   same entrance-room type (if visible) and same door facing
- vector similarity    the stage-1 similarity as a weak tie-breaker

``f1_type_adjacency`` (Elena's score: F1 of room types and adjacencies) is also
reported for comparison.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict

from src.prism.evaluation.metrics import LayoutSummary


def counter_coverage(query: Counter, candidate: Counter) -> float:
    """Fraction of the query multiset found in the candidate (1.0 for an empty query)."""
    total = sum(query.values())
    if total == 0:
        return 1.0
    return sum(min(c, candidate.get(k, 0)) for k, c in query.items()) / total


def set_coverage(query: Counter, candidate: Counter) -> float:
    """Fraction of distinct query keys present in the candidate."""
    if not query:
        return 1.0
    return sum(1 for k in query if k in candidate) / len(query)


def f1_counts(query: Counter, candidate: Counter) -> float:
    """F1 between two multisets (Elena's coverage, penalises extra candidate items)."""
    qt, ct = sum(query.values()), sum(candidate.values())
    if qt == 0 or ct == 0:
        return 0.0
    matched = sum(min(c, candidate.get(k, 0)) for k, c in query.items())
    p, r = matched / ct, matched / qt
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def compatibility(q: LayoutSummary, c: LayoutSummary, vector_sim: float, weights: Dict[str, float]) -> Dict[str, Any]:
    """Return every compatibility term and the weighted ``compatibility_score``."""
    room = counter_coverage(q.type_counts, c.type_counts)
    adj = set_coverage(q.adjacency, c.adjacency)
    rel = set_coverage(q.triplets, c.triplets)
    grid = set_coverage(q.grid, c.grid)
    size = 0.0 if c.n_rooms < q.n_rooms else math.exp(-(c.n_rooms - q.n_rooms) / 6.0)
    door_terms = []
    if q.entrance_type is not None:
        door_terms.append(float(q.entrance_type == c.entrance_type))
    if q.door_facing not in (None, "unknown"):
        door_terms.append(float(q.door_facing == c.door_facing))
    door = sum(door_terms) / len(door_terms) if door_terms else 1.0
    vec = max(0.0, min(1.0, (vector_sim + 1.0) / 2.0)) if vector_sim <= 1.0 else 1.0
    score = (weights["room"] * room + weights["adjacency"] * adj + weights["relation"] * rel
             + weights["grid"] * grid + weights["size"] * size + weights["door"] * door + weights["vector"] * vec)
    return {
        "compatibility_score": float(score),
        "room_coverage": room, "adjacency_coverage": adj, "relation_coverage": rel,
        "grid_coverage": grid, "size_compatibility": size, "door_compatibility": door,
        "vector_similarity": float(vector_sim),
        "f1_type_adjacency": 0.7 * f1_counts(q.type_counts, c.type_counts) + 0.3 * f1_counts(q.adjacency, c.adjacency),
        "hard_constraints_satisfied": float(room >= 0.999 and size > 0 and adj >= 0.5),
    }
