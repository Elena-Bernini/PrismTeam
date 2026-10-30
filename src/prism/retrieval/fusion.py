"""Rank fusion of several retrieval pipelines (Elena's hybrid retrieval)."""

from __future__ import annotations

from typing import Dict, List, Sequence


def reciprocal_rank_fusion(rankings: Sequence[List[Dict]], k: int = 60, top_k: int = 10) -> List[Dict]:
    """Reciprocal Rank Fusion (Cormack et al. 2009).

    Each input is a ranked list of ``{"id": ..., ...}``. The fused score of a
    document is ``sum_r 1 / (k + rank_r)`` over the rankings that contain it,
    so items ranked high by several pipelines win. ``k=60`` is the usual value.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [{"id": sid, "score": s} for sid, s in fused]
