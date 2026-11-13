"""Constraint search: score the catalog, keep plans meeting every hard requirement, rank, diversify.

Algorithm (see ``docs/constraint_search.md``):

1. restrict the catalog to the requested datasets and split;
2. score every plan (:func:`prism.constraints.scoring.score_record`);
3. candidates = plans passing every hard requirement, sorted by final score
   (ties: constraint score, then id) - deterministic, no RNG;
4. diversity: Maximal Marginal Relevance on the top ``20 k`` candidates with the
   graph-descriptor vectors (:func:`mmr`);
5. fewer than ``k`` candidates: all of them are returned and the shortfall is
   reported - violating plans are NEVER used to fill the list;
6. near misses: the best plans failing at least one hard requirement, with the
   violated requirements.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from prism.constraints.scoring import score_record
from prism.constraints.spec import requirements


def mmr(scores: Sequence[float], vectors: np.ndarray, k: int, lam: float) -> List[int]:
    """Greedy Maximal Marginal Relevance.

    ``next = argmax_i (1 - lam) * score_i - lam * max_{j in selected} cos(v_i, v_j)``.
    ``vectors`` must be L2-normalised (cosine = dot product). Returns row indices
    in pick order; ties go to the lower index, so the result is deterministic.
    """
    n = len(scores)
    if n == 0:
        return []
    scores = np.asarray(scores, dtype=np.float64)
    vecs = np.asarray(vectors, dtype=np.float64)
    selected: List[int] = []
    max_sim = np.full(n, -np.inf)
    remaining = np.ones(n, dtype=bool)
    for _ in range(min(k, n)):
        penalty = np.where(np.isfinite(max_sim), max_sim, 0.0)
        value = (1.0 - lam) * scores - lam * penalty
        value[~remaining] = -np.inf
        best = int(np.argmax(value))
        selected.append(best)
        remaining[best] = False
        max_sim = np.maximum(max_sim, vecs @ vecs[best])
    return selected


def _sort_key(s: Dict[str, Any]) -> tuple:
    """Ranking order: final desc, constraint score desc, id asc."""
    return (-round(s["final"], 12), -round(s["constraint_score"], 12), s["id"])


def funnel(scored: List[Dict[str, Any]], reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """How many plans each hard requirement removes, alone and cumulatively (in request order)."""
    rows = []
    alive = list(scored)
    for i, req in enumerate(reqs):
        if not req["hard"]:
            continue
        failing = lambda s: s["explanation"][i]["passed"] is False  # noqa: E731
        alone = sum(1 for s in scored if failing(s))
        before = len(alive)
        alive = [s for s in alive if not failing(s)]
        rows.append({"hard requirement": req["name"], "wanted": req["wanted"], "removes (alone)": alone,
                     "removed here": before - len(alive), "remaining": len(alive)})
    return rows


def blockers(scored: List[Dict[str, Any]], reqs: List[Dict[str, Any]], top: int = 5) -> List[Dict[str, Any]]:
    """Requirements that the most plans fail (hard or soft): what to relax first."""
    rows = []
    for i, req in enumerate(reqs):
        fail = sum(1 for s in scored if s["explanation"][i]["status"] in ("violated", "partly"))
        rows.append({"requirement": req["name"], "wanted": req["wanted"], "hard": req["hard"],
                     "plans not meeting it": fail, "share": fail / max(1, len(scored))})
    return sorted(rows, key=lambda r: (-r["plans not meeting it"], r["requirement"]))[:top]


def run_search(records: List[Dict[str, Any]], spec: Dict[str, Any], priors: Dict[str, Any], cfg: Dict[str, Any],
               vectors_for: Optional[Callable[[List[str]], Optional[np.ndarray]]] = None,
               fine_level: bool = False) -> Dict[str, Any]:
    """Run the whole search; returns results, near misses, all candidates and diagnostics.

    ``vectors_for(ids)`` returns L2-normalised descriptor vectors (rows in the
    order of ``ids``) or None; diversity is skipped without it.
    """
    ccfg = cfg.get("constraints", {})
    reqs = requirements(spec)
    pool = [r for r in records
            if (spec["datasets"] is None or r["dataset"] in spec["datasets"])
            and (spec["split"] == "all" or r["split"] == spec["split"])]
    rk = spec["ranking"]
    weights = dict(ccfg.get("plausibility_components", {"size": 0.4, "shape": 0.2, "reachability": 0.3,
                                                        "living_hub": 0.1}))
    by_id = {r["id"]: r for r in pool}
    scored = [score_record(r, reqs, priors, weights, rk["plausibility_weight"], fine_level) for r in pool]
    for s in scored:
        rec = by_id[s["id"]]
        s.update({"area_m2": rec.get("net_area_m2"), "gross_area_m2": rec.get("gross_area_m2"),
                  "area_source": rec.get("area_source"), "n_rooms": rec["n_rooms"], "counts": rec["counts"],
                  "split": rec["split"], "entrance_type": rec.get("entrance_type")})

    candidates = sorted([s for s in scored if s["passes_hard"]], key=_sort_key)
    k = spec["k"]
    lam = rk["diversity"]
    diversity_used = False
    if lam > 0 and len(candidates) > 1 and vectors_for is not None:
        head = candidates[: 20 * k]
        vecs = vectors_for([s["id"] for s in head])
        if vecs is not None and len(vecs) == len(head):
            order = mmr([s["final"] for s in head], vecs, k, lam)
            results = [head[i] for i in order]
            diversity_used = True
    if not diversity_used:
        results = candidates[:k]
    for rank, s in enumerate(results, 1):
        s["rank"] = rank

    misses = sorted([s for s in scored if not s["passes_hard"]], key=lambda s: (len(s["violated"]),) + _sort_key(s))
    return {"requirements": reqs, "pool_size": len(pool), "scored": scored, "candidates": candidates,
            "results": results, "near_misses": misses[: rk["near_misses"]], "diversity_used": diversity_used,
            "funnel": funnel(scored, reqs), "blockers": blockers(scored, reqs),
            "shortfall": max(0, k - len(candidates))}


def descriptor_vectors(ws: Any, store: Any, taxonomy: Any) -> Callable[[List[str]], Optional[np.ndarray]]:
    """Vector provider for MMR: the phase-07 ``graph_descriptor`` index, else descriptors computed on the fly."""
    from prism.retrieval.index import VectorIndex, l2_normalize

    index = None
    if (ws.index_dir("graph_descriptor") / "vectors.npy").exists():
        try:
            index = VectorIndex.load(ws.index_dir("graph_descriptor"), backend="numpy")
        except Exception as exc:  # unreadable index: fall back to on-the-fly descriptors
            print(f"  [warn] could not load the graph_descriptor index ({exc}); computing descriptors")

    def provide(ids: List[str]) -> Optional[np.ndarray]:
        """L2-normalised vectors for ``ids`` (index rows when all ids are there, else fresh descriptors)."""
        if index is not None and all(i in index.id_to_row for i in ids):
            return np.asarray(index.prepared[[index.id_to_row[i] for i in ids]], dtype=np.float64)
        from prism.graph.descriptor import graph_descriptor

        x = np.stack([graph_descriptor(store.graph(i), store.layout(i), taxonomy)[0] for i in ids]).astype(np.float64)
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)          # standardise among the candidates
        return l2_normalize(x)

    return provide
