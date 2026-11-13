"""Scores of the constraint search (every score is in [0, 1]).

- :func:`interval_satisfaction` / :func:`target_closeness`: satisfaction of one
  numeric requirement (counts use ``scale = 1`` room, areas ``tolerance_m2``);
- :func:`evaluate_requirement`: one explanation row per requirement
  (``wanted``, ``value``, ``score``, ``hard``, ``status``, ``passed``);
- :data:`PLAUSIBILITY`: registry of plausibility components (size, shape,
  reachability, living hub) computed from the dataset's own statistics;
- :func:`score_record`: constraint score, plausibility and final score.

::

    constraint_score = sum_c w_c * s_c / sum_c w_c        (evaluable requirements only)
    plausibility     = sum_i w_i * p_i / sum_i w_i
    final            = (1 - plausibility_weight) * constraint_score + plausibility_weight * plausibility
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

MET_EPS = 1e-9


# ----------------------------------------------------------------------------
# Numeric satisfaction
# ----------------------------------------------------------------------------
def interval_satisfaction(x: float, lo: Optional[float] = None, hi: Optional[float] = None, scale: float = 1.0) -> float:
    """1 inside [lo, hi]; decays as exp(-d / scale) with the distance d to the interval outside it."""
    d = max(0.0, (lo - x) if lo is not None else 0.0, (x - hi) if hi is not None else 0.0)
    return math.exp(-d / scale)


def target_closeness(x: float, target: float, scale: float) -> float:
    """Gaussian closeness to a target: exp(-0.5 * ((x - target) / scale) ** 2)."""
    return math.exp(-0.5 * ((x - target) / scale) ** 2)


def hard_interval(req: Dict[str, Any]) -> tuple:
    """Interval a HARD requirement must hit; a target alone means [target - tolerance, target + tolerance]."""
    lo, hi = req.get("min"), req.get("max")
    if req.get("kind") == "area" and lo is None and hi is None and req.get("target") is not None:
        return req["target"] - req["tolerance_m2"], req["target"] + req["tolerance_m2"]
    return lo, hi


def area_score(x: float, req: Dict[str, Any]) -> float:
    """Area satisfaction: interval and target -> average; interval only -> interval; target only -> closeness."""
    has_interval = req.get("min") is not None or req.get("max") is not None
    s_int = interval_satisfaction(x, req.get("min"), req.get("max"), req["tolerance_m2"]) if has_interval else None
    s_tgt = target_closeness(x, req["target"], req["tolerance_m2"]) if req.get("target") is not None else None
    if s_int is not None and s_tgt is not None:
        return 0.5 * s_int + 0.5 * s_tgt
    return s_int if s_int is not None else s_tgt


def _inside(x: float, lo: Optional[float], hi: Optional[float]) -> bool:
    """True when ``x`` lies in [lo, hi] (open sides allowed)."""
    return (lo is None or x >= lo - MET_EPS) and (hi is None or x <= hi + MET_EPS)


# ----------------------------------------------------------------------------
# One requirement
# ----------------------------------------------------------------------------
def requirement_value(record: Dict[str, Any], req: Dict[str, Any], fine_level: bool = False) -> Any:
    """Value of the plan for a requirement (None when unknown)."""
    kind = req["kind"]
    if kind == "area":
        return record.get("net_area_m2") if req.get("basis", "net") == "net" else record.get("gross_area_m2")
    if kind == "count":
        if req.get("field") == "n_rooms":
            return record["n_rooms"]
        t = req["type"]
        if t in record["counts"]:
            return record["counts"][t]
        return record.get("fine_counts", {}).get(t, 0) if fine_level else 0
    if kind == "entrance":
        return record.get("entrance_type")
    if kind == "adjacent":
        return list(req["pair"]) in [list(p) for p in record.get("adjacent_pairs", [])]
    raise ValueError(f"unknown requirement kind {kind!r}")


def evaluate_requirement(record: Dict[str, Any], req: Dict[str, Any], fine_level: bool = False) -> Dict[str, Any]:
    """Explanation row of one requirement for one plan.

    ``passed`` = whether a HARD requirement is met (None when unknown, True for
    soft ones). ``status`` is ``met`` / ``partly`` / ``violated`` / ``unknown``;
    ``score`` is None when the requirement cannot be evaluated (its weight is
    then removed from the average).
    """
    kind, value = req["kind"], requirement_value(record, req, fine_level)
    row = {"requirement": req["name"], "wanted": req["wanted"], "value": value, "score": None,
           "hard": bool(req["hard"]), "weight": float(req["weight"]), "status": "unknown", "passed": None}
    if kind == "area":
        if value is None:
            # Unknown area (RPLAN without calibration): the weight is dropped from the average.
            # A hard requirement keeps the plan unless require_known_area is set.
            if req["hard"] and req.get("require_known_area"):
                row["passed"], row["status"] = False, "violated"
            else:
                row["passed"] = None if req["hard"] else True
            return row
        row["value"] = round(float(value), 1)
        row["score"] = area_score(float(value), req)
        met = _inside(float(value), *hard_interval(req))
    elif kind == "count":
        row["score"] = interval_satisfaction(float(value), req.get("min"), req.get("max"), 1.0)
        met = _inside(float(value), req.get("min"), req.get("max"))
    elif kind == "entrance":
        if value is None:                       # no front door: unknown, half credit
            row["score"] = 0.5
            row["passed"] = None if req["hard"] else True
            return row
        met = value == req["type"]
        row["score"] = 1.0 if met else 0.0
    elif kind == "adjacent":
        has_both = all(record["counts"].get(t, 0) > 0 for t in req["pair"])
        met = bool(value) and has_both
        row["score"] = 1.0 if met else 0.0
        row["value"] = "yes" if met else ("no" if has_both else "type missing")
    else:
        raise ValueError(f"unknown requirement kind {kind!r}")
    row["passed"] = met if req["hard"] else True
    if met:
        row["status"] = "met"                   # inside the interval (possibly away from an area target)
    elif req["hard"] or row["score"] < 0.05:
        row["status"] = "violated"
    else:
        row["status"] = "partly"
    return row


# ----------------------------------------------------------------------------
# Plausibility components (registry: name -> function(record, priors) -> [0, 1])
# ----------------------------------------------------------------------------
def _type_prior(priors: Dict[str, Any], t: str) -> Dict[str, Any]:
    """Prior of a unified type (falls back to the prior over all rooms)."""
    return priors.get("types", {}).get(t) or priors.get("all", {})


def plaus_size(record: Dict[str, Any], priors: Dict[str, Any]) -> float:
    """Mean over rooms with a known area of exp(-0.5 z²), z = standardised log area of the room's type (0.5 if none)."""
    vals = []
    for r in record["rooms"]:
        p = _type_prior(priors, r["unified"])
        if r.get("area_m2") and p.get("mu_logA") is not None:
            z = (math.log(r["area_m2"]) - p["mu_logA"]) / p["sd_logA"]
            vals.append(math.exp(-0.5 * z * z))
    return sum(vals) / len(vals) if vals else 0.5


def plaus_shape(record: Dict[str, Any], priors: Dict[str, Any]) -> float:
    """Mean over rooms of exp(-0.5 z²), z = standardised log aspect ratio of the room's type."""
    vals = []
    for r in record["rooms"]:
        p = _type_prior(priors, r["unified"])
        if p.get("mu_logR") is not None:
            z = (math.log(max(1.0, r["aspect"])) - p["mu_logR"]) / p["sd_logR"]
            vals.append(math.exp(-0.5 * z * z))
    return sum(vals) / len(vals) if vals else 0.5


def plaus_reachability(record: Dict[str, Any], priors: Dict[str, Any]) -> float:
    """Share of rooms reachable from the entrance room through the room graph (0.5 without a front door)."""
    share = record.get("reachable_share")
    return 0.5 if share is None else float(share)


def plaus_living_hub(record: Dict[str, Any], priors: Dict[str, Any]) -> float:
    """1 if a living room touches >= 50% of the other rooms or is the entrance room; else that share; 0.5 if none."""
    share = record.get("living_neighbour_share")
    if share is None:
        return 0.5
    if share >= 0.5 or record.get("entrance_type") == "living":
        return 1.0
    return float(share)


PLAUSIBILITY: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], float]] = {
    "size": plaus_size,
    "shape": plaus_shape,
    "reachability": plaus_reachability,
    "living_hub": plaus_living_hub,
}


def plausibility(record: Dict[str, Any], priors: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    """Weighted plausibility and its components."""
    comps = {name: float(PLAUSIBILITY[name](record, priors)) for name in weights if name in PLAUSIBILITY}
    total = sum(float(weights[n]) for n in comps)
    value = sum(float(weights[n]) * comps[n] for n in comps) / total if total > 0 else 0.5
    return {"value": value, "components": comps}


# ----------------------------------------------------------------------------
# Whole plan
# ----------------------------------------------------------------------------
def score_record(record: Dict[str, Any], reqs: List[Dict[str, Any]], priors: Dict[str, Any],
                 plaus_weights: Dict[str, float], plausibility_weight: float,
                 fine_level: bool = False) -> Dict[str, Any]:
    """Explanations, hard pass / fail, constraint score, plausibility and final score of one plan."""
    rows = [evaluate_requirement(record, r, fine_level) for r in reqs]
    scored = [r for r in rows if r["score"] is not None and r["weight"] > 0]
    wsum = sum(r["weight"] for r in scored)
    constraint = sum(r["weight"] * r["score"] for r in scored) / wsum if wsum > 0 else 1.0
    plaus = plausibility(record, priors, plaus_weights)
    final = (1.0 - plausibility_weight) * constraint + plausibility_weight * plaus["value"]
    violated = [r["requirement"] for r in rows if r["hard"] and r["passed"] is False]
    return {"id": record["id"], "dataset": record["dataset"], "explanation": rows,
            "constraint_score": constraint, "plausibility": plaus["value"],
            "plausibility_components": plaus["components"], "final": final,
            "passes_hard": not violated, "violated": violated,
            "unknown": [r["requirement"] for r in rows if r["status"] == "unknown"]}
