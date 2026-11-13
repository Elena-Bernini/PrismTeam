"""Plan catalog for the constraint search: one record of searchable attributes per plan.

The catalog is a cache (``artifacts/<dataset>/<mode>/constraints/catalog.jsonl``)
rebuilt automatically when the processed graphs change. Each record holds room
counts (unified and fine types), per-room area and aspect ratio, net / gross
area in m², the entrance-room type, reachability from the front door and the
adjacent room-type pairs (see :func:`plan_record`).

**Where the m² come from** (every record states its ``area_source``):

- ResPlan: the dataset's own m² (``"dataset"``). ResPlan coordinates are
  normalised PER PLAN (each plan is scaled into a ~256-unit box), so only the
  plan's ``area`` field carries metres. Checked on the real 2026-07-28 release:
  ``area`` is consistent (48-511 m², and the wall thickness it implies is
  0.22 m +- 15% across plans), while ``net_area`` is not (0 in > 25% of plans,
  up to 2.5e7 in others). Hence: ``gross_area_m2 = area``; the plan's m² per
  canvas px² = ``area / outline px²``; each room = its ORIGINAL wall-free polygon
  (``source_polygons``) in px² x that factor; ``net_area_m2`` = sum of rooms.
  ``net_area`` is used only when ``area`` is missing and ``net_area`` looks valid.
  Without dataset areas, and when ``constraints.resplan_units == "m"``, the
  geometry in source units is used (``"geometry_metric"``).
- RPLAN has no metric scale: canvas px² are converted with a scale
  **calibrated** on ResPlan room sizes (median over ``calibration_types`` of
  ``median m²(ResPlan, t) / median px²(RPLAN, t)``), or with a manual
  ``constraints.rplan_m2_per_px2`` (``"calibrated"``). Without either, RPLAN
  areas are unknown (``None``).

The size / shape priors used by the plausibility score (:func:`fit_priors`)
are fitted on the catalog itself.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prism.common.io import progress, read_json, read_jsonl, write_json, write_jsonl
from prism.datasets.schema import dataset_of

CATALOG_VERSION = 1


# ----------------------------------------------------------------------------
# One record per plan
# ----------------------------------------------------------------------------
def _aspect(bbox: List[float]) -> float:
    """``max(w, h) / min(w, h)`` of a bounding box (>= 1)."""
    w, h = max(1e-6, float(bbox[2]) - float(bbox[0])), max(1e-6, float(bbox[3]) - float(bbox[1]))
    return max(w, h) / min(w, h)


def plan_record(sid: str, layout: Dict[str, Any], graph: Dict[str, Any], split: str,
                unified_names: List[str]) -> Dict[str, Any]:
    """Searchable attributes of one plan (areas in m² are filled later by :func:`apply_areas`)."""
    from prism.graph.build import room_edges, room_nodes

    width = float(layout["coordinate_system"]["width"])
    nodes = room_nodes(graph)
    by_id = {int(n["id"]): n for n in nodes}
    counts = {t: 0 for t in unified_names}
    fine = Counter()
    rooms = []
    from prism.geometry.polygon import polygon_area

    for r in layout["rooms"]:
        node = by_id.get(int(r["id"]), {})
        src = r.get("source_polygons") or []
        counts[r["unified_type"]] = counts.get(r["unified_type"], 0) + 1
        fine[r["room_type_name"]] += 1
        rooms.append({"unified": r["unified_type"], "fine": r["room_type_name"],
                      "px2": float(r.get("polygon_area_ratio", r["area_ratio"])) * width * width,
                      # wall-free room area (ResPlan original polygon); None for RPLAN
                      "src_px2": float(sum(polygon_area(q) for q in src)) if src else None,
                      "area_m2": None, "aspect": round(_aspect(r["bbox"]), 4),
                      "reachable": int(node.get("door_hops", -1)) >= 0})

    edges = room_edges(graph)
    pairs, neighbours = set(), {i: set() for i in by_id}
    for e in edges:
        u, v = int(e["source"]), int(e["target"])
        if u in by_id and v in by_id and u != v:
            pairs.add(tuple(sorted((by_id[u]["unified_type"], by_id[v]["unified_type"]))))
            neighbours[u].add(v)
            neighbours[v].add(u)
    n = len(rooms)
    living = [i for i, nd in by_id.items() if nd["unified_type"] == "living"]
    living_share = (max(len(neighbours[i]) for i in living) / max(1, n - 1)) if living else None

    fd = graph.get("front_door") or {}
    entrance = None
    if fd.get("entrance_room_id") is not None and int(fd["entrance_room_id"]) in by_id:
        entrance = by_id[int(fd["entrance_room_id"])]["unified_type"]
    reach = (sum(r["reachable"] for r in rooms) / max(1, n)) if fd else None
    cs = layout["coordinate_system"]
    raw = layout.get("raw") or {}
    return {
        "id": sid, "dataset": layout.get("dataset") or dataset_of(sid), "split": split,
        "n_rooms": n, "counts": counts, "fine_counts": dict(fine), "rooms": rooms,
        "outline_px2": float(layout.get("boundary_area_ratio", 0.0)) * width * width,
        "units_per_px": cs.get("source_units_per_px"),
        "dataset_net_m2": _num(raw.get("net_area")), "dataset_gross_m2": _num(raw.get("area")),
        "net_area_m2": None, "gross_area_m2": None, "area_source": None,
        "entrance_type": entrance, "reachable_share": reach, "living_neighbour_share": living_share,
        "adjacent_pairs": sorted([list(p) for p in pairs]),
    }


def _num(value: Any) -> Optional[float]:
    """Finite float or None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


# ----------------------------------------------------------------------------
# Metric areas and RPLAN calibration
# ----------------------------------------------------------------------------
def _set_areas(rec: Dict[str, Any], room_m2: List[Optional[float]], gross: Optional[float], source: Optional[str]) -> None:
    """Write per-room, net and gross areas (m²) and their source into a record."""
    for room, a in zip(rec["rooms"], room_m2):
        room["area_m2"] = None if a is None else round(float(a), 3)
    known = [a for a in room_m2 if a is not None]
    rec["net_area_m2"] = round(float(sum(known)), 3) if known and len(known) == len(room_m2) else None
    rec["gross_area_m2"] = None if gross is None else round(float(gross), 3)
    rec["area_source"] = source if rec["net_area_m2"] is not None else None


def resplan_metric_areas(rec: Dict[str, Any], resplan_units: str = "m") -> None:
    """ResPlan areas from the dataset's ``area`` (see module docstring), else geometry in metres."""
    room_px2 = [r["src_px2"] if r.get("src_px2") else r["px2"] for r in rec["rooms"]]
    gross, net = rec.get("dataset_gross_m2"), rec.get("dataset_net_m2")
    if gross and rec.get("outline_px2"):
        k = gross / rec["outline_px2"]                       # m² per canvas px² of this plan
        _set_areas(rec, [a * k for a in room_px2], gross, "dataset")
    elif net and net < 5000:                                 # no gross area: distribute a plausible net_area
        total = sum(room_px2) or 1.0
        _set_areas(rec, [net * a / total for a in room_px2], None, "dataset")
    elif rec.get("units_per_px") and resplan_units == "m":
        k = float(rec["units_per_px"]) ** 2
        _set_areas(rec, [a * k for a in room_px2], rec["outline_px2"] * k, "geometry_metric")
    else:
        _set_areas(rec, [None] * len(rec["rooms"]), None, None)


def units_cross_check(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Are the ResPlan coordinates metres? Compares the geometry (source units²) with the dataset's ``area``.

    Returns the median ratio outline-units² / area (1 = metres), its spread across
    plans (a large spread means coordinates are normalised per plan) and the
    share of plans whose ``net_area`` is consistent with ``area`` (0.6-1.0 x area).
    """
    ratios, net_ok, n = [], 0, 0
    for rec in records:
        if rec["dataset"] != "resplan" or not rec.get("dataset_gross_m2"):
            continue
        n += 1
        net = rec.get("dataset_net_m2") or 0.0
        net_ok += int(0.6 * rec["dataset_gross_m2"] <= net <= 1.0 * rec["dataset_gross_m2"])
        if rec.get("units_per_px"):
            ratios.append(rec["outline_px2"] * float(rec["units_per_px"]) ** 2 / rec["dataset_gross_m2"])
    if not ratios:
        return None
    med = float(np.median(ratios))
    cv = float(np.std(ratios) / np.mean(ratios))
    return {"median_ratio": med, "spread_cv": cv, "n_plans": n, "metres": 0.8 <= med <= 1.25 and cv < 0.2,
            "normalised_per_plan": cv >= 0.2, "net_area_consistent_share": net_ok / max(1, n),
            "implied_metres_per_unit": float(1.0 / math.sqrt(med)) if med > 0 else None}


def calibrate_rplan(resplan_rooms: List[Tuple[str, float]], rplan_rooms: List[Tuple[str, float]],
                    types: List[str], manual: Optional[float] = None,
                    n_plans: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
    """m² per canvas px² for RPLAN.

    ``resplan_rooms`` = ``[(unified type, m²)]``, ``rplan_rooms`` =
    ``[(unified type, px²)]``. Returns ``{scale, per_type_ratios, types, n_plans,
    method}`` or None when it cannot be estimated (no ResPlan data, no manual value).
    """
    if manual is not None:
        return {"scale": float(manual), "per_type_ratios": {}, "types": list(types), "n_plans": n_plans or {},
                "method": "manual"}
    ratios = {}
    for t in types:
        a_res = [a for tt, a in resplan_rooms if tt == t and a]
        a_rp = [a for tt, a in rplan_rooms if tt == t and a]
        if a_res and a_rp:
            ratios[t] = float(np.median(a_res)) / float(np.median(a_rp))
    if not ratios:
        return None
    return {"scale": float(np.median(list(ratios.values()))), "per_type_ratios": ratios, "types": list(types),
            "n_plans": n_plans or {}, "method": "median room size per type (ResPlan m² / RPLAN px²)"}


def rplan_calibrated_areas(rec: Dict[str, Any], scale: Optional[float]) -> None:
    """RPLAN areas = px² x scale (``area_source = "calibrated"``), or unknown without a scale."""
    if scale is None:
        _set_areas(rec, [None] * len(rec["rooms"]), None, None)
    else:
        _set_areas(rec, [r["px2"] * scale for r in rec["rooms"]], rec["outline_px2"] * scale, "calibrated")


# ----------------------------------------------------------------------------
# Priors for the plausibility score
# ----------------------------------------------------------------------------
def fit_priors(records: List[Dict[str, Any]], unified_names: List[str], min_samples: int = 20) -> Dict[str, Any]:
    """Per unified type: mean / std of log(area m²) and of log(aspect); types with few rooms use all rooms."""
    def stats(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
        """Mean and (floored) std of a list of logs."""
        if not values:
            return None, None
        return float(np.mean(values)), max(float(np.std(values)), 0.05)

    all_a = [math.log(r["area_m2"]) for rec in records for r in rec["rooms"] if r["area_m2"]]
    all_r = [math.log(r["aspect"]) for rec in records for r in rec["rooms"]]
    out: Dict[str, Any] = {"all": {}, "types": {}, "min_samples": min_samples}
    out["all"]["mu_logA"], out["all"]["sd_logA"] = stats(all_a)
    out["all"]["mu_logR"], out["all"]["sd_logR"] = stats(all_r)
    for t in unified_names:
        a = [math.log(r["area_m2"]) for rec in records for r in rec["rooms"] if r["unified"] == t and r["area_m2"]]
        ra = [math.log(r["aspect"]) for rec in records for r in rec["rooms"] if r["unified"] == t]
        entry = {"n_area": len(a), "n_shape": len(ra)}
        entry["mu_logA"], entry["sd_logA"] = stats(a) if len(a) >= min_samples else stats(all_a)
        entry["mu_logR"], entry["sd_logR"] = stats(ra) if len(ra) >= min_samples else stats(all_r)
        entry["fallback_area"], entry["fallback_shape"] = len(a) < min_samples, len(ra) < min_samples
        out["types"][t] = entry
    return out


# ----------------------------------------------------------------------------
# Build / load
# ----------------------------------------------------------------------------
def _ids_hash(ids: List[str]) -> str:
    """Hash of the sorted graph ids (cache key)."""
    return hashlib.md5("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _base_records(ws: Any, unified_names: List[str], desc: str) -> List[Dict[str, Any]]:
    """Catalog records (without m²) for every graph of a workspace."""
    from prism.datasets.store import LayoutStore

    store = LayoutStore(ws)
    ids = store.graph_ids()
    splits: Dict[str, str] = {}
    if ws.splits_file.exists():
        for name, members in read_json(ws.splits_file).items():
            for sid in members:
                splits[str(sid)] = name
    return [plan_record(sid, store.layout(sid), store.graph(sid), splits.get(sid, "unknown"), unified_names)
            for sid in progress(ids, desc)]


def resplan_workspace(ws: Any) -> Optional[Any]:
    """ResPlan workspace of the same mode (member of ``combined``, or built for an RPLAN-only run)."""
    from prism.common.workspace import Workspace

    if ws.dataset == "resplan":
        return ws
    if ws.is_combined and "resplan" in ws.member_datasets:
        return ws.member("resplan")
    cfg = copy.copy(ws.cfg)
    cfg["dataset"] = "resplan"
    cfg["paths"] = {k: v for k, v in ws.cfg["paths"].items() if k != "processed_override"}
    return Workspace(cfg)


def load_catalog(ws: Any, cfg: Dict[str, Any], unified_names: List[str],
                 rebuild: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Catalog of a workspace with m² filled in, plus ``info`` (calibration, priors, cross-check, cache status).

    Base records are cached in ``artifacts/.../constraints/catalog.jsonl`` and
    rebuilt when the graph ids change (or with ``rebuild``). Areas, calibration
    and priors are recomputed on every call (cheap), so config changes such as
    ``constraints.rplan_m2_per_px2`` take effect without a rebuild.
    """
    from prism.datasets.store import LayoutStore

    ccfg = cfg.get("constraints", {})
    out_dir = ws.artifacts / "constraints"
    cat_file, meta_file = out_dir / "catalog.jsonl", out_dir / "catalog_meta.json"
    ids = LayoutStore(ws).graph_ids()
    if not ids:
        raise RuntimeError(f"No graphs in workspace '{ws.dataset}' ({ws.mode}). Run phases 02-05 first.")
    key = {"version": CATALOG_VERSION, "ids_hash": _ids_hash(ids), "n": len(ids)}
    cached = (not rebuild and cat_file.exists() and meta_file.exists()
              and {k: read_json(meta_file).get(k) for k in key} == key)
    if cached:
        records = read_jsonl(cat_file)
    else:
        records = _base_records(ws, unified_names, "catalog")

    # ResPlan areas (dataset m² first), then the RPLAN calibration from ResPlan room sizes.
    units = str(ccfg.get("resplan_units", "m"))
    for rec in records:
        if rec["dataset"] == "resplan":
            resplan_metric_areas(rec, units)
    resplan_recs = [r for r in records if r["dataset"] == "resplan"]
    source_note = "catalog"
    if not resplan_recs and any(r["dataset"] == "rplan" for r in records) and ccfg.get("rplan_m2_per_px2") is None:
        rws = resplan_workspace(ws)
        if rws is not None and LayoutStore(rws).graph_ids():
            resplan_recs = _base_records(rws, unified_names, "ResPlan sizes for calibration")
            for rec in resplan_recs:
                resplan_metric_areas(rec, units)
            source_note = f"ResPlan workspace {rws.processed}"
        else:
            source_note = "no ResPlan data"
    types = list(ccfg.get("calibration_types", ["bedroom", "bathroom", "kitchen"]))
    rplan_recs = [r for r in records if r["dataset"] == "rplan"]
    calibration = None
    if rplan_recs:
        calibration = calibrate_rplan(
            [(r["unified"], r["area_m2"]) for rec in resplan_recs for r in rec["rooms"]],
            [(r["unified"], r["px2"]) for rec in rplan_recs for r in rec["rooms"]],
            types, ccfg.get("rplan_m2_per_px2"),
            {"resplan": len(resplan_recs), "rplan": len(rplan_recs)})
        if calibration is not None:
            calibration["resplan_source"] = source_note
        for rec in rplan_recs:
            rplan_calibrated_areas(rec, calibration["scale"] if calibration else None)

    priors = fit_priors(records, unified_names, int(ccfg.get("min_type_samples", 20)))
    cross = units_cross_check(resplan_recs)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cat_file, records)
    write_json(meta_file, {**key, "dataset": ws.dataset, "mode": ws.mode})
    write_json(out_dir / "calibration.json", calibration or {"scale": None, "method": "unavailable",
                                                            "reason": source_note})
    write_json(out_dir / "priors.json", priors)
    info = {"calibration": calibration, "priors": priors, "units_cross_check": cross, "cached": cached,
            "n_plans": len(records), "artifacts": str(out_dir), "calibration_source": source_note}
    return records, info
