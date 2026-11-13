"""Requests for the constraint search: parsing, shorthands, defaults and validation.

A request (YAML file or CLI flags, see ``docs/constraint_search.md``) is turned
into a **normalised** dictionary where every default is applied::

    {
      "name": "two_bed_flat", "k": 10, "datasets": ["rplan", "resplan"] | None, "split": "all",
      "area_m2": {"basis": "net", "min": 65, "max": 90, "target": 75, "tolerance_m2": 7.5,
                  "hard": False, "weight": 2.0, "require_known_area": False} | None,
      "rooms": {"min": 4, "max": 6, "hard": True, "weight": 1.0} | None,
      "room_types": {"bedroom": {"min": 2, "max": 2, "hard": True, "weight": 2.0}, ...},
      "entrance": {"type": "living", "hard": False, "weight": 0.5} | None,
      "adjacent": [{"pair": ["kitchen", "living"], "hard": False, "weight": 0.5}, ...],
      "ranking": {"plausibility_weight": 0.3, "diversity": 0.2, "near_misses": 5},
    }

:func:`requirements` flattens it into one record per requirement (``area``,
``rooms``, ``type:<name>``, ``entrance``, ``adjacent:<a>-<b>``), which is what
the scoring code consumes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SPLITS = ("all", "train", "val", "test")
AREA_BASES = ("net", "gross")


class SpecError(ValueError):
    """Invalid request (unknown room type, min > max, contradictory hard requirements ...)."""


# ----------------------------------------------------------------------------
# Shorthands
# ----------------------------------------------------------------------------
def parse_count(value: Any, what: str = "count") -> Dict[str, Optional[int]]:
    """Parse a count shorthand into ``{"min": .., "max": ..}``.

    ``2`` -> exactly 2; ``"2+"`` -> at least 2; ``"2-3"`` -> between 2 and 3;
    ``"<=1"`` -> at most 1; ``">=2"`` -> at least 2; ``0`` -> must not be present.
    """
    if isinstance(value, bool):
        raise SpecError(f"{what}: invalid value {value!r}")
    if isinstance(value, (int, float)):
        n = int(value)
        return {"min": n, "max": n}
    text = str(value).strip().replace(" ", "")
    patterns = [
        (r"^(\d+)$", lambda m: {"min": int(m[1]), "max": int(m[1])}),
        (r"^(\d+)\+$", lambda m: {"min": int(m[1]), "max": None}),
        (r"^>=(\d+)$", lambda m: {"min": int(m[1]), "max": None}),
        (r"^<=(\d+)$", lambda m: {"min": None, "max": int(m[1])}),
        (r"^(\d+)-(\d+)$", lambda m: {"min": int(m[1]), "max": int(m[2])}),
    ]
    for pattern, build in patterns:
        m = re.match(pattern, text)
        if m:
            return build(m)
    raise SpecError(f"{what}: cannot parse {value!r} (use 2, '2+', '2-3', '<=1' or a {{min, max}} mapping)")


def parse_range(value: Any, what: str = "range") -> Dict[str, Optional[float]]:
    """Parse a numeric range shorthand (``"65-90"``, ``"70+"``, ``"<=90"``, ``75``) into ``{"min", "max"}``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"min": float(value), "max": float(value)}
    text = str(value).strip().replace(" ", "")
    num = r"(\d+(?:\.\d+)?)"
    for pattern, build in [
        (rf"^{num}-{num}$", lambda m: {"min": float(m[1]), "max": float(m[2])}),
        (rf"^{num}\+$", lambda m: {"min": float(m[1]), "max": None}),
        (rf"^>={num}$", lambda m: {"min": float(m[1]), "max": None}),
        (rf"^<={num}$", lambda m: {"min": None, "max": float(m[1])}),
        (rf"^{num}$", lambda m: {"min": float(m[1]), "max": float(m[1])}),
    ]:
        m = re.match(pattern, text)
        if m:
            return build(m)
    raise SpecError(f"{what}: cannot parse {value!r} (use 65-90, 70+, <=90 or a number)")


def _count_entry(value: Any, what: str, default_weight: float = 1.0) -> Dict[str, Any]:
    """Full form of a count requirement (``rooms`` or one ``room_types`` entry)."""
    if isinstance(value, dict):
        out = {"min": value.get("min"), "max": value.get("max")}
        if "exactly" in value:
            out = {"min": value["exactly"], "max": value["exactly"]}
        hard, weight = bool(value.get("hard", False)), float(value.get("weight", default_weight))
    else:
        out = parse_count(value, what)
        hard, weight = False, default_weight
    for key in ("min", "max"):
        if out[key] is not None:
            if float(out[key]) != int(out[key]):
                raise SpecError(f"{what}: {key} must be an integer, got {out[key]!r}")
            out[key] = int(out[key])
            if out[key] < 0:
                raise SpecError(f"{what}: {key} cannot be negative ({out[key]})")
    if out["min"] is not None and out["max"] is not None and out["min"] > out["max"]:
        raise SpecError(f"{what}: min ({out['min']}) > max ({out['max']})")
    if weight < 0:
        raise SpecError(f"{what}: weight cannot be negative")
    return {**out, "hard": hard, "weight": weight}


# ----------------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------------
def valid_type_names(taxonomy: Any) -> List[str]:
    """Room-type names accepted in a request: unified types, plus fine names when the taxonomy level is fine."""
    names = list(taxonomy.unified_names)
    if getattr(taxonomy, "level", "unified") == "fine":
        names += [n for n in taxonomy.vocab() if n not in names]
    return names


def _area_entry(value: Dict[str, Any], tolerance_ratio: float) -> Dict[str, Any]:
    """Full form of the ``area_m2`` requirement (default tolerance from ``tolerance_ratio``)."""
    if not isinstance(value, dict):
        value = {**parse_range(value, "area_m2")}
    lo, hi, target = value.get("min"), value.get("max"), value.get("target")
    lo = float(lo) if lo is not None else None
    hi = float(hi) if hi is not None else None
    target = float(target) if target is not None else None
    if lo is None and hi is None and target is None:
        raise SpecError("area_m2: give at least one of target, min, max")
    for key, v in (("min", lo), ("max", hi), ("target", target)):
        if v is not None and v <= 0:
            raise SpecError(f"area_m2: {key} must be positive ({v})")
    if lo is not None and hi is not None and lo > hi:
        raise SpecError(f"area_m2: min ({lo}) > max ({hi})")
    basis = str(value.get("basis", "net"))
    if basis not in AREA_BASES:
        raise SpecError(f"area_m2: basis must be one of {AREA_BASES}, got {basis!r}")
    tol = value.get("tolerance_m2")
    if tol is None:
        if target is not None:
            ref = target
        elif lo is not None and hi is not None:
            ref = (lo + hi) / 2.0
        else:
            ref = lo if lo is not None else hi
        tol = tolerance_ratio * ref
    tol = float(tol)
    if tol <= 0:
        raise SpecError("area_m2: tolerance_m2 must be positive")
    return {"basis": basis, "min": lo, "max": hi, "target": target, "tolerance_m2": tol,
            "hard": bool(value.get("hard", False)), "weight": float(value.get("weight", 1.0)),
            "require_known_area": bool(value.get("require_known_area", False))}


def normalize_spec(raw: Dict[str, Any], cfg: Dict[str, Any], taxonomy: Any,
                   available_datasets: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Apply shorthands and defaults to a raw request and validate it (raises :class:`SpecError`)."""
    ccfg = cfg.get("constraints", {})
    raw = copy.deepcopy(raw or {})
    out: Dict[str, Any] = {}
    out["name"] = str(raw.get("name") or f"cli_{spec_hash(raw)}")
    if not re.match(r"^[\w.\-]+$", out["name"]):
        raise SpecError(f"name {out['name']!r}: use letters, digits, '_', '-' or '.' only (it is a folder name)")
    out["k"] = int(raw.get("k", ccfg.get("k", 10)))
    if out["k"] < 1:
        raise SpecError(f"k must be >= 1, got {out['k']}")

    datasets = raw.get("datasets")
    if datasets is not None:
        datasets = [str(d) for d in (datasets if isinstance(datasets, (list, tuple)) else [datasets])]
        if available_datasets is not None:
            unknown = [d for d in datasets if d not in set(available_datasets)]
            if unknown:
                raise SpecError(f"datasets {unknown} are not part of this workspace ({list(available_datasets)})")
    out["datasets"] = datasets
    out["split"] = str(raw.get("split", "all"))
    if out["split"] not in SPLITS:
        raise SpecError(f"split must be one of {SPLITS}, got {out['split']!r}")

    out["area_m2"] = _area_entry(raw["area_m2"], float(ccfg.get("area_tolerance_ratio", 0.10))) \
        if raw.get("area_m2") is not None else None
    out["rooms"] = _count_entry(raw["rooms"], "rooms") if raw.get("rooms") is not None else None

    valid = valid_type_names(taxonomy)
    types: Dict[str, Any] = {}
    for name, value in (raw.get("room_types") or {}).items():
        name = str(name)
        if name not in valid:
            level = getattr(taxonomy, "level", "unified")
            raise SpecError(f"unknown room type {name!r} for taxonomy level {level!r}. Valid types: {', '.join(valid)}")
        types[name] = _count_entry(value, f"room_types.{name}")
    out["room_types"] = types

    ent = raw.get("entrance")
    if ent is not None:
        if not isinstance(ent, dict):
            ent = {"type": ent}
        if str(ent.get("type")) not in valid:
            raise SpecError(f"entrance: unknown room type {ent.get('type')!r}. Valid types: {', '.join(valid)}")
        out["entrance"] = {"type": str(ent["type"]), "hard": bool(ent.get("hard", False)),
                           "weight": float(ent.get("weight", 1.0))}
    else:
        out["entrance"] = None

    adjacent = []
    for item in raw.get("adjacent") or []:
        if not isinstance(item, dict):
            item = {"pair": item}
        pair = item.get("pair")
        if isinstance(pair, str):
            pair = re.split(r"[-,]", pair)
        if not pair or len(pair) != 2:
            raise SpecError(f"adjacent: a pair needs two room types, got {pair!r}")
        pair = [str(p).strip() for p in pair]
        bad = [p for p in pair if p not in taxonomy.unified_names]
        if bad:
            raise SpecError(f"adjacent: unknown unified room type(s) {bad}. Valid types: "
                            f"{', '.join(taxonomy.unified_names)}")
        adjacent.append({"pair": sorted(pair), "hard": bool(item.get("hard", False)),
                         "weight": float(item.get("weight", 1.0))})
    out["adjacent"] = adjacent

    rk = raw.get("ranking") or {}
    out["ranking"] = {
        "plausibility_weight": float(rk.get("plausibility_weight", ccfg.get("plausibility_weight", 0.3))),
        "diversity": float(rk.get("diversity", ccfg.get("diversity", 0.2))),
        "near_misses": int(rk.get("near_misses", ccfg.get("near_misses", 5))),
    }
    if not 0.0 <= out["ranking"]["plausibility_weight"] <= 1.0:
        raise SpecError("ranking.plausibility_weight must be in [0, 1]")
    if not 0.0 <= out["ranking"]["diversity"] < 1.0:
        raise SpecError("ranking.diversity must be in [0, 1)")

    # Contradiction: the hard room total cannot hold the minimum number of rooms requested per type.
    if out["rooms"] and out["rooms"]["hard"] and out["rooms"]["max"] is not None:
        need = sum(v["min"] or 0 for v in types.values())
        if need > out["rooms"]["max"]:
            raise SpecError(f"contradictory request: the room-type minimums add up to {need} rooms, "
                            f"but rooms.max is {out['rooms']['max']} (hard)")
    return out


# ----------------------------------------------------------------------------
# Sources: YAML file, CLI flags
# ----------------------------------------------------------------------------
def load_spec_file(path: Path) -> Dict[str, Any]:
    """Read a request from a YAML file."""
    import yaml

    if not path.exists():
        raise SpecError(f"request file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SpecError(f"{path}: the request must be a YAML mapping")
    return data


def apply_cli(raw: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Override / extend a raw request with the phase-13 CLI flags (see ``scripts/13_constraint_search.py``)."""
    raw = copy.deepcopy(raw or {})
    if getattr(args, "k", None):
        raw["k"] = int(args.k)
    if getattr(args, "name", None):
        raw["name"] = args.name
    if getattr(args, "area", None) or getattr(args, "area_target", None) is not None or getattr(args, "area_basis", None):
        area = dict(raw.get("area_m2") or {}) if isinstance(raw.get("area_m2"), dict) else {}
        if args.area:
            area.update(parse_range(args.area, "--area"))
        if args.area_target is not None:
            area["target"] = float(args.area_target)
        if args.area_basis:
            area["basis"] = args.area_basis
        raw["area_m2"] = area
    if getattr(args, "rooms", None):
        rooms = dict(raw.get("rooms") or {}) if isinstance(raw.get("rooms"), dict) else {}
        rooms.update(parse_count(args.rooms, "--rooms"))
        raw["rooms"] = rooms
    for item in getattr(args, "type", None) or []:
        if "=" not in item:
            raise SpecError(f"--type {item!r}: use name=value, e.g. bedroom=2 or bathroom=1+")
        name, value = item.split("=", 1)
        types = raw.setdefault("room_types", {}) or {}
        prev = types.get(name.strip())
        entry = dict(prev) if isinstance(prev, dict) else {}
        entry.update(parse_count(value, f"--type {name}"))
        types[name.strip()] = entry
        raw["room_types"] = types
    if getattr(args, "entrance", None):
        ent = dict(raw.get("entrance") or {}) if isinstance(raw.get("entrance"), dict) else {}
        ent["type"] = args.entrance
        raw["entrance"] = ent
    for pair in getattr(args, "adjacent", None) or []:
        raw.setdefault("adjacent", [])
        raw["adjacent"] = list(raw["adjacent"] or []) + [{"pair": pair}]
    for name in getattr(args, "hard", None) or []:
        _mark_hard(raw, name)
    return raw


def _mark_hard(raw: Dict[str, Any], name: str) -> None:
    """Set ``hard: true`` on the requirement named ``area``, ``rooms``, ``type:<t>``, ``entrance`` or ``adjacent:a-b``."""
    def as_dict(value: Any, parser) -> Dict[str, Any]:
        """Full mapping form of a shorthand value."""
        return dict(value) if isinstance(value, dict) else dict(parser(value))

    if name == "area":
        if raw.get("area_m2") is None:
            raise SpecError("--hard area: no area requirement given")
        raw["area_m2"] = {**as_dict(raw["area_m2"], parse_range), "hard": True}
    elif name == "rooms":
        if raw.get("rooms") is None:
            raise SpecError("--hard rooms: no rooms requirement given")
        raw["rooms"] = {**as_dict(raw["rooms"], parse_count), "hard": True}
    elif name.startswith("type:"):
        t = name.split(":", 1)[1]
        types = raw.get("room_types") or {}
        if t not in types:
            raise SpecError(f"--hard {name}: no requirement for room type {t!r}")
        types[t] = {**as_dict(types[t], parse_count), "hard": True}
    elif name == "entrance":
        if raw.get("entrance") is None:
            raise SpecError("--hard entrance: no entrance requirement given")
        ent = raw["entrance"] if isinstance(raw["entrance"], dict) else {"type": raw["entrance"]}
        raw["entrance"] = {**ent, "hard": True}
    elif name.startswith("adjacent:"):
        want = sorted(re.split(r"[-,]", name.split(":", 1)[1]))
        hit = False
        new = []
        for item in raw.get("adjacent") or []:
            item = item if isinstance(item, dict) else {"pair": item}
            pair = item["pair"] if not isinstance(item["pair"], str) else re.split(r"[-,]", item["pair"])
            if sorted(str(p).strip() for p in pair) == want:
                item = {**item, "hard": True}
                hit = True
            new.append(item)
        if not hit:
            raise SpecError(f"--hard {name}: no adjacency requirement for {want}")
        raw["adjacent"] = new
    else:
        raise SpecError(f"--hard {name!r}: use area, rooms, type:<name>, entrance or adjacent:<a>-<b>")


def spec_hash(raw: Dict[str, Any]) -> str:
    """Short stable hash of a raw request (default output name for CLI-only requests)."""
    clean = {k: v for k, v in (raw or {}).items() if k != "name"}
    return hashlib.md5(json.dumps(clean, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]


# ----------------------------------------------------------------------------
# Flattened requirements
# ----------------------------------------------------------------------------
def describe_interval(lo: Any, hi: Any, unit: str = "") -> str:
    """Human-readable interval: ``2``, ``2+``, ``<=1``, ``2-3`` (with an optional unit)."""
    def fmt(v):
        """Compact number."""
        return f"{v:g}" if isinstance(v, float) else str(v)
    if lo is not None and hi is not None:
        text = fmt(lo) if lo == hi else f"{fmt(lo)}-{fmt(hi)}"
    elif lo is not None:
        text = f"{fmt(lo)}+"
    elif hi is not None:
        text = f"<={fmt(hi)}"
    else:
        text = "any"
    return text + (f" {unit}" if unit and text != "any" else "")


def requirements(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a normalised request into one record per requirement (in a stable order)."""
    reqs: List[Dict[str, Any]] = []
    if spec.get("area_m2"):
        a = spec["area_m2"]
        wanted = describe_interval(a["min"], a["max"], "m²")
        if a["target"] is not None:
            wanted = (f"~{a['target']:g} m²" if wanted == "any" else f"{wanted}, target {a['target']:g}")
        reqs.append({"name": "area", "kind": "area", "wanted": f"{wanted} ({a['basis']})", **a})
    if spec.get("rooms"):
        r = spec["rooms"]
        reqs.append({"name": "rooms", "kind": "count", "field": "n_rooms",
                     "wanted": describe_interval(r["min"], r["max"]), **r})
    for t, v in (spec.get("room_types") or {}).items():
        wanted = "none" if v["max"] == 0 and not v["min"] else describe_interval(v["min"], v["max"])
        reqs.append({"name": f"type:{t}", "kind": "count", "field": "type", "type": t, "wanted": wanted, **v})
    if spec.get("entrance"):
        e = spec["entrance"]
        reqs.append({"name": "entrance", "kind": "entrance", "wanted": e["type"], **e})
    for item in spec.get("adjacent") or []:
        a, b = item["pair"]
        reqs.append({"name": f"adjacent:{a}-{b}", "kind": "adjacent", "wanted": f"{a} next to {b}", **item})
    return reqs
