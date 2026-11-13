"""Unit tests for the constraint search (phase 13).

Run with:  python -m pytest tests -q
"""

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from prism.common.config import load_config  # noqa: E402
from prism.constraints.catalog import calibrate_rplan, fit_priors  # noqa: E402
from prism.constraints.scoring import evaluate_requirement, interval_satisfaction, score_record, target_closeness  # noqa: E402
from prism.constraints.search import mmr, run_search  # noqa: E402
from prism.constraints.spec import SpecError, normalize_spec, parse_count, requirements  # noqa: E402
from prism.datasets.schema import Taxonomy  # noqa: E402


def _cfg_tax():
    """Default config and the combined (unified) taxonomy."""
    cfg = load_config("local")
    return cfg, Taxonomy(cfg, "combined")


def make_record(sid, dataset="resplan", bedrooms=2, bathrooms=1, area=70.0, source="dataset"):
    """In-memory catalog record with a living room, a kitchen and the given bedrooms / bathrooms."""
    _, tax = _cfg_tax()
    rooms = ([{"unified": "living", "fine": "living", "area_m2": 25.0, "aspect": 1.3}]
             + [{"unified": "kitchen", "fine": "kitchen", "area_m2": 8.0, "aspect": 1.4}]
             + [{"unified": "bedroom", "fine": "bedroom", "area_m2": 14.0, "aspect": 1.2}] * bedrooms
             + [{"unified": "bathroom", "fine": "bathroom", "area_m2": 5.0, "aspect": 1.5}] * bathrooms)
    if area is None:
        rooms = [{**r, "area_m2": None} for r in rooms]
    counts = {t: 0 for t in tax.unified_names}
    for r in rooms:
        counts[r["unified"]] += 1
    return {"id": sid, "dataset": dataset, "split": "test", "n_rooms": len(rooms), "counts": counts,
            "fine_counts": {}, "rooms": rooms, "net_area_m2": area, "gross_area_m2": area,
            "area_source": source if area is not None else None, "entrance_type": "living",
            "reachable_share": 1.0, "living_neighbour_share": 0.8,
            "adjacent_pairs": [["kitchen", "living"], ["bedroom", "living"]]}


def test_shorthand_parsing_and_validation():
    """Shorthands give the full form; unknown types and min > max are rejected with a clear message."""
    assert parse_count(2) == {"min": 2, "max": 2}
    assert parse_count("2+") == {"min": 2, "max": None}
    assert parse_count("2-3") == {"min": 2, "max": 3}
    assert parse_count("<=1") == {"min": None, "max": 1}
    assert parse_count(0) == {"min": 0, "max": 0}
    cfg, tax = _cfg_tax()
    with pytest.raises(SpecError, match="Valid types"):
        normalize_spec({"room_types": {"sauna": 1}}, cfg, tax)
    with pytest.raises(SpecError, match="min"):
        normalize_spec({"room_types": {"bedroom": {"min": 3, "max": 2}}}, cfg, tax)
    with pytest.raises(SpecError, match="contradictory"):
        normalize_spec({"rooms": {"max": 3, "hard": True}, "room_types": {"bedroom": "2+", "bathroom": "2+"}}, cfg, tax)
    spec = normalize_spec({"area_m2": {"target": 80}}, cfg, tax)
    assert spec["area_m2"]["tolerance_m2"] == pytest.approx(8.0)


def test_satisfaction_functions():
    """interval_satisfaction is 1 inside and e^-1 at distance = scale; target_closeness is 1 at the target."""
    assert interval_satisfaction(5, 4, 6) == 1.0
    assert interval_satisfaction(7, 4, 6, scale=1.0) == pytest.approx(math.exp(-1))
    assert interval_satisfaction(2, lo=4, scale=2.0) == pytest.approx(math.exp(-1))
    assert target_closeness(75, 75, 7.5) == 1.0
    assert interval_satisfaction(2, None, 0) == pytest.approx(math.exp(-2))      # "must not be present", x = 2


def test_hard_filter_and_near_misses():
    """A plan with 3 bedrooms fails 'exactly 2 (hard)' and appears only among the near misses."""
    cfg, tax = _cfg_tax()
    records = [make_record("resplan_1", bedrooms=2), make_record("resplan_2", bedrooms=3)]
    spec = normalize_spec({"k": 5, "room_types": {"bedroom": {"min": 2, "max": 2, "hard": True}},
                           "ranking": {"diversity": 0}}, cfg, tax)
    res = run_search(records, spec, fit_priors(records, tax.unified_names, 1), cfg)
    assert [s["id"] for s in res["results"]] == ["resplan_1"]
    assert [s["id"] for s in res["near_misses"]] == ["resplan_2"]
    assert res["near_misses"][0]["violated"] == ["type:bedroom"]


def test_unknown_area():
    """Unknown area: a soft requirement leaves the average; a hard one keeps the plan unless require_known_area."""
    cfg, tax = _cfg_tax()
    rec = make_record("rplan_1", dataset="rplan", area=None)
    pri = fit_priors([make_record("resplan_9")], tax.unified_names, 1)
    soft = requirements(normalize_spec({"area_m2": {"min": 60, "max": 90}, "room_types": {"bedroom": 2}}, cfg, tax))
    s = score_record(rec, soft, pri, {"size": 1.0}, 0.0)
    assert s["explanation"][0]["status"] == "unknown" and s["explanation"][0]["score"] is None
    assert s["constraint_score"] == pytest.approx(1.0)             # only the (met) bedroom requirement counts
    hard = requirements(normalize_spec({"area_m2": {"min": 60, "max": 90, "hard": True}}, cfg, tax))
    assert score_record(rec, hard, pri, {"size": 1.0}, 0.0)["passes_hard"]
    strict = requirements(normalize_spec({"area_m2": {"min": 60, "max": 90, "hard": True,
                                                      "require_known_area": True}}, cfg, tax))
    assert not score_record(rec, strict, pri, {"size": 1.0}, 0.0)["passes_hard"]
    assert evaluate_requirement(make_record("resplan_3", area=100.0), hard[0])["status"] == "violated"


def test_calibration_scale():
    """ResPlan bedrooms of 12 m² and RPLAN bedrooms of 2400 px² give 0.005 m²/px²."""
    cal = calibrate_rplan([("bedroom", 12.0)] * 5, [("bedroom", 2400.0)] * 7, ["bedroom", "bathroom"])
    assert cal["scale"] == pytest.approx(0.005)
    assert set(cal["per_type_ratios"]) == {"bedroom"}
    assert calibrate_rplan([], [("bedroom", 2400.0)], ["bedroom"]) is None
    assert calibrate_rplan([], [], ["bedroom"], manual=0.05)["method"] == "manual"


def test_mmr_prefers_diversity():
    """With two identical vectors and one different, lambda 0.5 picks the different one second."""
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert mmr([1.0, 0.99, 0.5], v, 3, 0.5)[:2] == [0, 2]
    assert mmr([1.0, 0.99, 0.5], v, 2, 0.0) == [0, 1]


def test_end_to_end_synthetic(tmp_path):
    """Synthetic data -> phases 02-05 -> phase 13: ranked results with explanations and the output files.

    Diversity is switched off so that the order must be the plain ranking by final score
    (with MMR the order is the pick order, which trades score for variety).
    """
    py = sys.executable
    data = tmp_path / "data root with space"
    sets = ["--set", f"paths.data_root={data}", "--set", f"paths.artifacts_root={tmp_path / 'artifacts'}",
            "--set", f"paths.outputs_root={tmp_path / 'outputs'}"]
    subprocess.run([py, str(ROOT / "scripts/make_synthetic_dataset.py"), "--n", "40", "--data-root", str(data)],
                   check=True, capture_output=True)
    subprocess.run([py, str(ROOT / "scripts/run_all.py"), "--from", "2", "--to", "5", "--mode", "preview",
                    "--n", "30"] + sets, check=True, capture_output=True)
    subprocess.run([py, str(ROOT / "scripts/13_constraint_search.py"), "--mode", "preview", "--rooms", "4-8",
                    "--type", "bedroom=1+", "--k", "6", "--name", "e2e", "--set", "constraints.diversity=0"] + sets,
                   check=True, capture_output=True)
    out = tmp_path / "outputs" / "combined" / "preview" / "13_constraint_search" / "e2e"
    for name in ("spec_normalised.json", "results.json", "results.csv", "near_misses.json", "report.html"):
        assert (out / name).exists(), name
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))["results"]
    assert 0 < len(results) <= 6
    finals = [r["final"] for r in results]
    assert all(0.0 <= f <= 1.0 for f in finals)
    assert finals == sorted(finals, reverse=True)               # diversity off: plain ranking by final score
    assert all(r["explanation"] for r in results)
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
