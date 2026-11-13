"""Phase 13 - constraint search: the k most plausible floor plans for a user's requirements.

Requirements: floor area in m² (target and/or range, net or gross), number of
rooms, room types wanted / excluded with counts, the room behind the front door
and adjacency wishes ("kitchen next to living"). Each requirement is soft
(scored) or hard (filter). Results are ranked by

    final = (1 - plausibility_weight) * constraint_score + plausibility_weight * plausibility

and diversified with MMR on the graph-descriptor vectors. Every result comes
with one explanation row per requirement. RPLAN areas are ESTIMATES calibrated
on ResPlan room sizes and are shown as "≈". See docs/constraint_search.md.

    python scripts/13_constraint_search.py --spec configs/specs/example_2bed.yaml
    python scripts/13_constraint_search.py --area 65-90 --area-target 75 --rooms 4-6 \\
        --type bedroom=2 --type bathroom=1+ --type kitchen=1 --type study=0 --k 10
    python scripts/13_constraint_search.py --spec configs/specs/example_2bed.yaml --set constraints.rplan_m2_per_px2=0.05

CPU only (no torch / FAISS needed). Outputs: outputs/<dataset>/<mode>/13_constraint_search/<name>/.
"""

import sys
from collections import Counter
from pathlib import Path


# Make `import prism` work however the script is launched (python scripts/x.py,
# VS Code "Run Python File", python -m scripts.x) even without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.common.cli import Timer, build_parser, setup
from prism.common.config import resolve_path
from prism.common.io import write_csv, write_json
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import grouped_bars, image_grid, plt
from prism.constraints.catalog import load_catalog
from prism.constraints.search import descriptor_vectors, run_search
from prism.constraints.spec import SpecError, apply_cli, load_spec_file, normalize_spec
from prism.datasets.schema import Taxonomy
from prism.datasets.store import LayoutStore

PHASE = "13_constraint_search"
CLI_REQUIREMENT_FLAGS = ("area", "area_target", "rooms", "type", "entrance", "adjacent")


def build_cli():
    """Shared phase flags plus the request flags of phase 13."""
    parser = build_parser(__doc__)
    parser.add_argument("--spec", default=None, help="YAML request (see configs/specs/).")
    parser.add_argument("--k", type=int, default=None, help="Number of plans to return.")
    parser.add_argument("--area", default=None, help="Area range in m², e.g. 65-90, 70+, <=90.")
    parser.add_argument("--area-target", type=float, default=None, help="Target area in m².")
    parser.add_argument("--area-basis", choices=["net", "gross"], default=None, help="net = sum of rooms, gross = outline.")
    parser.add_argument("--rooms", default=None, help="Number of rooms: 5, 4-6, 4+, <=6.")
    parser.add_argument("--type", action="append", default=[], help="Room type count, e.g. bedroom=2, bathroom=1+, study=0.")
    parser.add_argument("--entrance", default=None, help="Unified type of the room behind the front door.")
    parser.add_argument("--adjacent", action="append", default=[], help="Pair that should be adjacent, e.g. kitchen-living.")
    parser.add_argument("--hard", action="append", default=[],
                        help="Make a requirement hard: area, rooms, type:<name>, entrance, adjacent:<a>-<b>.")
    parser.add_argument("--name", default=None, help="Output folder name (default: spec name or cli_<hash>).")
    parser.add_argument("--rebuild-catalog", action="store_true", help="Rebuild the cached plan catalog.")
    return parser


def area_text(area, source):
    """'≈ 74 m²' for calibrated areas, '74 m²' for dataset / geometric ones, '? m²' when unknown."""
    if area is None:
        return "? m²"
    return f"{'≈ ' if source == 'calibrated' else ''}{area:.0f} m²"


SHORT = {"bedroom": "bed", "bathroom": "bath", "kitchen": "kit", "living": "liv", "balcony": "balc",
         "storage": "stor", "entrance": "entr", "dining": "din"}


def caption_counts(counts, types, per_line=3):
    """Compact counts for image captions, e.g. '2 bed · 1 bath · 1 kit' wrapped every ``per_line`` types."""
    items = [f"{counts.get(t, 0)} {SHORT.get(t, t)}" for t in types]
    return "\n".join(" · ".join(items[i:i + per_line]) for i in range(0, len(items), per_line))


def counts_text(counts, types):
    """Short room counts for the requested types, e.g. 'bedroom 2 · bathroom 1'."""
    return " · ".join(f"{t} {counts.get(t, 0)}" for t in types) if types else ""


def result_row(s):
    """Serialisable result (explanation included)."""
    return {"rank": s.get("rank"), "id": s["id"], "dataset": s["dataset"], "split": s["split"],
            "final": round(s["final"], 6), "constraint_score": round(s["constraint_score"], 6),
            "plausibility": round(s["plausibility"], 6),
            "plausibility_components": {k: round(v, 6) for k, v in s["plausibility_components"].items()},
            "area_m2": s["area_m2"], "gross_area_m2": s["gross_area_m2"], "area_source": s["area_source"],
            "area_estimated": s["area_source"] == "calibrated", "n_rooms": s["n_rooms"],
            "counts": {k: v for k, v in s["counts"].items() if v}, "entrance_type": s["entrance_type"],
            "violated": s["violated"], "unknown": s["unknown"],
            "explanation": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                            for row in s["explanation"]]}


def explanation_table(s):
    """Rows of the per-result explanation table for the report."""
    rows = []
    for row in s["explanation"]:
        value = row["value"]
        if row["requirement"] == "area" and value is not None:
            value = area_text(value, s["area_source"]) + f" ({s['area_source']})"
        rows.append({"requirement": row["requirement"], "wanted": row["wanted"], "value": value,
                     "score": row["score"] if row["score"] is not None else "-",
                     "hard": "yes" if row["hard"] else "no", "status": row["status"]})
    return rows


def plan_images(store, ids, cfg, taxonomy):
    """Phase-04 renders (re-rendered from the layout when the PNG is missing)."""
    from prism.vision.render import render_layout

    images = []
    for sid in ids:
        try:
            images.append(store.image(sid))
        except (FileNotFoundError, OSError):
            images.append(render_layout(store.layout(sid), taxonomy, cfg["render"]))
    return images


def main():
    """Parse the request, build the catalog, search, write outputs and the report."""
    args = build_cli().parse_args()
    cfg, ws = setup(args, PHASE)
    taxonomy = Taxonomy(cfg, ws.dataset)

    # ----------------------------------------------------------- the request
    try:
        cli_given = any(getattr(args, f) not in (None, [], "") for f in CLI_REQUIREMENT_FLAGS)
        spec_path = args.spec or (None if cli_given else cfg.get("constraints", {}).get("default_spec"))
        raw = load_spec_file(resolve_path(spec_path)) if spec_path else {}
        raw = apply_cli(raw, args)
        spec = normalize_spec(raw, cfg, taxonomy, ws.member_datasets)
    except SpecError as exc:
        print(f"\n  Invalid request: {exc}\n")
        sys.exit(2)
    out = ws.phase_dir(PHASE) / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "spec_normalised.json", {**spec, "source": str(spec_path) if spec_path else "cli"})

    # ----------------------------------------------------------- catalog + search
    with Timer("catalog"):
        records, info = load_catalog(ws, cfg, list(taxonomy.unified_names), args.rebuild_catalog)
    store = LayoutStore(ws)
    with Timer("search"):
        res = run_search(records, spec, info["priors"], cfg, descriptor_vectors(ws, store, taxonomy),
                         fine_level=taxonomy.level == "fine")
    results, misses, reqs = res["results"], res["near_misses"], res["requirements"]
    req_types = list(spec["room_types"])

    write_json(out / "results.json", {"spec": spec["name"], "dataset": ws.dataset, "mode": ws.mode,
                                      "gallery": res["pool_size"], "candidates": len(res["candidates"]),
                                      "diversity_used": res["diversity_used"], "calibration": info["calibration"],
                                      "results": [result_row(s) for s in results]})
    write_json(out / "near_misses.json", [result_row(s) for s in misses])
    write_csv(out / "results.csv", [{
        "rank": s["rank"], "id": s["id"], "dataset": s["dataset"], "final": round(s["final"], 6),
        "constraint_score": round(s["constraint_score"], 6), "plausibility": round(s["plausibility"], 6),
        "area_m2": s["area_m2"], "area_source": s["area_source"], "n_rooms": s["n_rooms"],
        **{f"n_{t}": s["counts"].get(t, 0) for t in req_types}} for s in results])

    # ----------------------------------------------------------- console
    print(f"\n  {len(res['candidates'])} of {res['pool_size']} plans meet every hard requirement; "
          f"showing {len(results)}.")
    for s in results:
        print(f"  {s['rank']:>3}  {s['id']:<18} final {s['final']:.3f}  {area_text(s['area_m2'], s['area_source']):>9}"
              f"  rooms {s['n_rooms']}  {counts_text(s['counts'], req_types)}")
    if not results:
        print("  No plan meets every hard requirement. Most restrictive: "
              + ", ".join(f"{b['requirement']} ({b['plans not meeting it']})" for b in res["blockers"][:3]))

    # ----------------------------------------------------------- report
    rep = HtmlReport(f"13 - Constraint search: {spec['name']}", f"{ws.dataset} · {ws.mode}")
    rep.heading("Request")
    rep.table([{"requirement": r["name"], "wanted": r["wanted"], "hard": "yes" if r["hard"] else "no",
                "weight": r["weight"]} for r in reqs] or [{"requirement": "(none)", "wanted": "-", "hard": "-",
                                                          "weight": "-"}])
    rep.text(f"k = {spec['k']}, datasets = {spec['datasets'] or 'all of the workspace'}, split = {spec['split']}, "
             f"plausibility weight = {spec['ranking']['plausibility_weight']}, "
             f"diversity (MMR λ) = {spec['ranking']['diversity']}"
             + ("" if res["diversity_used"] or spec["ranking"]["diversity"] == 0 else " (not applied)") + "."
             + (" Results are in MMR pick order: each pick trades final score against similarity to the plans "
                "already picked, so scores are not strictly decreasing." if res["diversity_used"] else
                " Results are ranked by final score."))
    share = Counter(s["dataset"] for s in results)
    unknown_area = sum(1 for s in results if s["area_m2"] is None)
    rep.kpis({"gallery": res["pool_size"], "pass every hard requirement": len(res["candidates"]),
              "results": len(results), "results with unknown area": unknown_area,
              **{f"results from {d}": n for d, n in sorted(share.items())}})
    if res["shortfall"]:
        rep.note(f"Only {len(res['candidates'])} plans meet every hard requirement (k = {spec['k']}). The list is "
                 "NOT filled with plans that violate a hard requirement; see the near misses below.", warn=True)
    if unknown_area:
        rep.note(f"{unknown_area} results have no known area (RPLAN without calibration): area requirements are "
                 "marked 'unknown' for them and left out of their score.", warn=True)
    if res["funnel"]:
        rep.heading("Hard-requirement funnel")
        rep.text("How many plans each hard requirement removes, alone and in request order.")
        rep.table(res["funnel"])
    if not res["candidates"]:
        rep.heading("Why nothing matches")
        rep.text("The requirements that the most plans fail - relax these first.")
        rep.table(res["blockers"])

    if results:
        rep.heading(f"Top {len(results)}")
        titles = [f"#{s['rank']} {s['id']}\nfinal {s['final']:.2f} · {area_text(s['area_m2'], s['area_source'])}\n"
                  f"{caption_counts(s['counts'], req_types)}" for s in results]
        rep.figure(image_grid(plan_images(store, [s["id"] for s in results], cfg, taxonomy), titles, ncols=5,
                              cell=2.8), "Renders of the results ('≈' = area estimated by calibration)")
        for s in results:
            rep.heading(f"#{s['rank']} {s['id']} - final {s['final']:.3f} (constraints {s['constraint_score']:.3f}, "
                        f"plausibility {s['plausibility']:.3f})", level=3)
            rep.table(explanation_table(s))
            rep.text("Plausibility: " + ", ".join(f"{k} {v:.2f}" for k, v in s["plausibility_components"].items())
                     + f". Rooms {s['n_rooms']}, area {area_text(s['area_m2'], s['area_source'])} "
                     f"({s['area_source'] or 'unknown'}), entrance {s['entrance_type'] or '-'}.")

    if misses:
        rep.heading("Near misses (violate at least one hard requirement)")
        titles = [f"{s['id']}\nfinal {s['final']:.2f}\nviolates: {', '.join(s['violated'])}\n"
                  f"{caption_counts(s['counts'], req_types)}" for s in misses]
        rep.figure(image_grid(plan_images(store, [s["id"] for s in misses], cfg, taxonomy), titles, ncols=5, cell=2.8),
                   "Best plans that fail a hard requirement - never returned as results")
        rep.table([{"id": s["id"], "final": s["final"], "violated": ", ".join(s["violated"]),
                    "area": area_text(s["area_m2"], s["area_source"]), "rooms": s["n_rooms"],
                    "counts": counts_text(s["counts"], req_types)} for s in misses])

    # Area calibration.
    rep.heading("Area in m²: where it comes from")
    cal, cross = info["calibration"], info["units_cross_check"]
    src = Counter(r["area_source"] or "unknown" for r in records)
    rep.table([{"area source": k, "plans": v} for k, v in src.most_common()])
    if cal:
        rep.kpis({"RPLAN m² per px²": cal["scale"], "method": cal["method"],
                  "ResPlan plans used": (cal.get("n_plans") or {}).get("resplan", "-")})
        if cal.get("per_type_ratios"):
            rep.table([{"type": t, "ratio m²/px²": v} for t, v in cal["per_type_ratios"].items()])
        rep.text("RPLAN has no metric scale: its areas are ESTIMATES (canvas px² x scale), shown as '≈'. "
                 "Force a scale with --set constraints.rplan_m2_per_px2=<value>.")
    elif any(r["dataset"] == "rplan" for r in records):
        rep.note("No RPLAN calibration (no ResPlan data and no constraints.rplan_m2_per_px2): RPLAN areas are "
                 "unknown.", warn=True)
    if cross:
        msg = (f"ResPlan units check over {cross['n_plans']} plans: outline in source units² / dataset area = "
               f"{cross['median_ratio']:.3g} (median), spread across plans (CV) {cross['spread_cv']:.2f}; "
               f"net_area consistent with area in {cross['net_area_consistent_share']:.0%} of plans.")
        if cross["metres"]:
            rep.text(msg + " The coordinates are metres.")
        else:
            rep.note(msg + (" The ResPlan coordinates are normalised per plan (not metres), so ResPlan m² are "
                            "anchored on each plan's own 'area' field; net_area is not reliable."
                            if cross["normalised_per_plan"] else " The ResPlan coordinates are NOT metres."))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for d in sorted({r["dataset"] for r in records}):
        vals = [r["net_area_m2"] for r in records if r["dataset"] == d and r["net_area_m2"]]
        if vals:
            label = f"{d} ({'calibrated' if d == 'rplan' else 'dataset'})"
            ax.hist(vals, bins=30, alpha=0.55, label=label)
    ax.set_xlabel("net area (m²)")
    ax.set_ylabel("plans")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    rep.figure(fig, "Net area per dataset after calibration")

    if res["candidates"]:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        ax.hist([s["final"] for s in res["candidates"]], bins=30, color="#2f6fde")
        ax.set_xlabel("final score")
        ax.set_ylabel("candidates")
        ax.spines[["top", "right"]].set_visible(False)
        rep.figure(fig, "Score distribution of the plans meeting every hard requirement")
    if results:
        comps = list(results[0]["plausibility_components"])
        rep.figure(grouped_bars(comps, {f"#{s['rank']}": [s["plausibility_components"][c] for c in comps]
                                        for s in results}, "Plausibility components of the top-k"))
    rep.json_block({"catalog": {k: v for k, v in info.items() if k not in ("priors",)}, "spec": spec}, max_chars=5000)
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
