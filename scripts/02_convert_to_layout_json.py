"""Phase 02 - convert the dataset into one layout JSON per floor plan.

Keeps Pier's RPLAN -> JSON conversion (same field names) and adds the official
room taxonomy, unified types, polygon/box alignment, true polygon areas and the
FRONT DOOR block (segment, facing, entrance room).

In preview mode a fixed random subset of ``--n`` plans is converted and its ids
are saved, so that every later phase in preview mode works on the same plans.

    python scripts/02_convert_to_layout_json.py --mode preview --n 200
    python scripts/02_convert_to_layout_json.py --mode full
"""

import random
from collections import Counter

import numpy as np

import sys
from pathlib import Path

# Make `import prism` work however the script is launched (python scripts/x.py,
# VS Code "Run Python File", python -m scripts.x) even without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.common.cli import Timer, build_parser, setup
from prism.common.io import clean_dir, ensure_dir, progress, write_json
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import draw_elements, draw_layout, plt
from prism.datasets.rplan import convert_sample, load_g2p_pickle, sample_id, samples_from_pickle
from prism.datasets.schema import Taxonomy

PHASE = "02_convert_to_layout_json"


def convert_rplan(cfg, ws, taxonomy):
    """Convert RPLAN samples and return the list of layouts written."""
    pkl = ws.raw_path("rplan.pkl")
    samples = samples_from_pickle(load_g2p_pickle(pkl))
    if ws.mode == "preview":
        rng = random.Random(int(cfg["preview"]["seed"]))
        samples = rng.sample(samples, min(int(cfg["preview"]["n"]), len(samples)))
    layouts, failures = [], []
    for s in progress(samples, desc="Converting to layout JSON", total=len(samples)):
        try:
            layout = convert_sample(s, taxonomy, cfg, pkl)
        except Exception as exc:  # keep going, report at the end
            failures.append({"id": sample_id(s), "error": repr(exc)})
            continue
        write_json(ws.layouts_dir / f"{layout['id']}.json", layout)
        layouts.append(layout)
    return layouts, failures


def convert_resplan(cfg, ws, taxonomy):
    """Convert ResPlan plans (augmented copies skipped unless resplan.include_augmented)."""
    from prism.datasets.resplan import (convert_plan, iter_selected_plans, load_resplan_pickle,
                                        load_resplan_splits, plan_source_id)

    plans = load_resplan_pickle(ws.raw_path("resplan.pkl"))
    splits = load_resplan_splits(ws.raw_path("resplan.split_json"))
    selected = list(iter_selected_plans(plans, splits, bool(cfg["resplan"]["include_augmented"])))
    if ws.mode == "preview":
        rng = random.Random(int(cfg["preview"]["seed"]))
        selected = rng.sample(selected, min(int(cfg["preview"]["n"]), len(selected)))
    layouts, failures = [], []
    for index, plan in progress(selected, desc="Converting ResPlan", total=len(selected)):
        try:
            layout = convert_plan(plan, index, taxonomy, cfg)
        except Exception as exc:
            failures.append({"id": plan_source_id(plan, index), "error": repr(exc)})
            continue
        write_json(ws.layouts_dir / f"{layout['id']}.json", layout)
        layouts.append(layout)
    return layouts, failures


def main():
    """Convert, write metadata and build the visual report."""
    args = build_parser(__doc__).parse_args()
    cfg, ws = setup(args, PHASE)
    for member in ws.members():          # --dataset combined: run once per member dataset
        if ws.is_combined:
            print(f"--- {PHASE} on {member.dataset} ---")
        run(member.cfg, member)


def run(cfg, ws):
    """Run the phase on one dataset (called once per member dataset for --dataset combined)."""
    out = ws.phase_dir(PHASE)
    taxonomy = Taxonomy(cfg, ws.dataset)
    clean_dir(ws.layouts_dir)
    ensure_dir(ws.metadata_dir)

    with Timer("conversion"):
        layouts, failures = (convert_rplan if ws.dataset == "rplan" else convert_resplan)(cfg, ws, taxonomy)
    if not layouts:
        raise RuntimeError(f"No layout converted. First failures: {failures[:3]}")

    metadata = []
    for l in layouts:
        fd = l.get("front_door") or {}
        metadata.append({
            "id": l["id"], "num_rooms": len(l["rooms"]),
            "unified_counts": dict(Counter(r["unified_type"] for r in l["rooms"])),
            "has_front_door": bool(fd), "door_facing": fd.get("facing"),
            "entrance_room_type": (l["rooms"][fd["entrance_room_id"]]["room_type_name"]
                                   if fd.get("entrance_room_id") is not None else None),
            "polygon_alignment": (l.get("raw") or {}).get("polygon_alignment", {}).get("method"),
            "wall_thickness_px": (l.get("geometry_hints") or {}).get("wall_thickness_px"),
        })
    write_json(ws.metadata_dir / "layouts_metadata.json", metadata)
    write_json(ws.metadata_dir / "ids.json", [m["id"] for m in metadata], indent=None)
    write_json(out / "failures.json", failures)

    door_ok = sum(m["has_front_door"] for m in metadata)
    entrance_ok = sum(m["entrance_room_type"] is not None for m in metadata)
    facing = Counter(m["door_facing"] for m in metadata if m["has_front_door"])
    entrance_types = Counter(m["entrance_room_type"] for m in metadata if m["entrance_room_type"])
    align = Counter(m["polygon_alignment"] for m in metadata)

    rep = HtmlReport("02 - Layout JSON conversion", f"{ws.dataset} · {ws.mode}")
    rep.kpis({"layouts": len(layouts), "failures": len(failures), "with front door": door_ok,
              "with entrance room": entrance_ok, "mean rooms": float(np.mean([m["num_rooms"] for m in metadata]))})
    rep.heading("Front door")
    rep.text("Facing = which side of the building the door opens to (north = top of the image). The entrance "
             "room is the room closest to the door segment (within geometry.entrance_max_dist_px).")
    rep.table([{"facing": k, "count": v} for k, v in facing.most_common()])
    rep.table([{"entrance room type": k, "count": v} for k, v in entrance_types.most_common()])
    if ws.dataset == "rplan":
        rep.heading("Polygon / box alignment")
        rep.table([{"method": k, "layouts": v} for k, v in align.most_common()])
    else:
        walls = [m["wall_thickness_px"] for m in metadata if m["wall_thickness_px"]]
        rep.heading("Alignment with RPLAN")
        rep.text("With resplan.room_partition the wall-separated ResPlan rooms are turned into an RPLAN-style "
                 "partition: rooms meet on the wall centre lines, balconies are attached, small steps are "
                 "straightened and the rooms tile the outline (see src/prism/datasets/resplan_geometry.py and "
                 "scripts/resplan_geometry_check.py). The plan is then fitted into the 256 px canvas with "
                 f"resplan.fit_ratio={cfg['resplan']['fit_ratio']}, y flipped to image convention. The original "
                 "room polygons are kept (source_polygons) to label doors / arches / windows.")
        if walls:
            rep.kpis({"wall thickness px (median)": float(np.median(walls)), "min": float(np.min(walls)),
                      "max": float(np.max(walls))})
        rep.table([{"fine type": k, "rooms": v} for k, v in
                   Counter(r["room_type_name"] for l in layouts for r in l["rooms"]).most_common()])

    title = ("Examples: snapshot (left) vs layout JSON (right)" if ws.dataset == "rplan"
             else "Examples: walls / doors / windows kept from ResPlan (left) vs layout JSON (right)")
    rep.heading(title)
    snap_dir = ws.raw_path("rplan.snapshot_dir") if ws.dataset == "rplan" else None
    for l in layouts[:12]:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        if snap_dir is not None:
            sp = snap_dir / f"{l['source_id']}.png"
            if sp.exists():
                axes[0].imshow(plt.imread(str(sp)), extent=(0, 256, 256, 0))
            axes[0].set_title("snapshot" if sp.exists() else "snapshot not found", fontsize=8)
            axes[0].axis("off")
        else:
            draw_elements(axes[0], l, title="walls / doors / windows")
        fd = l.get("front_door") or {}
        draw_layout(axes[1], l, taxonomy,
                    title=f"{l['id']} · door {fd.get('facing', '-')} · entrance {fd.get('entrance_room_id', '-')}")
        rep.figure(fig, f"Layout {l['id']} - yellow segment = front door")
    if failures:
        rep.heading("Failures")
        rep.table(failures[:50])
    rep.heading("Example layout JSON")
    rep.json_block(layouts[0], max_chars=4000)
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
