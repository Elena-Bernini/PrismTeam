"""ResPlan geometry check - does a ResPlan render look like an RPLAN render?

ResPlan rooms are separated by real walls; RPLAN rooms tile the building outline.
Phase 02 turns ResPlan into an RPLAN-style partition (``resplan.room_partition``,
see ``src/prism/datasets/resplan_geometry.py``). This script shows, for a sample
of plans, the original ResPlan geometry, the render WITHOUT the partition and the
render WITH it, next to RPLAN renders for reference, and measures:

- ``gap share``: share of the building interior left grey in the render (double
  walls, holes between rooms). RPLAN is the reference (~0);
- ``detached parts``: connected pieces of the plan in the render (1 = one building);
- ``rooms outside canvas``, partition overlap and conversion time.

It reads ``ResPlan.pkl`` directly and writes nothing under ``data/processed``.

    python scripts/resplan_geometry_check.py                       # 24 random plans
    python scripts/resplan_geometry_check.py --ids 13997 1022 10494
    python scripts/resplan_geometry_check.py --n 200 --set resplan.partition.growth=0.6
"""

import copy
import random
import time

import numpy as np

import sys
from pathlib import Path

# Make `import prism` work however the script is launched (python scripts/x.py,
# VS Code "Run Python File", python -m scripts.x) even without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.common.cli import build_parser, setup
from prism.common.io import write_json
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import plt
from prism.datasets.resplan import (iter_polygons, iter_selected_plans, load_resplan_pickle, load_resplan_splits,
                                    plan_source_id, convert_plan)
from prism.datasets.schema import Taxonomy
from prism.vision.render import BACKGROUND, hex_to_rgb, render_layout

PHASE = "resplan_geometry_check"
ORIGINAL_COLORS = {"living": "#e8e8e8", "bedroom": "#f6d77a", "kitchen": "#f19e9e", "bathroom": "#8cc7ec",
                   "balcony": "#9bd17a", "storage": "#c9ad94", "stair": "#8d8d8d", "dining": "#f4c38b",
                   "veranda": "#9bd17a", "garden": "#cfe8c0", "parking": "#d0d0d0", "pool": "#a0d8ef"}


def render_metrics(img, rcfg):
    """Gap share inside the building and number of detached parts, measured on a render."""
    from PIL import Image  # noqa: F401  (PIL is a base dependency; imported for clarity)

    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    grey = np.all(np.abs(a - np.asarray(hex_to_rgb(rcfg.get("unknown_fill", "#bdbdbd")))) <= 3, axis=-1)
    background = np.all(np.abs(a - np.asarray(BACKGROUND)) <= 3, axis=-1)
    building = ~background
    parts = _components(building)
    return {"gap_share": float(grey.sum() / max(1, building.sum())), "detached_parts": parts}


def _components(mask):
    """Number of 4-connected components of a boolean mask with at least 20 pixels."""
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    count = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack, size = [(y0, x0)], 0
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        count += int(size >= 20)
    return count


def draw_original(ax, plan, title):
    """ResPlan plan in its original coordinates: rooms, walls, doors, windows, front door."""
    from matplotlib.patches import Polygon as MplPolygon

    for cat, col in ORIGINAL_COLORS.items():
        for ext, _ in iter_polygons(plan.get(cat)):
            ax.add_patch(MplPolygon(ext, closed=True, facecolor=col, edgecolor="#333", lw=0.4, zorder=2))
    for key, col in (("wall", "#555555"), ("door", "#e78ac3"), ("window", "#4caf50"), ("front_door", "#ffd600")):
        for ext, _ in iter_polygons(plan.get(key)):
            ax.add_patch(MplPolygon(ext, closed=True, facecolor=col, edgecolor="none", zorder=3))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8)


def main():
    """Convert a sample of ResPlan plans with and without the partition and compare the renders."""
    parser = build_parser(__doc__)
    parser.add_argument("--ids", nargs="+", default=None, help="ResPlan source ids to show (default: random sample)")
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = "resplan"
    cfg, ws = setup(args, PHASE)
    out = ws.phase_dir(PHASE)
    rcfg = cfg["render"]
    taxonomy = Taxonomy(cfg, "resplan")

    plans = load_resplan_pickle(ws.raw_path("resplan.pkl"))
    selected = list(iter_selected_plans(plans, load_resplan_splits(ws.raw_path("resplan.split_json")),
                                        bool(cfg["resplan"]["include_augmented"])))
    if args.ids:
        wanted = {str(i) for i in args.ids}
        sample = [(i, p) for i, p in selected if plan_source_id(p, i) in wanted]
    else:
        n = int(args.n or 24)
        sample = random.Random(int(cfg["preview"]["seed"])).sample(selected, min(n, len(selected)))

    cfg_old = copy.deepcopy(cfg)
    cfg_old["resplan"]["room_partition"] = False
    rows, examples = [], []
    for i, plan in sample:
        sid = plan_source_id(plan, i)
        row = {"id": sid}
        for name, c in (("before", cfg_old), ("after", cfg)):
            t0 = time.perf_counter()
            try:
                layout = convert_plan(plan, i, taxonomy, c)
            except Exception as exc:
                row[f"{name}_error"] = repr(exc)
                continue
            row[f"{name}_ms"] = 1000 * (time.perf_counter() - t0)
            img = render_layout(layout, taxonomy, rcfg)
            m = render_metrics(img, rcfg)
            row[f"{name}_gap_share"] = m["gap_share"]
            row[f"{name}_parts"] = m["detached_parts"]
            row[f"{name}_outside"] = sum(1 for r in layout["rooms"]
                                         if any(not (-1 <= v <= 257) for pt in r["polygon"] for v in pt))
            if name == "after":
                row["overlap_ratio"] = (layout["geometry_hints"].get("partition") or {}).get("overlap_ratio", 0.0)
            if len(examples) < 12 or args.ids:
                row.setdefault("_imgs", {})[name] = img
        if "_imgs" in row:
            examples.append((plan, row.pop("_imgs"), sid))
        rows.append(row)

    def agg(key, fn=np.mean):
        """Aggregate a numeric column over the sample."""
        vals = [r[key] for r in rows if key in r]
        return float(fn(vals)) if vals else float("nan")

    summary = {
        "plans": len(rows),
        "failures_before": sum(1 for r in rows if "before_error" in r),
        "failures_after": sum(1 for r in rows if "after_error" in r),
        "gap_share_before_mean": agg("before_gap_share"), "gap_share_after_mean": agg("after_gap_share"),
        "gap_share_after_max": agg("after_gap_share", np.max),
        "plans_with_detached_parts_before": sum(1 for r in rows if r.get("before_parts", 1) > 1),
        "plans_with_detached_parts_after": sum(1 for r in rows if r.get("after_parts", 1) > 1),
        "rooms_outside_canvas_after": int(sum(r.get("after_outside", 0) for r in rows)),
        "overlap_ratio_max": agg("overlap_ratio", np.max),
        "ms_per_plan_after": agg("after_ms"),
    }

    # RPLAN reference renders (if the rplan preview exists).
    rplan_imgs, rplan_gap = [], []
    rplan_dir = ws.data_root / "processed" / "rplan" / ws.mode / "images"
    if rplan_dir.exists():
        from PIL import Image
        for p in sorted(rplan_dir.glob("*.png"))[:60]:
            img = Image.open(p).convert("RGB")
            rplan_gap.append(render_metrics(img, rcfg)["gap_share"])
            if len(rplan_imgs) < 6:
                rplan_imgs.append((p.stem, img))
        summary["rplan_reference_gap_share_mean"] = float(np.mean(rplan_gap))
    write_json(out / "summary.json", {"summary": summary, "rows": rows})
    for k, v in summary.items():
        print(f"  {k}: {v}")

    rep = HtmlReport("ResPlan geometry check: RPLAN-style partition", f"{ws.mode} · {len(rows)} plans")
    rep.kpis({"gap share before": summary["gap_share_before_mean"], "gap share after": summary["gap_share_after_mean"],
              "RPLAN gap share": summary.get("rplan_reference_gap_share_mean", "n/a"),
              "detached plans before": summary["plans_with_detached_parts_before"],
              "detached plans after": summary["plans_with_detached_parts_after"],
              "failures after": summary["failures_after"], "ms / plan": summary["ms_per_plan_after"]})
    rep.text("gap share = grey pixels inside the building (double walls, holes between rooms) / building pixels. "
             "detached = plans whose render has more than one separate piece (e.g. a floating balcony).")
    if rplan_imgs:
        fig, axes = plt.subplots(1, len(rplan_imgs), figsize=(2.6 * len(rplan_imgs), 2.8))
        for ax, (name, img) in zip(np.atleast_1d(axes), rplan_imgs):
            ax.imshow(img)
            ax.set_title(name, fontsize=8)
            ax.axis("off")
        rep.heading("Reference: RPLAN renders")
        rep.figure(fig, "Target style: rooms tile the outline, one line per wall")
    rep.heading("ResPlan: original / before / after")
    for plan, imgs, sid in examples:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        draw_original(axes[0], plan, f"ResPlan.pkl {sid} (original, y up)")
        for ax, name in zip(axes[1:], ("before", "after")):
            if name in imgs:
                ax.imshow(imgs[name])
            ax.axis("off")
            row = next(r for r in rows if r["id"] == sid)
            ax.set_title(f"{name}: gap {row.get(f'{name}_gap_share', float('nan')):.3f}, "
                         f"parts {row.get(f'{name}_parts', '-')}", fontsize=8)
        fig.tight_layout()
        rep.figure(fig, f"resplan_{sid}")
    worst = sorted((r for r in rows if "after_gap_share" in r), key=lambda r: -r["after_gap_share"])[:15]
    rep.heading("Worst plans after the partition")
    rep.table(worst)
    rep.json_block(summary)
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
