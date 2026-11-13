"""Phase 01 - inspect a raw dataset before converting it (RPLAN pickle or ResPlan.pkl).

ResPlan (``--dataset resplan``): keys and geometry types of the plan
dictionaries (unknown keys are flagged), rooms per category, units (wall depth,
outline size), front door, ``split.json`` and the edge types of ResPlan's own
graph when present (Kaggle release), plus plans drawn in original coordinates.

RPLAN (default):

What it checks (and shows in the report):

- the files present under data/raw/rplan and the pickle structure;
- the attributes of a sample (Pier's audit);
- what the THIRD COLUMN of ``sample.edge`` actually contains (value histogram):
  this is the column Pier's code labelled as a spatial relation;
- the front door: rows 0-1 of ``sample.boundary`` (Graph2Plan convention),
  whether column 3 flags exactly those rows, and an overlay of the door segment
  on the snapshot next to the yellow pixels detected in it;
- whether ``rBoundary[i]`` (room polygon) really matches ``box[i]`` (IoU).

    python scripts/01_inspect_dataset.py --mode preview --n 200
    python scripts/01_inspect_dataset.py --mode full
"""

import random
from collections import Counter

import numpy as np
from PIL import Image

import sys
from pathlib import Path

# Make `import prism` work however the script is launched (python scripts/x.py,
# VS Code "Run Python File", python -m scripts.x) even without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.common.cli import build_parser, setup
from prism.common.io import to_jsonable, write_json
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import plt
from prism.datasets.rplan import load_g2p_pickle, parse_g2p_edges, sample_id, samples_from_pickle
from prism.datasets.schema import G2P_RELATION_GUESS, Taxonomy
from prism.geometry.front_door import detect_yellow_door_pixels, door_segment_from_g2p_boundary
from prism.geometry.polygon import as_points, bbox_iou, point_segment_distance, polygon_bbox

PHASE = "01_inspect_dataset"


def summarize_value(value, max_items=6):
    """Compact, JSON-safe description of an attribute value (Pier's audit helper)."""
    if isinstance(value, np.ndarray):
        out = {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        if value.dtype != object:
            out["preview"] = to_jsonable(value.flatten()[:max_items])
        return out
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "len": len(value),
                "first": summarize_value(value[0]) if value else None}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(type(value))


def door_vs_snapshot(sample, snapshot_path):
    """Distance (px, 256 canvas) between the door segment and the yellow pixels of the snapshot."""
    seg = door_segment_from_g2p_boundary(sample.boundary)
    if seg is None or not snapshot_path.exists():
        return None, None, None
    img = np.asarray(Image.open(snapshot_path).convert("RGB"))
    pix = detect_yellow_door_pixels(img)
    sx, sy = 256.0 / img.shape[1], 256.0 / img.shape[0]
    if pix is None:
        return None, img, seg
    centre = (pix[:, 0].mean() * sx, pix[:, 1].mean() * sy)
    return point_segment_distance(centre, seg[0], seg[1]), img, seg


def inspect_resplan(cfg, ws, out):
    """ResPlan inspection: keys, geometry types, units, wall depth, front door, split, own graph."""
    from prism.datasets.resplan import (geometry_points, iter_polygons, load_resplan_pickle, load_resplan_splits,
                                        plan_source_id, room_categories, unknown_keys)

    taxonomy = Taxonomy(cfg, "resplan")
    plans = load_resplan_pickle(ws.raw_path("resplan.pkl"))
    splits = load_resplan_splits(ws.raw_path("resplan.split_json"))
    rng = random.Random(int(cfg["preview"]["seed"]))
    subset = plans if ws.mode == "full" else rng.sample(plans, min(int(cfg["preview"]["n"]), len(plans)))

    key_counts, geom_types, unknown, rooms_per_cat = Counter(), {}, Counter(), Counter()
    wall_depths, spans, fd_parts, graph_edge_types = [], [], Counter(), Counter()
    for p in subset:
        key_counts.update(p.keys())
        for k, v in p.items():
            geom_types.setdefault(k, Counter())[type(v).__name__] += 1
        unknown.update(unknown_keys(p, taxonomy))
        for c in room_categories(p, taxonomy):
            rooms_per_cat[c] += 1 if c == "living" else len(iter_polygons(p[c]))
        try:
            wall_depths.append(float(p.get("wall_depth")))
        except (TypeError, ValueError):
            pass
        pts = np.asarray(geometry_points(p.get("inner")) or [[0, 0]], dtype=float)
        spans.append(float((pts.max(axis=0) - pts.min(axis=0)).max()))
        fd_parts[len(iter_polygons(p.get("front_door")))] += 1
        g = p.get("graph")
        if g is not None and hasattr(g, "edges"):
            graph_edge_types.update(str((d or {}).get("type", "?")) for _, _, d in g.edges(data=True))

    ids = [plan_source_id(p, i) for i, p in enumerate(plans)]
    split_info = None
    if splits:
        known = set(ids)
        split_info = {k: {"size": len(v), "ids_found_in_pickle": sum(1 for x in v if x in known)} for k, v in splits.items()}
    stats = {"num_plans": len(plans), "inspected": len(subset), "keys": dict(key_counts),
             "geometry_types": {k: dict(v) for k, v in geom_types.items()}, "unknown_keys": dict(unknown),
             "rooms_per_category": dict(rooms_per_cat),
             "wall_depth": {"min": float(np.min(wall_depths)), "median": float(np.median(wall_depths)),
                            "max": float(np.max(wall_depths))} if wall_depths else None,
             "inner_longest_side": {"min": float(np.min(spans)), "median": float(np.median(spans)),
                                    "max": float(np.max(spans))},
             "front_door_parts": dict(fd_parts), "own_graph_edge_types": dict(graph_edge_types),
             "split_json": split_info}
    write_json(out / "inspection_stats.json", stats)

    rep = HtmlReport("01 - Dataset inspection (ResPlan)", f"{ws.dataset} · {ws.mode}")
    rep.kpis({"plans in pickle": len(plans), "inspected": len(subset),
              "with front door": sum(v for k, v in fd_parts.items() if k > 0),
              "median wall depth": stats["wall_depth"]["median"] if wall_depths else "n/a",
              "median outline size": stats["inner_longest_side"]["median"],
              "own graph": "yes" if graph_edge_types else "no (GitHub release)"})
    if unknown:
        rep.note("Keys not listed in configs/room_types.yaml (resplan / resplan_non_room): "
                 + ", ".join(f"{k} ({v})" for k, v in unknown.most_common())
                 + ". If they are room categories, add them to room_types.yaml -> resplan.", warn=True)
    rep.heading("Keys and geometry types")
    rep.table([{"key": k, "plans": key_counts[k], "types": ", ".join(f"{t} ({n})" for t, n in geom_types[k].items())}
               for k in sorted(key_counts)])
    rep.heading("Rooms per category")
    rep.table([{"category": k, "rooms": v, "unified": taxonomy.resplan_unified(k)} for k, v in rooms_per_cat.most_common()])
    rep.heading("Units")
    rep.text("The outline size and wall depth tell the coordinate units (metres: ~5-30 and ~0.1-0.3). Phase 02 "
             "rescales every plan into the 256 px canvas anyway, so both cases are handled.")
    rep.json_block({"wall_depth": stats["wall_depth"], "inner_longest_side": stats["inner_longest_side"]})
    if split_info:
        rep.heading("split.json")
        rep.table([{"split": k, **v} for k, v in split_info.items()])
    if graph_edge_types:
        rep.heading("Edge types of ResPlan's own graph")
        rep.table([{"type": k, "edges": v} for k, v in graph_edge_types.most_common()])

    colors = {"living": "#e8e8e8", "bedroom": "#f6d77a", "kitchen": "#f19e9e", "bathroom": "#8cc7ec",
              "balcony": "#9bd17a", "storage": "#c9ad94", "stair": "#8d8d8d"}
    from matplotlib.patches import Polygon as MplPolygon
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    for ax, p in zip(axes.ravel(), subset[:8]):
        for c in room_categories(p, taxonomy):
            for ext, _ in iter_polygons(p[c]):
                ax.add_patch(MplPolygon(ext, closed=True, facecolor=colors.get(c, "#d9d9d9"), edgecolor="#333", lw=0.4))
        for key, col in (("wall", "#555555"), ("door", "#e78ac3"), ("window", "#a6d854"), ("front_door", "#ffd600")):
            for ext, _ in iter_polygons(p.get(key)):
                ax.add_patch(MplPolygon(ext, closed=True, facecolor=col, edgecolor="none", alpha=0.9))
        pts = np.asarray(geometry_points(p.get("inner")) or [[0, 0], [1, 1]], dtype=float)
        ax.set_xlim(pts[:, 0].min() - 1, pts[:, 0].max() + 1)
        ax.set_ylim(pts[:, 1].min() - 1, pts[:, 1].max() + 1)
        ax.set_aspect("equal")
        ax.set_title(f"id {plan_source_id(p, 0)}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(subset[:8]):]:
        ax.axis("off")
    rep.heading("Plans in their original coordinates")
    rep.figure(fig, "Rooms by category, walls grey, doors pink, windows green, front door yellow (original axes, y up)")
    rep.heading("Statistics")
    rep.json_block(stats)
    finalize_report(rep, out, ws.mode)


def main():
    """Run the inspection and write the report."""
    args = build_parser(__doc__).parse_args()
    cfg, ws = setup(args, PHASE)
    for member in ws.members():          # --dataset combined: run once per member dataset
        if ws.is_combined:
            print(f"--- {PHASE} on {member.dataset} ---")
        run(member.cfg, member)


def run(cfg, ws):
    """Run the phase on one dataset (called once per member dataset for --dataset combined)."""
    out = ws.phase_dir(PHASE)
    if ws.dataset == "resplan":
        inspect_resplan(cfg, ws, out)
        return
    taxonomy = Taxonomy(cfg, "rplan")
    pkl = ws.raw_path("rplan.pkl")
    snap_dir = ws.raw_path("rplan.snapshot_dir")

    raw_root = ws.raw_root / "rplan"
    files = [p for p in raw_root.rglob("*") if p.is_file()] if raw_root.exists() else []
    file_audit = {"root": str(raw_root), "num_files": len(files),
                  "extensions": dict(Counter(p.suffix.lower() for p in files)),
                  "pkl": str(pkl), "pkl_exists": pkl.exists(), "snapshot_dir_exists": snap_dir.exists()}
    write_json(out / "file_audit.json", file_audit)

    obj = load_g2p_pickle(pkl)
    samples = samples_from_pickle(obj)
    first = samples[0]
    attrs = [a for a in dir(first) if not a.startswith("_") and not callable(getattr(first, a))]
    sample_report = {"pkl_keys": list(obj.keys()) if isinstance(obj, dict) else None,
                     "num_samples": len(samples), "attributes": {a: summarize_value(getattr(first, a)) for a in attrs}}
    write_json(out / "sample_report.json", sample_report)

    rng = random.Random(int(cfg["preview"]["seed"]))
    subset = samples if ws.mode == "full" else rng.sample(samples, min(int(cfg["preview"]["n"]), len(samples)))

    edge_col3, edge_cols, room_types = Counter(), Counter(), Counter()
    flag_rows_ok, flag_rows_total, door_widths, ious = 0, 0, [], []
    boundary_col2, boundary_col3 = Counter(), Counter()
    for s in subset:
        e = parse_g2p_edges(getattr(s, "edge", None))
        edge_col3.update(int(v) for v in e[:, 2])
        raw_e = np.asarray(getattr(s, "edge", np.zeros((0, 3))))
        edge_cols[int(raw_e.shape[-1]) if raw_e.ndim >= 2 else int(raw_e.size)] += 1
        b = np.asarray(s.boundary)
        if b.ndim == 2 and b.shape[1] >= 4:
            boundary_col2.update(int(v) for v in b[:, 2])
            boundary_col3.update(int(v) for v in b[:, 3])
            flag_rows_total += 1
            flagged = set(np.nonzero(b[:, 3] == 1)[0].tolist())
            flag_rows_ok += int(flagged == {0, 1})
        seg = door_segment_from_g2p_boundary(b)
        if seg:
            door_widths.append(float(np.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1])))
        boxes = np.asarray(s.box).reshape(-1, 5)
        room_types.update(int(t) for t in boxes[:, 4])
        polys = getattr(s, "rBoundary", None)
        if polys is not None:
            for i, bx in enumerate(boxes):
                if i < len(polys):
                    bb = polygon_bbox(as_points(polys[i]))
                    if bb:
                        ious.append(bbox_iou(bx[:4], bb))

    stats = {
        "inspected_samples": len(subset),
        "edge_third_column_values": dict(sorted(edge_col3.items())),
        "edge_num_columns": dict(edge_cols),
        "boundary_col2_values": dict(sorted(boundary_col2.items())),
        "boundary_col3_values": dict(sorted(boundary_col3.items())),
        "boundary_col3_flags_exactly_rows_0_1": f"{flag_rows_ok}/{flag_rows_total}",
        "door_width_px": {"min": float(np.min(door_widths)), "mean": float(np.mean(door_widths)),
                          "max": float(np.max(door_widths))} if door_widths else None,
        "room_type_counts": {f"{k} {taxonomy.rplan_name(k)}": v for k, v in sorted(room_types.items())},
        "polygon_box_iou_by_index": {"mean": float(np.mean(ious)), "p05": float(np.percentile(ious, 5)),
                                     "share_below_0.5": float(np.mean(np.asarray(ious) < 0.5))} if ious else None,
    }
    write_json(out / "inspection_stats.json", stats)

    # ------------------------------------------------------------------ report
    rep = HtmlReport("01 - Dataset inspection (RPLAN / Graph2Plan pickle)", f"{ws.dataset} · {ws.mode}")
    rep.kpis({"samples in pickle": len(samples), "inspected": len(subset), "raw files": len(files),
              "col3 flags door rows": stats["boundary_col3_flags_exactly_rows_0_1"],
              "mean polygon/box IoU": stats["polygon_box_iou_by_index"]["mean"] if ious else "n/a"})
    rep.heading("Third column of sample.edge")
    rep.text("Pier's code read this column as a relation id. Graph2Plan's own floorplan.py ignores it and "
             "recomputes the relation from the room boxes, so we treat it as an opaque value. Its histogram is "
             "shown here; phase 03 compares it with our recomputed relations.")
    rep.table([{"value": k, "count": v, "label in Pier's table": G2P_RELATION_GUESS.get(k, "-")}
               for k, v in sorted(edge_col3.items())])
    rep.heading("Front door")
    rep.text("Graph2Plan stores the front door as the first two points of sample.boundary (door = boundary[:2,:2]). "
             "Below, the door segment (cyan) is drawn on the snapshot; the yellow line of the snapshot should lie "
             "under it.")
    rep.json_block({"boundary_col2_values": stats["boundary_col2_values"],
                    "boundary_col3_values": stats["boundary_col3_values"], "door_width_px": stats["door_width_px"]})

    dists, shown = [], 0
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for s in subset[:60]:
        d, img, seg = door_vs_snapshot(s, snap_dir / f"{sample_id(s)}.png")
        if d is not None:
            dists.append(d)
        if img is not None and shown < 8:
            ax = axes.ravel()[shown]
            ax.imshow(img, extent=(0, 256, 256, 0))
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color="cyan", linewidth=2)
            ax.set_title(f"{sample_id(s)}  d={d:.1f}px" if d is not None else f"{sample_id(s)}  (no yellow found)",
                         fontsize=8)
            ax.axis("off")
            shown += 1
    for ax in axes.ravel()[shown:]:
        ax.axis("off")
    if shown:
        rep.figure(fig, "Door segment from boundary rows 0-1 (cyan) over the Graph2Plan snapshot")
    else:
        plt.close(fig)
        rep.note("No snapshots found: put snapshot_train/ next to the pickle to enable this check.", warn=True)
    if dists:
        rep.kpis({"snapshots checked": len(dists), "median door-to-yellow distance (px)": float(np.median(dists)),
                  "share within 6 px": float(np.mean(np.asarray(dists) <= 6))})
    rep.heading("Room polygons vs boxes")
    rep.text("IoU between the bounding box of rBoundary[i] and box[i]. A low mean means the polygon order differs "
             "from the box order; phase 02 then re-matches polygons by IoU automatically.")
    rep.json_block(stats["polygon_box_iou_by_index"])
    rep.heading("Room types (official RPLAN ids)")
    rep.table([{"type": k, "count": v} for k, v in stats["room_type_counts"].items()])
    rep.heading("Sample attributes")
    rep.json_block(sample_report)
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
