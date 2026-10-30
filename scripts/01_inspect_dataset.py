"""Phase 01 - inspect the raw RPLAN / Graph2Plan pickle before converting it.

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

import scripts._bootstrap as _bootstrap  # noqa: F401

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


def main():
    """Run the inspection and write the report."""
    args = build_parser(__doc__).parse_args()
    cfg, ws = setup(args, PHASE)
    out = ws.phase_dir(PHASE)
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
        "room_type_counts": {f"{k} {taxonomy.name(k)}": v for k, v in sorted(room_types.items())},
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
