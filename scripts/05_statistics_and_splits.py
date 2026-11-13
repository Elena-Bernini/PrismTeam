"""Phase 05 - dataset statistics, train/val/test split and (optional) packing.

- Statistics over layouts and graphs (rooms, types, relations, front door).
- Split. RPLAN: each id is hashed with ``splits.seed`` (same plan -> same split,
  in preview and full mode, on every machine). ResPlan: the official
  ``split.json``. ``--dataset combined``: the union of the member splits, plus a
  report comparing the two datasets side by side (alignment check).
  The GNN trains on ``train``, early-stops on ``val``; retrieval queries come
  from ``retrieval.query_split`` (default ``test``) and search the whole gallery.
- ``--pack`` writes ``packed/layouts.jsonl.gz``, ``graphs.jsonl.gz`` and
  ``images.zip``: three large files instead of ~240k small ones, which is much
  faster on the cluster's BeeGFS ``/work`` (see cluster/README_cluster.md).

    python scripts/05_statistics_and_splits.py --mode preview
    python scripts/05_statistics_and_splits.py --mode full --pack
    python scripts/05_statistics_and_splits.py --dataset combined          # after rplan and resplan
"""

import hashlib
import zipfile
from collections import Counter

import numpy as np

import sys
from pathlib import Path

# Make `import prism` work however the script is launched (python scripts/x.py,
# VS Code "Run Python File", python -m scripts.x) even without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.common.cli import build_parser, setup
from prism.common.io import ensure_dir, progress, write_json, write_jsonl_gz
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import bar_chart, grouped_bars
from prism.datasets.schema import RELATIONS, dataset_of, make_id
from prism.datasets.store import LayoutStore, load_splits
from prism.graph.build import room_edges

PHASE = "05_statistics_and_splits"


def split_of(sid, seed, train, val):
    """Deterministic split for one id from an MD5 hash (independent of the subset)."""
    h = int(hashlib.md5(f"{seed}:{sid}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < train else ("val" if h < train + val else "test")


def pack(ws, store, ids):
    """Write the packed files used on the cluster."""
    ensure_dir(ws.packed_dir)
    n = write_jsonl_gz(ws.packed_dir / "layouts.jsonl.gz", (store.layout(s) for s in progress(ids, "pack layouts")))
    write_jsonl_gz(ws.packed_dir / "graphs.jsonl.gz", (store.graph(s) for s in progress(ids, "pack graphs")))
    with zipfile.ZipFile(ws.packed_dir / "images.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        for sid in progress(ids, "pack images"):
            p = ws.images_dir / f"{sid}.png"
            if p.exists():
                zf.write(p, arcname=f"{sid}.png")
    return n


def make_splits(cfg, ws, ids):
    """Train/val/test split for one dataset or the merge of the member splits (combined).

    - RPLAN: deterministic hash of the id (no official split exists).
    - ResPlan: the official ``split.json`` when present (augmented copies -> train
      if they were converted), hash for ids it does not list.
    - combined: union of the member datasets' splits (run phase 05 on each first).
    """
    if ws.is_combined:
        merged = {"train": [], "val": [], "test": []}
        available = set(ids)
        for d in ws.member_datasets:
            member_splits = load_splits(ws.member(d))
            for k in merged:
                merged[k].extend(x for x in member_splits.get(k, []) if x in available)
        return merged, "merged member splits"
    s = cfg["splits"]
    splits = {"train": [], "val": [], "test": []}
    official = None
    if ws.dataset == "resplan":
        from prism.datasets.resplan import load_resplan_splits
        official = load_resplan_splits(ws.raw_path("resplan.split_json"))
    lookup = {}
    if official:
        for k in ("train", "val", "test"):
            lookup.update({make_id("resplan", x): k for x in official.get(k, [])})
        lookup.update({make_id("resplan", x): "train" for x in official.get("augmented", [])})
    for sid in ids:
        splits[lookup.get(sid) or split_of(sid, int(s["seed"]), float(s["train"]), float(s["val"]))].append(sid)
    return splits, ("official split.json (+hash for unlisted ids)" if official else "hash of the id")


def dataset_profile(store, ids):
    """Per-dataset distributions used to check that RPLAN and ResPlan are aligned."""
    out = {"rooms": Counter(), "unified": Counter(), "relations": Counter(), "connections": Counter(),
           "boundary_area": [], "outline_side": [], "room_area": [], "entrance": Counter()}
    for sid in ids:
        l, g = store.layout(sid), store.graph(sid)
        out["rooms"][len(l["rooms"])] += 1
        out["unified"].update(r["unified_type"] for r in l["rooms"])
        re = room_edges(g)
        out["relations"].update(e["relation"] for e in re)      # both directions: order-independent
        out["connections"].update(e["connection"] for e in re if e["source"] < e["target"])
        out["boundary_area"].append(float(l.get("boundary_area_ratio", 0)))
        b = np.asarray(l.get("boundary") or [[0, 0]], dtype=float)
        out["outline_side"].append(float((b.max(axis=0) - b.min(axis=0)).max()))
        out["room_area"].extend(float(r.get("polygon_area_ratio", r["area_ratio"])) for r in l["rooms"])
        fd = l.get("front_door") or {}
        if fd.get("entrance_room_id") is not None:
            out["entrance"][l["rooms"][fd["entrance_room_id"]]["unified_type"]] += 1
    return out


def normalised(counter, keys):
    """Shares of ``keys`` in a counter (sum to 1)."""
    total = max(1, sum(counter.values()))
    return [counter.get(k, 0) / total for k in keys]


def report_alignment(rep, store, ids):
    """Side-by-side distributions of the member datasets (combined workspace only)."""
    by_ds = {}
    for sid in ids:
        by_ds.setdefault(dataset_of(sid), []).append(sid)
    prof = {d: dataset_profile(store, v) for d, v in by_ds.items()}
    rep.heading("Are the datasets aligned?")
    rep.text("The combined gallery is meaningful only if the same kind of plan looks the same in both datasets. "
             "Large differences below (outline size, room size) can be reduced with resplan.fit_ratio; differences "
             "in room counts or types are real properties of the datasets.")
    rows = []
    for d, p in prof.items():
        rows.append({"dataset": d, "plans": len(by_ds[d]),
                     "mean rooms": float(np.mean([k for k, v in p["rooms"].items() for _ in range(v)])),
                     "outline longest side px (median)": float(np.median(p["outline_side"])),
                     "outline area ratio (median)": float(np.median(p["boundary_area"])),
                     "room area ratio (median)": float(np.median(p["room_area"]))})
    rep.table(rows)
    uni = sorted({k for p in prof.values() for k in p["unified"]})
    rep.figure(grouped_bars(uni, {d: normalised(p["unified"], uni) for d, p in prof.items()},
                            "Unified room types (share of rooms)"))
    nr = sorted({k for p in prof.values() for k in p["rooms"]})
    rep.figure(grouped_bars([str(k) for k in nr], {d: normalised(p["rooms"], nr) for d, p in prof.items()},
                            "Rooms per plan (share of plans)"))
    rep.figure(grouped_bars(RELATIONS, {d: normalised(p["relations"], RELATIONS) for d, p in prof.items()},
                            "Spatial relations (share of directed edges)"))
    ent = sorted({k for p in prof.values() for k in p["entrance"]})
    rep.figure(grouped_bars(ent, {d: normalised(p["entrance"], ent) for d, p in prof.items()},
                            "Room behind the front door (share of plans)"))
    conn = sorted({k for p in prof.values() for k in p["connections"]})
    rep.figure(grouped_bars(conn, {d: normalised(p["connections"], conn) for d, p in prof.items()},
                            "Connection types (share of adjacencies)"))
    rep.note("Connection types differ by construction (RPLAN has no door/wall geometry, so all its edges are "
             "'wall'). That is why graph.connection_features is off by default for the combined dataset.")


def main():
    """Compute statistics, write splits, optionally pack, write the report."""
    parser = build_parser(__doc__)
    parser.add_argument("--pack", action="store_true", help="Write packed files for fast cluster I/O.")
    args = parser.parse_args()
    cfg, ws = setup(args, PHASE)
    if ws.is_combined:
        # Member datasets first (their own splits and statistics), then the merge.
        for member in ws.members():
            print(f"--- {PHASE} on {member.dataset} ---")
            run(member.cfg, member, pack_files=False)
        print(f"--- {PHASE} on combined ---")
    run(cfg, ws, pack_files=args.pack)


def run(cfg, ws, pack_files=False):
    """Statistics, splits (and packing) for one workspace."""
    out = ws.phase_dir(PHASE)
    store = LayoutStore(ws, prefer_packed=False)
    ids = store.graph_ids()
    if not ids:
        raise RuntimeError("No graphs found. Run phases 02-03 first"
                           + (" for every member dataset." if ws.is_combined else "."))

    splits, split_method = make_splits(cfg, ws, ids)
    write_json(ws.splits_file, splits)

    n_rooms, n_edges, types, unified, rels, facing, entrance, hops, widths = [], [], Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), []
    for sid in progress(ids, desc="Statistics"):
        l, g = store.layout(sid), store.graph(sid)
        n_rooms.append(len(l["rooms"]))
        re = room_edges(g)
        n_edges.append(len(re) // 2)
        types.update(r["room_type_name"] for r in l["rooms"])
        unified.update(r["unified_type"] for r in l["rooms"])
        rels.update(e["relation"] for e in re)
        fd = l.get("front_door")
        if fd:
            facing[fd["facing"]] += 1
            widths.append(fd["width_px"])
            if fd.get("entrance_room_id") is not None:
                entrance[l["rooms"][fd["entrance_room_id"]]["room_type_name"]] += 1
        hops.update(nd["door_hops"] for nd in g["nodes"] if nd.get("kind") == "room")

    stats = {
        "num_layouts": len(ids), "splits": {k: len(v) for k, v in splits.items()}, "split_method": split_method,
        "datasets": dict(Counter(dataset_of(i) for i in ids)),
        "rooms_per_plan": {"min": int(np.min(n_rooms)), "mean": float(np.mean(n_rooms)), "max": int(np.max(n_rooms))},
        "edges_per_plan": {"min": int(np.min(n_edges)), "mean": float(np.mean(n_edges)), "max": int(np.max(n_edges))},
        "room_types": dict(types), "unified_types": dict(unified), "relations": dict(rels),
        "door_facing": dict(facing), "entrance_room_types": dict(entrance),
        "door_width_px_mean": float(np.mean(widths)) if widths else None,
        "door_hops": {str(k): v for k, v in sorted(hops.items())},
    }
    write_json(ws.metadata_dir / "dataset_statistics.json", stats)
    if pack_files:
        packed = pack(ws, store, ids)
        stats["packed_layouts"] = packed

    rep = HtmlReport("05 - Statistics and splits", f"{ws.dataset} · {ws.mode}")
    kp = {"layouts": len(ids), "train": len(splits["train"]), "val": len(splits["val"]),
          "test": len(splits["test"]), "mean rooms": stats["rooms_per_plan"]["mean"],
          "mean adjacencies": stats["edges_per_plan"]["mean"], "packed": "yes" if pack_files else "no"}
    if ws.is_combined:
        kp.update({f"{d} plans": n for d, n in stats["datasets"].items()})
    rep.kpis(kp)
    rep.text(f"Split: {split_method}.")
    if ws.is_combined:
        report_alignment(rep, store, ids)
        rep.heading("Pooled statistics")
    rc = Counter(n_rooms)
    rep.figure(bar_chart([str(k) for k in sorted(rc)], [rc[k] for k in sorted(rc)], "Rooms per plan", "plans"))
    if not ws.is_combined:
        rep.figure(bar_chart(list(types.keys()), list(types.values()), "Fine room types", "rooms"))
    rep.figure(bar_chart(list(unified.keys()), list(unified.values()), "Unified room types", "rooms"))
    rep.figure(bar_chart(RELATIONS, [rels.get(r, 0) for r in RELATIONS], "Spatial relations (directed edges, both directions)", "edges"))
    rep.figure(bar_chart(list(entrance.keys()), list(entrance.values()), "Room behind the front door", "plans"))
    rep.figure(bar_chart(list(facing.keys()), list(facing.values()), "Front door facing", "plans"))
    hk = sorted(hops)
    rep.figure(bar_chart([str(k) for k in hk], [hops[k] for k in hk], "Hops from the entrance room (-1 = unreachable)", "rooms"))
    rep.json_block(stats)
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
