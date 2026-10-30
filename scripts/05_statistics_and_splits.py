"""Phase 05 - dataset statistics, train/val/test split and (optional) packing.

- Statistics over layouts and graphs (rooms, types, relations, front door).
- Deterministic split: each id is hashed with ``splits.seed``, so the same plan
  always falls in the same split, in preview and in full mode, on every machine.
  The GNN trains on ``train``, early-stops on ``val``; retrieval queries come
  from ``retrieval.query_split`` (default ``test``) and search the whole gallery.
- ``--pack`` writes ``packed/layouts.jsonl.gz``, ``graphs.jsonl.gz`` and
  ``images.zip``: three large files instead of ~240k small ones, which is much
  faster on the cluster's BeeGFS ``/work`` (see cluster/README_cluster.md).

    python scripts/05_statistics_and_splits.py --mode preview
    python scripts/05_statistics_and_splits.py --mode full --pack
"""

import hashlib
import zipfile
from collections import Counter

import numpy as np

import scripts._bootstrap as _bootstrap  # noqa: F401

from prism.common.cli import build_parser, setup
from prism.common.io import ensure_dir, progress, write_json, write_jsonl_gz
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import bar_chart
from prism.datasets.schema import RELATIONS
from prism.datasets.store import LayoutStore
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


def main():
    """Compute statistics, write splits, optionally pack, write the report."""
    parser = build_parser(__doc__)
    parser.add_argument("--pack", action="store_true", help="Write packed files for fast cluster I/O.")
    args = parser.parse_args()
    cfg, ws = setup(args, PHASE)
    out = ws.phase_dir(PHASE)
    store = LayoutStore(ws, prefer_packed=False)
    ids = store.graph_ids()
    if not ids:
        raise RuntimeError("No graphs found. Run phases 02-03 first.")

    s = cfg["splits"]
    splits = {"train": [], "val": [], "test": []}
    for sid in ids:
        splits[split_of(sid, int(s["seed"]), float(s["train"]), float(s["val"]))].append(sid)
    write_json(ws.splits_file, splits)

    n_rooms, n_edges, types, unified, rels, facing, entrance, hops, widths = [], [], Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), []
    for sid in progress(ids, desc="Statistics"):
        l, g = store.layout(sid), store.graph(sid)
        n_rooms.append(len(l["rooms"]))
        re = room_edges(g)
        n_edges.append(len(re) // 2)
        types.update(r["room_type_name"] for r in l["rooms"])
        unified.update(r["unified_type"] for r in l["rooms"])
        rels.update(e["relation"] for e in re if e["source"] < e["target"])
        fd = l.get("front_door")
        if fd:
            facing[fd["facing"]] += 1
            widths.append(fd["width_px"])
            if fd.get("entrance_room_id") is not None:
                entrance[l["rooms"][fd["entrance_room_id"]]["room_type_name"]] += 1
        hops.update(nd["door_hops"] for nd in g["nodes"] if nd.get("kind") == "room")

    stats = {
        "num_layouts": len(ids), "splits": {k: len(v) for k, v in splits.items()},
        "rooms_per_plan": {"min": int(np.min(n_rooms)), "mean": float(np.mean(n_rooms)), "max": int(np.max(n_rooms))},
        "edges_per_plan": {"min": int(np.min(n_edges)), "mean": float(np.mean(n_edges)), "max": int(np.max(n_edges))},
        "room_types": dict(types), "unified_types": dict(unified), "relations": dict(rels),
        "door_facing": dict(facing), "entrance_room_types": dict(entrance),
        "door_width_px_mean": float(np.mean(widths)) if widths else None,
        "door_hops": {str(k): v for k, v in sorted(hops.items())},
    }
    write_json(ws.metadata_dir / "dataset_statistics.json", stats)
    if args.pack:
        packed = pack(ws, store, ids)
        stats["packed_layouts"] = packed

    rep = HtmlReport("05 - Statistics and splits", f"{ws.dataset} · {ws.mode}")
    rep.kpis({"layouts": len(ids), "train": len(splits["train"]), "val": len(splits["val"]),
              "test": len(splits["test"]), "mean rooms": stats["rooms_per_plan"]["mean"],
              "mean adjacencies": stats["edges_per_plan"]["mean"], "packed": "yes" if args.pack else "no"})
    rc = Counter(n_rooms)
    rep.figure(bar_chart([str(k) for k in sorted(rc)], [rc[k] for k in sorted(rc)], "Rooms per plan", "plans"))
    rep.figure(bar_chart(list(types.keys()), list(types.values()), "Fine room types", "rooms"))
    rep.figure(bar_chart(RELATIONS, [rels.get(r, 0) for r in RELATIONS], "Spatial relations (undirected pairs)", "pairs"))
    rep.figure(bar_chart(list(entrance.keys()), list(entrance.values()), "Room behind the front door", "plans"))
    rep.figure(bar_chart(list(facing.keys()), list(facing.values()), "Front door facing", "plans"))
    hk = sorted(hops)
    rep.figure(bar_chart([str(k) for k in hk], [hops[k] for k in hk], "Hops from the entrance room (-1 = unreachable)", "rooms"))
    rep.json_block(stats)
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
