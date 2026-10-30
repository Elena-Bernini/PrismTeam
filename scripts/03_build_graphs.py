"""Phase 03 - build graph JSON files with recomputed adjacency, spatial relations and front door.

For every layout:

1. adjacency from the room polygons (dilate by ``geometry.wall_dilation_px``,
   keep pairs sharing at least ``geometry.min_contact_px`` of wall);
2. spatial relation of every adjacent pair from the bounding boxes
   (containment -> axis overlap -> diagonal), stored in BOTH directions;
3. a virtual ``FrontDoor`` node linked to the entrance room, plus per-room
   hop distance from the entrance (see docs/relations.md).

The report compares our adjacency with Graph2Plan's edge list and shows our
relation labels against the raw third column of ``sample.edge``.

    python scripts/03_build_graphs.py --mode preview
    python scripts/03_build_graphs.py --mode full
"""

from collections import Counter, defaultdict

import numpy as np

import scripts._bootstrap as _bootstrap  # noqa: F401

from prism.common.cli import Timer, build_parser, setup
from prism.common.io import clean_dir, progress, read_json, write_json
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import draw_graph, plt
from prism.datasets.schema import G2P_RELATION_GUESS, RELATIONS, Taxonomy
from prism.datasets.store import LayoutStore
from prism.geometry.adjacency import geometric_adjacency
from prism.geometry.relations import spatial_relation
from prism.graph.build import _room_polygon, build_graph, g2p_pairs

PHASE = "03_build_graphs"


def adjacency_agreement(layouts, cfg, radius, min_contact):
    """Precision / recall of geometric adjacency against the Graph2Plan edge list."""
    tp = fp = fn = 0
    for l in layouts:
        polys = [_room_polygon(r) for r in l["rooms"]]
        geo = set(geometric_adjacency(polys, [r["bbox"] for r in l["rooms"]], int(l["coordinate_system"]["width"]),
                                      radius, min_contact))
        ref = g2p_pairs(l)
        tp += len(geo & ref)
        fp += len(geo - ref)
        fn += len(ref - geo)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"wall_dilation_px": radius, "min_contact_px": min_contact, "precision_vs_g2p": prec,
            "recall_vs_g2p": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec)}


def main():
    """Build graphs, compute diagnostics and write the report."""
    args = build_parser(__doc__).parse_args()
    cfg, ws = setup(args, PHASE)
    out = ws.phase_dir(PHASE)
    taxonomy = Taxonomy(cfg, ws.dataset)
    store = LayoutStore(ws, prefer_packed=False)
    ids = store.ids()
    if not ids:
        raise RuntimeError(f"No layouts in {ws.layouts_dir}. Run phase 02 first.")
    clean_dir(ws.graphs_dir)

    rel_counts, conn_counts, deg = Counter(), Counter(), []
    raw_vs_ours = defaultdict(Counter)
    totals = Counter()
    ct, ot = float(cfg["geometry"]["containment_threshold"]), float(cfg["geometry"]["axis_overlap_threshold"])
    with Timer("graph construction"):
        for sid in progress(ids, desc="Building graphs"):
            layout = store.layout(sid)
            graph = build_graph(layout, cfg)
            write_json(ws.graphs_dir / f"{sid}.json", graph)
            for e in graph["edges"]:
                conn_counts[e["connection"]] += 1
                if e["connection"] != "entrance" and e["source"] < e["target"]:
                    rel_counts[e["relation"]] += 1
            deg.append(len([e for e in graph["edges"] if e["connection"] != "entrance"]) / max(1, len(layout["rooms"])))
            for k, v in graph["comparison_with_g2p"].items():
                totals[k] += v
            # Our relation for each Graph2Plan (u, v, raw) triple, in the same direction u -> v.
            rooms = layout["rooms"]
            for e in layout.get("source_edges", []):
                u, v = int(e["source"]), int(e["target"])
                if 0 <= u < len(rooms) and 0 <= v < len(rooms) and u != v:
                    raw_vs_ours[int(e["raw_value"])][spatial_relation(rooms[u]["bbox"], rooms[v]["bbox"], ct, ot)] += 1

    prec = totals["common_pairs"] / max(1, totals["geometric_pairs"])
    rec = totals["common_pairs"] / max(1, totals["g2p_pairs"])
    stats = {"graphs": len(ids), "relation_counts_undirected": dict(rel_counts), "connection_counts_directed": dict(conn_counts),
             "adjacency_vs_g2p": {"precision": prec, "recall": rec, **totals},
             "raw_column_vs_our_relation": {str(k): dict(v) for k, v in sorted(raw_vs_ours.items())}}
    write_json(ws.metadata_dir / "graph_statistics.json", stats)

    rep = HtmlReport("03 - Graphs: adjacency, spatial relations, front door", f"{ws.dataset} · {ws.mode}")
    rep.kpis({"graphs": len(ids), "mean degree": float(np.mean(deg)), "adjacency precision vs G2P": prec,
              "adjacency recall vs G2P": rec, "door edges": conn_counts.get("entrance", 0) // 2})
    rep.heading("How relations are computed")
    rep.text("relation(u, v) = position of room u relative to room v, from bounding boxes: (1) containment >= "
             f"{ct:.0%} -> inside / surrounding; (2) projection overlap >= {ot:.0%} on x -> above / below, on y -> "
             "left_of / right_of; (3) otherwise diagonal (left_above, ...). Each adjacency is stored in both "
             "directions with the inverse label. Full description in docs/relations.md.")
    rep.heading("Relation distribution (undirected pairs)")
    rep.table([{"relation": r, "pairs": rel_counts.get(r, 0)} for r in RELATIONS])

    if raw_vs_ours:
        rep.heading("Graph2Plan raw third column vs our relation")
        rep.text("Rows: value of the third column of sample.edge. Columns: our relation for the same (u, v). If the "
                 "column encoded a relation, every row would be concentrated on one column. 'best share' is the "
                 "fraction of the row taken by its most common relation.")
        rows = []
        for raw, cnt in sorted(raw_vs_ours.items()):
            total = sum(cnt.values())
            best, best_n = cnt.most_common(1)[0]
            row = {"raw": raw, "Pier's label": G2P_RELATION_GUESS.get(raw, "-"), "n": total,
                   "best match": best, "best share": best_n / total}
            row.update({r: cnt.get(r, 0) for r in RELATIONS})
            rows.append(row)
        rep.table(rows)

    if ws.mode == "preview":
        rep.heading("Sensitivity of adjacency to the dilation radius (preview subset)")
        sample_layouts = [store.layout(s) for s in ids[:150]]
        mc = int(cfg["geometry"]["min_contact_px"])
        rep.table([adjacency_agreement(sample_layouts, cfg, r, mc) for r in (1, 2, 3, 4, 5, 6)])
        rep.text("Pick the radius where precision and recall against Graph2Plan are both high; Graph2Plan also adds "
                 "door-based pairs that pure geometry may not see. Set it with --set geometry.wall_dilation_px=N.")

    rep.heading("Examples")
    rep.text("Blue lines: room adjacency with label 'u REL v' (room u relative to room v; L=left_of, R=right_of, "
             "A=above, B=below, LA/RA/LB/RB = diagonals, IN/SUR = inside/surrounding). Yellow square: FrontDoor "
             "node, dashed orange: entrance edge.")
    for sid in ids[:12]:
        layout = store.layout(sid)
        graph = read_json(ws.graphs_dir / f"{sid}.json")
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
        c = graph["comparison_with_g2p"]
        draw_graph(ax, layout, graph, taxonomy,
                   title=f"{sid} · geo pairs {c['geometric_pairs']} · G2P pairs {c['g2p_pairs']} · common {c['common_pairs']}")
        rep.figure(fig, f"Graph {sid}")
    rep.heading("Example graph JSON")
    rep.json_block(read_json(ws.graphs_dir / f"{ids[0]}.json"), max_chars=4000)
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
