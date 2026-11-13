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

import sys
from pathlib import Path

# Make `import prism` work however the script is launched (python scripts/x.py,
# VS Code "Run Python File", python -m scripts.x) even without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.common.cli import Timer, build_parser, setup
from prism.common.io import clean_dir, progress, read_json, write_json
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import CONNECTION_COLORS, draw_elements, draw_graph, plt
from prism.datasets.schema import G2P_RELATION_GUESS, RELATIONS, Taxonomy
from prism.datasets.store import LayoutStore
from prism.geometry.adjacency import RoomMasks, geometric_adjacency
from prism.geometry.polygon import room_shapes
from prism.geometry.relations import spatial_relation
from prism.graph.build import build_graph, dilation_radius, source_pairs

PHASE = "03_build_graphs"


def adjacency_agreement(layouts, cfg, radius, min_contact):
    """Precision / recall of geometric adjacency against the dataset's own edge list (RPLAN: Graph2Plan)."""
    tp = fp = fn = 0
    for l in layouts:
        masks = RoomMasks([room_shapes(r) for r in l["rooms"]], int(l["coordinate_system"]["width"]), radius)
        geo = set(geometric_adjacency(masks, [r["bbox"] for r in l["rooms"]], min_contact))
        ref = source_pairs(l)
        tp += len(geo & ref)
        fp += len(geo - ref)
        fn += len(ref - geo)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"wall_dilation_px": radius, "min_contact_px": min_contact, "precision_vs_source": prec,
            "recall_vs_source": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec)}


def main():
    """Build graphs, compute diagnostics and write the report."""
    args = build_parser(__doc__).parse_args()
    cfg, ws = setup(args, PHASE)
    for member in ws.members():          # --dataset combined: run once per member dataset
        if ws.is_combined:
            print(f"--- {PHASE} on {member.dataset} ---")
        run(member.cfg, member)


def run(cfg, ws):
    """Run the phase on one dataset (called once per member dataset for --dataset combined)."""
    out = ws.phase_dir(PHASE)
    src_name = {"rplan": "Graph2Plan", "resplan": "ResPlan graph"}.get(ws.dataset, "source")
    taxonomy = Taxonomy(cfg, ws.dataset)
    store = LayoutStore(ws, prefer_packed=False)
    ids = store.ids()
    if not ids:
        raise RuntimeError(f"No layouts in {ws.layouts_dir}. Run phase 02 first.")
    clean_dir(ws.graphs_dir)

    rel_counts, conn_counts, deg, radii = Counter(), Counter(), [], []
    conn_vs_source = defaultdict(Counter)
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
                if e["connection"] != "entrance":        # both directions: independent of room numbering
                    rel_counts[e["relation"]] += 1
            deg.append(len([e for e in graph["edges"] if e["connection"] != "entrance"]) / max(1, len(layout["rooms"])))
            for k, v in graph["comparison_with_source"].items():
                totals[k] += v
            radii.append(dilation_radius(layout, cfg))
            ours = {(min(e["source"], e["target"]), max(e["source"], e["target"])): e["connection"]
                    for e in graph["edges"] if e["connection"] != "entrance"}
            # Our relation for each source (u, v, raw) triple, in the same direction u -> v (RPLAN only),
            # and our connection type vs the source edge type (ResPlan graph, when present).
            rooms = layout["rooms"]
            for e in layout.get("source_edges", []):
                u, v = int(e["source"]), int(e["target"])
                if 0 <= u < len(rooms) and 0 <= v < len(rooms) and u != v:
                    if int(e.get("raw_value", -1)) >= 0:
                        raw_vs_ours[int(e["raw_value"])][spatial_relation(rooms[u]["bbox"], rooms[v]["bbox"], ct, ot)] += 1
                    if e.get("source_type"):
                        conn_vs_source[e["source_type"]][ours.get((min(u, v), max(u, v)), "not adjacent")] += 1

    prec = totals["common_pairs"] / max(1, totals["geometric_pairs"])
    rec = totals["common_pairs"] / max(1, totals["source_pairs"])
    has_source = totals["source_pairs"] > 0
    stats = {"graphs": len(ids), "relation_counts_directed": dict(rel_counts), "connection_counts_directed": dict(conn_counts),
             "adjacency_vs_source": {"source": src_name, "precision": prec, "recall": rec, **totals},
             "connection_vs_source_type": {k: dict(v) for k, v in conn_vs_source.items()},
             "raw_column_vs_our_relation": {str(k): dict(v) for k, v in sorted(raw_vs_ours.items())}}
    write_json(ws.metadata_dir / "graph_statistics.json", stats)

    rep = HtmlReport("03 - Graphs: adjacency, spatial relations, front door", f"{ws.dataset} · {ws.mode}")
    kp = {"graphs": len(ids), "mean degree": float(np.mean(deg)), "front-door edges": conn_counts.get("entrance", 0) // 2,
          "dilation radius px": f"{min(radii)}-{max(radii)}" if min(radii) != max(radii) else str(radii[0])}
    if has_source:
        kp.update({f"adjacency precision vs {src_name}": prec, f"adjacency recall vs {src_name}": rec})
    rep.kpis(kp)
    if not has_source:
        rep.note("No source edge list in this dataset (ResPlan GitHub release has no plan['graph']; the Kaggle "
                 "release has one), so adjacency cannot be compared with a reference here.")
    rep.heading("How relations are computed")
    rep.text("relation(u, v) = position of room u relative to room v, from bounding boxes: (1) containment >= "
             f"{ct:.0%} -> inside / surrounding; (2) projection overlap >= {ot:.0%} on x -> above / below, on y -> "
             "left_of / right_of; (3) otherwise diagonal (left_above, ...). Each adjacency is stored in both "
             "directions with the inverse label. Full description in docs/relations.md.")
    rep.heading("Relation distribution (directed edges, both directions)")
    rep.table([{"relation": r, "edges": rel_counts.get(r, 0)} for r in RELATIONS])
    rep.heading("Connection types (directed edges)")
    rep.text("wall / door / arch (open passage) / window come from the wall, door and window geometry (ResPlan). "
             "RPLAN has no such geometry, so its room edges are all 'wall'. entrance = edges to the FrontDoor node.")
    rep.table([{"connection": k, "edges": v} for k, v in conn_counts.most_common()])
    if conn_vs_source:
        rep.heading("Our connection type vs ResPlan's own edge type")
        rep.text("Rows: edge type in plan['graph'] (via_door, adjacency, via_window, via_opening ...). Columns: our "
                 "label for the same room pair. A diagonal-like table means the geometric classification agrees.")
        rep.table([{"ResPlan type": k, **dict(v)} for k, v in conn_vs_source.items()])

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

    if ws.mode == "preview" and ws.dataset == "rplan" and has_source:
        rep.heading("Sensitivity of adjacency to the dilation radius (preview subset)")
        sample_layouts = [store.layout(s) for s in ids[:150]]
        mc = int(cfg["geometry"]["min_contact_px"])
        rep.table([adjacency_agreement(sample_layouts, cfg, r, mc) for r in (1, 2, 3, 4, 5, 6)])
        rep.text("Pick the radius where precision and recall against Graph2Plan are both high; Graph2Plan also adds "
                 "door-based pairs that pure geometry may not see. Set it with --set geometry.wall_dilation_px=N.")

    rep.heading("Examples")
    rep.text("Edges are labelled 'u REL v' (room u relative to room v; L=left_of, R=right_of, A=above, B=below, "
             "LA/RA/LB/RB = diagonals, IN/SUR = inside/surrounding). Edge colour = connection type: "
             + ", ".join(f"{k} {v}" for k, v in CONNECTION_COLORS.items())
             + ". Yellow square: FrontDoor node.")
    for sid in ids[:12]:
        layout = store.layout(sid)
        graph = read_json(ws.graphs_dir / f"{sid}.json")
        c = graph["comparison_with_source"]
        title = f"{sid} · geo pairs {c['geometric_pairs']}"
        if c["source_pairs"]:
            title += f" · {src_name} pairs {c['source_pairs']} · common {c['common_pairs']}"
        if layout.get("walls"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            draw_elements(axes[0], layout, title="walls / doors / windows")
            draw_graph(axes[1], layout, graph, taxonomy, title=title)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
            draw_graph(ax, layout, graph, taxonomy, title=title)
        rep.figure(fig, f"Graph {sid}")
    rep.heading("Example graph JSON")
    rep.json_block(read_json(ws.graphs_dir / f"{ids[0]}.json"), max_chars=4000)
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
