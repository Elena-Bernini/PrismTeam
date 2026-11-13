"""Phase 04 - render one semantic image per layout (input of the vision pipeline).

Rooms are filled with the colour of their type, walls are dark lines and the
front door is a thick yellow segment on the exterior wall. The same renderer
is reused in phase 10 to draw partial queries (masked rooms in grey).

    python scripts/04_render_images.py --mode preview
    python scripts/04_render_images.py --mode full
    python scripts/04_render_images.py --mode full --set render.size=224
"""

import sys
from pathlib import Path

# Make `import prism` work however the script is launched (python scripts/x.py,
# VS Code "Run Python File", python -m scripts.x) even without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.common.cli import Timer, build_parser, setup
from prism.common.io import clean_dir, progress
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import image_grid
from prism.datasets.schema import Taxonomy
from prism.datasets.store import LayoutStore
from prism.vision.render import render_layout

PHASE = "04_render_images"


def main():
    """Render every layout to PNG and show a grid of renders next to the snapshots."""
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
    store = LayoutStore(ws, prefer_packed=False)
    ids = store.ids()
    clean_dir(ws.images_dir)
    with Timer("rendering"):
        for sid in progress(ids, desc="Rendering"):
            render_layout(store.layout(sid), taxonomy, cfg["render"]).save(ws.images_dir / f"{sid}.png")

    rep = HtmlReport("04 - Semantic renders", f"{ws.dataset} · {ws.mode}")
    rep.kpis({"images": len(ids), "size": f"{cfg['render']['size']} px", "front door drawn": cfg["render"]["draw_front_door"]})
    images, titles = [], []
    snap_dir = ws.raw_path("rplan.snapshot_dir") if ws.dataset == "rplan" else None
    from PIL import Image
    for sid in ids[:12]:
        if snap_dir is not None:
            sp = snap_dir / f"{store.layout(sid)['source_id']}.png"
            images.append(Image.open(sp).convert("RGB") if sp.exists() else None)
            titles.append(f"{sid} snapshot")
        images.append(Image.open(ws.images_dir / f"{sid}.png"))
        titles.append(f"{sid} render")
    rep.figure(image_grid(images, titles, ncols=6, cell=2.2),
               "Original snapshot vs our render (pairs)" if snap_dir is not None else "Our renders")

    # Legend of colours.
    palette = cfg["render"].get("palette", "unified")
    seen = {}
    for sid in ids:
        for r in store.layout(sid)["rooms"]:
            seen.setdefault(r["room_type_name"], r)
    legend = [{"fine type": k, "unified": r["unified_type"], "colour": taxonomy.color(r, palette)}
              for k, r in sorted(seen.items())]
    rep.heading("Colour legend")
    rep.table(legend)
    rep.heading("Partial-query style")
    l0 = store.layout(ids[0])
    visible = [r["id"] for r in l0["rooms"]][: max(1, len(l0["rooms"]) // 2)]
    fig = image_grid([render_layout(l0, taxonomy, cfg["render"]),
                      render_layout(l0, taxonomy, cfg["render"], visible_room_ids=visible)],
                     ["full layout", "partial query (grey = unknown)"], ncols=2, cell=3)
    rep.figure(fig, "How phase 10 renders a partial layout")
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
