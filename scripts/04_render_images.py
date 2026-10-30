"""Phase 04 - render one semantic image per layout (input of the vision pipeline).

Rooms are filled with the colour of their type, walls are dark lines and the
front door is a thick yellow segment on the exterior wall. The same renderer
is reused in phase 10 to draw partial queries (masked rooms in grey).

    python scripts/04_render_images.py --mode preview
    python scripts/04_render_images.py --mode full
    python scripts/04_render_images.py --mode full --set render.size=224
"""

import scripts._bootstrap as _bootstrap  # noqa: F401

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
        sp = snap_dir / f"{sid}.png" if snap_dir else None
        images.append(Image.open(sp).convert("RGB") if sp is not None and sp.exists() else None)
        titles.append(f"{sid} snapshot")
        images.append(Image.open(ws.images_dir / f"{sid}.png"))
        titles.append(f"{sid} render")
    rep.figure(image_grid(images, titles, ncols=6, cell=2.2), "Original snapshot vs our render (pairs)")

    # Legend of colours.
    legend = [{"id": t, "fine type": taxonomy.name(t), "unified": taxonomy.unified(t), "colour": taxonomy.color(t)}
              for t in taxonomy.fine_ids]
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
