"""Phase 08 - vision pipeline: embed every floor-plan image and build the vector index.

Image source (``vision.image_source``): ``render`` (phase 04, default: same
style as partial queries and as ResPlan renders), ``snapshot`` (Graph2Plan
snapshots, Pier's previous input) or ``img`` (Graph2Plan boundary images,
Elena's previous input).

Encoders (``vision.encoders`` or ``--encoders``): colorhist (CPU baseline),
resnet18/50, dinov2_small/base/large, clip_b32, timm:<name>. Deep encoders run
on CUDA / MPS / CPU (``--device``). Model weights are downloaded once into
``HF_HOME`` / ``TORCH_HOME`` (on the cluster run cluster/predownload_models.py
on the login node first).

    python scripts/08_vision_embed_index.py --mode preview --encoders colorhist dinov2_base
    python scripts/08_vision_embed_index.py --mode full --encoders dinov2_base resnet50 --device cuda
"""

import random

import numpy as np
from PIL import Image

import scripts._bootstrap as _bootstrap  # noqa: F401

from prism.common.cli import Timer, build_parser, setup
from prism.common.device import describe_device, empty_cache, resolve_device, torch_available
from prism.common.io import progress
from prism.common.report import HtmlReport, finalize_report
from prism.datasets.store import LayoutStore, load_splits, resolve_image_path
from prism.retrieval.index import VectorIndex
from prism.retrieval.visual import retrieval_figure
from prism.vision.encoders import build_encoder, index_name, needs_torch

PHASE = "08_vision_embed_index"
CHUNK = 512  # images loaded in memory at once


def load_image(store, ws, sid, source):
    """Load the image of ``sid`` for the chosen source (renders may come from the packed zip)."""
    if source == "render":
        return store.image(sid)
    return Image.open(resolve_image_path(ws, sid, source)).convert("RGB")


def main():
    """Embed images with every requested encoder and save the indexes."""
    parser = build_parser(__doc__)
    parser.add_argument("--encoders", nargs="+", default=None)
    parser.add_argument("--image-source", default=None, help="render | snapshot | img")
    args = parser.parse_args()
    cfg, ws = setup(args, PHASE)
    out = ws.phase_dir(PHASE)
    vcfg, rcfg = cfg["vision"], cfg["retrieval"]
    source = args.image_source or vcfg["image_source"]
    encoders = args.encoders or list(vcfg["encoders"])
    store = LayoutStore(ws)
    splits = load_splits(ws)
    ids = store.graph_ids()
    if source != "render":
        missing = [s for s in ids if not resolve_image_path(ws, s, source).exists()]
        if missing:
            print(f"  [warn] {len(missing)} ids have no '{source}' image and are skipped")
            ids = [s for s in ids if s not in set(missing)]
    device = resolve_device(cfg["runtime"]["device"])

    rep = HtmlReport("08 - Vision pipeline: embeddings and index", f"{ws.dataset} · {ws.mode}")
    built = {}
    for name in encoders:
        if needs_torch(name) and not torch_available():
            print(f"  [skip] {name}: PyTorch not installed")
            rep.note(f"{name} skipped: PyTorch is not installed on this machine.", warn=True)
            continue
        with Timer(f"{name} on {source}") as t:
            encoder = build_encoder(name, vcfg, device)
            chunks = []
            for start in progress(range(0, len(ids), CHUNK), f"{name} chunks", total=(len(ids) + CHUNK - 1) // CHUNK):
                imgs = [load_image(store, ws, s, source) for s in ids[start:start + CHUNK]]
                chunks.append(encoder.encode(imgs))
            X = np.concatenate(chunks, axis=0)
            del encoder
            empty_cache(device)
        iname = index_name(name, source)
        index = VectorIndex(ids, X, rcfg["metric"], rcfg["backend"], rcfg["index"], None,
                            meta={"pipeline": "vision", "encoder": name, "image_source": source,
                                  "device": device, "seconds": t.elapsed})
        index.save(ws.index_dir(iname))
        built[iname] = index
        print(f"  saved index {iname}: {X.shape}")

    rep.kpis({"images": len(ids), "source": source, "device": describe_device(device),
              "indexes": ", ".join(built) or "none"})
    rng = random.Random(int(cfg["runtime"]["seed"]))
    pool = [s for s in splits["test"] if s in set(ids)] or ids
    queries = rng.sample(pool, min(int(rcfg["grids"]), len(pool)))
    for iname, index in built.items():
        rep.heading(f"{iname}  (dim {index.raw.shape[1]})")
        for q in queries:
            rep.figure(retrieval_figure(store, q, index.search_by_id(q, 7), iname), f"{iname} · query {q}")
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
