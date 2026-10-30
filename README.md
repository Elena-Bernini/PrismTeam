# Prism — CVCS Phase 1 (Maticad)

**Vision vs graph representations for semantic floor-plan retrieval.**

One codebase for the whole Phase 1 workflow:

```
RPLAN pickle ─► 01 inspect ─► 02 layout JSON ─► 03 graphs ─► 04 renders ─► 05 stats/splits
                                                    │              │
                                     06 GNN training│              │
                                                    ▼              ▼
                                  07 graph index (descriptor, GNN)   08 vision index (DINOv2, ResNet, CLIP, colorhist)
                                                    └──────┬───────┘
                                   09 retrieval (complete queries + hybrid RRF)
                                   10 partial retrieval (masked queries + compatibility re-ranking)
                                   11 evaluation (architectural metrics, Recall@K, front door)
                                   12 comparison across pipelines / runs
```

It merges Pier's code (JSON preprocessing, descriptor baseline, retrieval,
partial retrieval, evaluation) and Elena's code (DINOv2 embeddings, GraphSAGE
auto-encoder, incomplete retrieval, hybrid RRF). `MIGRATION.md` lists where
each old file went.

Main changes:
- spatial relations are **recomputed** in a deterministic, documented way;
- the **front door** is represented end to end;
- one schema for RPLAN and ResPlan.

See `docs/relations.md` for both.

---

## 1. Installation

Python 3.9–3.12. Create a virtual environment in the project folder.

### Pier: Windows + NVIDIA GPU

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements\base.txt -r requirements\vision.txt
python -m pip install -e .
python scripts\00_check_env.py          # should say: device cuda (NVIDIA ...)
```

If `cu124` does not match the driver, pick the right command on the PyTorch
"Get started" page (`nvidia-smi` shows the supported CUDA version).

### Elena: macOS + Apple M1

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision          # includes Apple MPS support
python -m pip install -r requirements/base.txt -r requirements/vision.txt
python -m pip install -e .
python scripts/00_check_env.py                   # should say: device mps
```

The code sets `PYTORCH_ENABLE_MPS_FALLBACK=1` (ops not yet supported on MPS run
on the CPU) and `KMP_DUPLICATE_LIB_OK=TRUE` (avoids the OpenMP clash between
torch and faiss on macOS) automatically. If `faiss-cpu` does not install,
remove it from `requirements/base.txt`: the code falls back to exact NumPy
search with the same results.

### Cluster (AImageLab-HPC)

See [`cluster/README_cluster.md`](cluster/README_cluster.md). In short:
`bash cluster/setup_env.sh` once, then `sbatch cluster/slurm/preview_all.sbatch`.

### Without a GPU / without PyTorch

Phases 01–05, the descriptor graph index, the `colorhist` vision index and
phases 09–12 need only NumPy, Pillow and Matplotlib. Phase 06 (GNN) and the
deep vision encoders need PyTorch, and are skipped with a clear message when
it is missing. `--device cpu` forces the CPU everywhere.

---

## 2. Where to put the data

```
prism/
└── data/
    ├── raw/
    │   ├── rplan/                                   ← unzip the Graph2Plan / RPLAN release here
    │   │   ├── Interface/static/Data/
    │   │   │   ├── data_train_converted.pkl         ← REQUIRED (main input, ~80k plans)
    │   │   │   ├── snapshot_train/*.png             ← optional (visual checks, --image-source snapshot)
    │   │   │   └── Img/*.png                        ← optional (--image-source img)
    │   │   └── Network/data/data_{train,valid,test}.mat   ← optional, not used yet
    │   └── resplan/                                 ← ResPlan files (*.pkl) when available
    └── processed/<dataset>/<preview|full>/          ← written by phases 02–05 (do not edit)
        ├── layouts_json/   graphs_json/   images/
        ├── metadata/       splits/splits.json
        └── packed/         (phase 05 --pack: layouts.jsonl.gz, graphs.jsonl.gz, images.zip)

artifacts/<dataset>/<mode>/          ← models, embeddings, indexes (all .npy / .faiss / .pt files)
    ├── gnn/model.pt  history.json
    └── indexes/<name>/vectors.npy  ids.json  meta.json  [index.faiss] [scaler.json]
outputs/<dataset>/<mode>/<NN_phase>/report.html   ← one visual report per phase
```

The old `.pkl` files (`dataset_index.pkl`, `vision_embeddings*.pkl`,
`floorplan_graphs.pkl`, `graph_embeddings.pkl`, `faiss_names.pkl`, the
`.faiss` indexes, `gnn_encoder_weights.pt`) are **not needed**: phases 03, 06,
07 and 08 regenerate their equivalents on the whole dataset. Paths can be
changed in `configs/env/local.yaml` or with `--set paths.data_root=...`.

No real data yet? `python scripts/make_synthetic_dataset.py` writes a tiny fake
dataset into `data_synthetic/` to try the pipeline (see section 5).

---

## 3. Preview first, then full

**Every phase has two modes:**

- `--mode preview` (default): runs on a fixed random subset (`--n`, default
  200) and writes a visual `report.html` (figures, tables, examples).
  The subset is chosen in phase 02 and reused by every later phase, so you can
  run the whole chain quickly.
- `--mode full`: the same command on the whole dataset.

Preview and full write to different folders (`.../preview/` and
`.../full/`), so they never overwrite each other.

```bash
python scripts/03_build_graphs.py --mode preview        # look at outputs/rplan/preview/03_build_graphs/report.html
python scripts/03_build_graphs.py --mode full           # happy? run it on everything
```

Run everything in one go:

```bash
python scripts/run_all.py --mode preview                # 01 → 12
python scripts/run_all.py --mode full --from 2 --to 5   # a range of phases
```

---

## 4. The phases

| # | Script | What it does | What to look at in the report |
|---|---|---|---|
| 00 | `00_check_env.py` | libraries, device (cuda/mps/cpu), data files | which phases can run on this machine |
| 01 | `01_inspect_dataset.py` | pickle structure, **third column of `edge`**, front door vs snapshot, polygon/box alignment | does the cyan door segment sit on the yellow line? |
| 02 | `02_convert_to_layout_json.py` | one layout JSON per plan (Pier's schema + taxonomy fix + front door) | snapshot vs JSON render, door facing / entrance room |
| 03 | `03_build_graphs.py` | adjacency from polygons, **new relations**, FrontDoor node | graphs drawn over plans, agreement with Graph2Plan, radius sensitivity |
| 04 | `04_render_images.py` | semantic renders (rooms by type, yellow door) | renders vs snapshots, partial-query style |
| 05 | `05_statistics_and_splits.py` | statistics, deterministic train/val/test split, `--pack` | distributions of rooms, relations, door |
| 06 | `06_train_gnn.py` | GCN / GraphSAGE / GAT with edge features; GAE, contrastive or both | loss curves, PCA of embeddings |
| 07 | `07_graph_embed_index.py` | graph indexes: `descriptor`, `gnn` | nearest neighbours of sample queries |
| 08 | `08_vision_embed_index.py` | vision indexes: `colorhist`, `resnet18/50`, `dinov2_*`, `clip_b32`, `timm:*` | nearest neighbours of sample queries |
| 09 | `09_retrieval_full.py` | complete-plan queries on every index + hybrid RRF | pipelines side by side for the same query |
| 10 | `10_retrieval_partial.py` | masked queries (random, remove type/bedroom/smallest/largest, keep connected) → pool → compatibility re-ranking | is the original plan recovered? re-ranking gain |
| 11 | `11_evaluate.py` | architectural metrics, Recall@K, Hit@K, MRR, front-door metrics, optional GED | summary table, metric vs rank |
| 12 | `12_compare_pipelines.py` | merges phases 10–11 and other runs | final comparison |

Common options for every script:

```
--mode preview|full   --n 500   --env local|cluster   --device auto|cpu|cuda|mps
--set key.subkey=value   (any value in configs/default.yaml)   --config extra.yaml
```

Examples:

```bash
python scripts/06_train_gnn.py --mode full --set graph.model=gat --set graph.objective=both
python scripts/08_vision_embed_index.py --mode full --encoders dinov2_base resnet50 clip_b32
python scripts/08_vision_embed_index.py --mode full --encoders dinov2_base --image-source snapshot
python scripts/10_retrieval_partial.py --mode full --set partial.strategy=keep_connected --set partial.keep_count=3
python scripts/03_build_graphs.py --mode preview --set geometry.wall_dilation_px=4
```

---

## 5. Smoke test with synthetic data (no dataset needed)

```bash
python scripts/make_synthetic_dataset.py
python scripts/run_all.py --mode preview --n 100 \
  --set paths.data_root=data_synthetic --set paths.artifacts_root=artifacts_synthetic \
  --set paths.outputs_root=outputs_synthetic --set "vision.encoders=[colorhist]"
python -m pytest tests -q
```

---

## 6. Code structure

```
configs/            default.yaml (all parameters), env/{local,cluster}.yaml, room_types.yaml
scripts/            00_… 12_ numbered phases, run_all.py, make_synthetic_dataset.py
src/prism/
  common/           config, workspace paths, device (cuda/mps/cpu), io, HTML reports, plots
  datasets/         schema + taxonomy, rplan.py, resplan.py (experimental), store.py, synthetic.py
  geometry/         polygon ops, adjacency.py, relations.py, front_door.py
  graph/            build.py (graph JSON), descriptor.py, features.py, data.py, models.py, train.py
  vision/           render.py, encoders.py
  retrieval/        index.py (FAISS / NumPy), fusion.py (RRF), query_encoders.py, visual.py
  partial/          masking.py, compatibility.py
  evaluation/       metrics.py (architectural metrics, Recall@K, GED)
cluster/            config.sh, setup_env.sh, predownload_models.py, slurm/*.sbatch
docs/relations.md   how adjacency, relations and the front door are computed
tests/              unit tests (pytest)
```

Every function has a docstring. The GNN layers are written in plain PyTorch
(no PyTorch Geometric extensions to compile), so the same code runs on
Windows/CUDA, macOS/MPS and the cluster.

---

## 7. RPLAN + ResPlan

All downstream code works on the unified **layout JSON / graph JSON** schema
(`src/prism/datasets/schema.py`):

- room types carry a fine type (dataset-specific) **and** a unified type
  (`configs/room_types.yaml`, ResPlan granularity);
- edges carry a connection type (`wall` / `door` / `arch` / `entrance`);
- the front door has the same representation in both datasets.

To add ResPlan:
1. put its files in `data/raw/resplan/`;
2. check `src/prism/datasets/resplan.py` against the real format (it was
   written from the paper);
3. run the phases with `--dataset resplan`.

To train or retrieve across both datasets, use `--set graph.type_vocab=unified`.

---

## 8. Working together

- Never commit `data/`, `artifacts/`, `outputs/` (already in `.gitignore`).
  Share results by copying the `report.html` files: they are self-contained.
- Change parameters in `configs/default.yaml` (shared defaults) or with
  `--set` (experiments). Machine-specific paths go in `configs/env/*.yaml`.
- To compare experiments, run them with different output roots, e.g.
  `--set paths.outputs_root=outputs_gat`, then
  `python scripts/12_compare_pipelines.py --evaluation outputs_gat/rplan/full/11_evaluate:gat`.
- Before pushing: `python -m pytest tests -q` and `ruff check src scripts`.
