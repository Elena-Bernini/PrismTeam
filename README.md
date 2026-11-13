# PrismPlan
### AI-Assisted Floor Plan Retrieval and Search

![Status](https://img.shields.io/badge/status-in%20development-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)
![Datasets](https://img.shields.io/badge/datasets-RPLAN%20%2B%20ResPlan-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An AI-powered system for floor plan retrieval and constraint-based search, developed by **Prism Team** in collaboration with **Maticad** (CVCS project, Phase 1).

---

## 👥 Team

**Prism Team**

- Elena Bernini
- Piergiorgio Signorino

---

## 📖 Project Overview

Designing interior spaces requires balancing structural constraints, functional requirements and aesthetic considerations. While modern CAD software provides powerful design tools, the early conceptual phase remains largely manual and time-consuming.

**PrismPlan** explores how Artificial Intelligence can assist designers by retrieving existing layouts that match a partial plan or a list of requirements. It compares two families of representations:

- **graph** representations (rooms as nodes, adjacencies as edges, hand-crafted descriptors and Graph Neural Networks);
- **vision** representations (semantic renders encoded by DINOv2, CLIP, ResNet).

One codebase covers the whole workflow, from the raw datasets to evaluated rankings:

```text
RPLAN + ResPlan ─► 01 inspect ─► 02 layout JSON ─► 03 graphs ─► 04 renders ─► 05 stats / splits
                                                      │               │
                                       06 GNN training│               │
                                                      ▼               ▼
                              07 graph index (descriptor, GNN)    08 vision index (DINOv2, CLIP, ResNet, colorhist)
                                                      └───────┬───────┘
                                     09 retrieval (complete queries + hybrid RRF)
                                     10 partial retrieval (masked queries + compatibility re-ranking)
                                     11 evaluation (architectural metrics, Recall@K, front door)
                                     12 comparison across pipelines / runs
                                     13 constraint search (k most plausible plans for a request)
```

---

## 🎯 Objectives

### Floor Plan Retrieval

Given a complete or **partial** floor plan, retrieve the most similar layouts from the gallery.

- Graph-based floor plan representation with recomputed spatial relations and the front door
- Graph embeddings: interpretable descriptor + GNN (GCN / GraphSAGE / GAT, auto-encoder or contrastive)
- Vision embeddings of semantic renders (DINOv2, CLIP, ResNet, colour histogram)
- Similarity search with FAISS (exact NumPy fallback) and hybrid fusion (Reciprocal Rank Fusion)
- Partial queries (rooms masked) re-ranked by architectural compatibility

### Constraint-Based Search

Given **requirements** instead of a drawing, return the **k most plausible floor plans**, each with an explanation of how well it meets every requirement.

- Floor area in m² (target and/or range, net or gross)
- Number of rooms
- Room types wanted or excluded, with counts ("2 bedrooms, at least 1 bathroom, no study")
- Room behind the front door
- Adjacency wishes ("kitchen next to living")

Each requirement is soft (scored) or hard (filter). Results are diversified, and the best plans that fail a hard requirement are shown separately as near misses.

### Floor Plan Generation (next phase)

Generating new layouts from incomplete specifications is the goal of the next project phase (see [Future Developments](#-future-developments)).

---

## 🏗 Datasets

### RPLAN

Public benchmark of ~80k residential plans, used through the Graph2Plan release (`data_train_converted.pkl`). It provides 13 room types, room polygons and boxes, the exterior outline and the front door. It has no metric scale.

### ResPlan

~17k residential plans ([github.com/m-agour/ResPlan](https://github.com/m-agour/ResPlan)) with room polygons, walls, doors, windows, the front door, an official train/val/test split and the authors' own room graph (`plan["graph"]`).

### Combined gallery

By default both datasets form **one gallery** (`--dataset combined`). Each dataset can also be used alone (`--dataset rplan | resplan`). To make them comparable:

- layout ids are prefixed with the dataset (`rplan_75342`, `resplan_5012`);
- room types have a fine (dataset-specific) and a **unified** level (bedroom, bathroom, kitchen …);
- ResPlan's wall-separated rooms are turned into an **RPLAN-style partition**: rooms meet on wall centre lines and balconies are attached, so both datasets render in the same style;
- ResPlan edges come from ResPlan's own graph; RPLAN edges from our geometric adjacency;
- RPLAN areas in m² are **estimates**, calibrated on ResPlan room sizes.

### Maticad

Professional interior design projects from Maticad, to be integrated in a later phase. They are not part of the current pipeline.

Details: [`docs/datasets.md`](docs/datasets.md) and [`docs/relations.md`](docs/relations.md).

---

## 🧠 Methodology

### Graph Representation

Each floor plan is represented as an attributed graph.

**Nodes**
- Room type (fine and unified), area, centroid, bounding box, 5×5 grid cell
- Entrance-room flag, hop distance and distance to the front door
- A virtual **FrontDoor** node linked to the entrance room

**Edges**
- Adjacency (shared wall), stored in both directions
- Spatial relation (left_of, above, inside …), recomputed deterministically from the rooms
- Connection type: wall / door / arch / window / entrance

### Retrieval Pipeline

1. Dataset conversion into one layout JSON per plan (phase 02)
2. Graph construction (phase 03) and semantic renders (phase 04)
3. Graph embeddings: descriptor and GNN (phases 06–07)
4. Vision embeddings of the renders (phase 08)
5. Vector indexing (FAISS or NumPy) and similarity search, single pipelines and hybrid RRF (phase 09)
6. Partial queries with compatibility re-ranking (phase 10)
7. Evaluation and comparison (phases 11–12)

### Constraint Search Pipeline

1. A plan catalog with room counts, m² per room (dataset or calibrated) and room-graph facts
2. A satisfaction score per requirement, combined into a constraint score
3. A plausibility prior from the gallery's own statistics (room sizes and shapes, reachability from the entrance, living room as a hub)
4. Hard filtering, ranking, MMR diversity, near misses and an explanation for every result

---

## 🛠 Tech Stack

- Python
- PyTorch (GNN layers in plain PyTorch, no compiled extensions)
- Hugging Face Transformers and timm (DINOv2, CLIP, ResNet)
- FAISS (with an exact NumPy fallback)
- NumPy, SciPy, Shapely
- NetworkX
- Pillow and Matplotlib (single-file HTML reports)
- PyYAML, pytest, ruff

Supported machines: Windows + NVIDIA (CUDA), macOS Apple Silicon (MPS), CPU only, and the AImageLab-HPC SLURM cluster.

---

## ⚙️ Installation

Python 3.9–3.12. Create a virtual environment in the project folder.

**Windows + NVIDIA GPU**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements\base.txt -r requirements\vision.txt
python -m pip install -e .
python scripts\00_check_env.py          # should say: device cuda (NVIDIA ...)
```

If `cu124` does not match the driver, pick the right command on the PyTorch "Get started" page (`nvidia-smi` shows the supported CUDA version).

**macOS + Apple Silicon**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision          # includes Apple MPS support
python -m pip install -r requirements/base.txt -r requirements/vision.txt
python -m pip install -e .
python scripts/00_check_env.py                   # should say: device mps
```

The code sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` on macOS automatically: unsupported ops run on the CPU, and torch and faiss can share a process. If `faiss-cpu` does not install, remove it from `requirements/base.txt`: the code falls back to exact NumPy search with the same results.

**Cluster (AImageLab-HPC)**: see [`cluster/README_cluster.md`](cluster/README_cluster.md). In short, run `bash cluster/setup_env.sh` once, then `sbatch cluster/slurm/preview_all.sbatch`.

**Without a GPU / without PyTorch**: phases 01–05, the descriptor index, the `colorhist` vision index and phases 09–13 need no PyTorch. Phase 06 and the deep vision encoders are skipped with a clear message when it is missing. `--device cpu` forces the CPU everywhere.

---

## 📂 Repository Structure

```text
.
├── configs/                 default.yaml (all parameters), room_types.yaml, env/{local,cluster}.yaml,
│                            specs/ (example requests for phase 13)
├── scripts/                 00_ … 13_ numbered phases, run_all.py, make_synthetic_dataset.py,
│                            resplan_geometry_check.py
├── src/prism/
│   ├── common/              config, workspace paths, device (cuda/mps/cpu), io, HTML reports, plots
│   ├── datasets/            schema + taxonomy, rplan.py, resplan.py, resplan_geometry.py, store.py, synthetic.py
│   ├── geometry/            polygon ops, adjacency, spatial relations, front door
│   ├── graph/               graph JSON, descriptor, GNN features / models / training
│   ├── vision/              semantic renderer, vision encoders
│   ├── retrieval/           FAISS / NumPy index, RRF fusion, query encoders
│   ├── partial/             masking strategies, compatibility re-ranking
│   ├── evaluation/          architectural metrics, Recall@K, GED
│   └── constraints/         requests, plan catalog (m², calibration), scoring, search
├── cluster/                 SLURM jobs and environment setup
├── docs/                    relations.md, datasets.md, constraint_search.md
├── tests/                   unit tests (pytest)
├── data/                    raw/ (datasets) and processed/ (phases 02–05)   ← not in git
├── artifacts/               models, embeddings, indexes, caches             ← not in git
└── outputs/                 one report.html per phase                       ← not in git
```

### Where to put the data

```text
data/raw/rplan/Interface/static/Data/
    data_train_converted.pkl          ← REQUIRED for rplan / combined (~80k plans)
    snapshot_train/*.png              ← optional (visual checks, --image-source snapshot)
    Img/*.png                         ← optional (--image-source img)
data/raw/resplan/
    ResPlan.pkl                       ← REQUIRED for resplan / combined (~17k plans)
    split.json                        ← optional (official train/val/test split)
```

Everything generated is scoped by dataset (`rplan | resplan | combined`) and mode (`preview | full`):

- `data/processed/<dataset>/<mode>/`
- `artifacts/<dataset>/<mode>/`
- `outputs/<dataset>/<mode>/<NN_phase>/report.html`

Paths can be changed in `configs/env/local.yaml` or with `--set paths.data_root=...`.

---

## 🚀 Usage

### Preview first, then full

Every phase has two modes:

- `--mode preview` (default) runs on a fixed random subset (`--n`, default 200 per dataset) and writes a visual `report.html`;
- `--mode full` runs the same command on the whole dataset.

The two modes write to different folders and never overwrite each other.

```bash
python scripts/run_all.py --mode preview                # phases 01 → 13
python scripts/run_all.py --mode full --from 2 --to 5   # a range of phases
python scripts/03_build_graphs.py --mode preview        # a single phase
```

### The phases

| # | Script | What it does | What to look at in the report |
|---|---|---|---|
| 00 | `00_check_env.py` | libraries, device (cuda/mps/cpu), data files | which phases can run on this machine |
| 01 | `01_inspect_dataset.py` | raw pickle structure, units, front door, ResPlan's own graph | unknown keys, edge types, door alignment |
| 02 | `02_convert_to_layout_json.py` | one layout JSON per plan (taxonomy, front door, ResPlan partition) | layouts vs originals, entrance room, graph matching |
| 03 | `03_build_graphs.py` | adjacency, spatial relations, connection types, FrontDoor node | graphs over plans, where the edges come from |
| 04 | `04_render_images.py` | semantic renders (rooms by type, yellow door) | renders, partial-query style |
| 05 | `05_statistics_and_splits.py` | statistics, train/val/test split, `--pack` | distributions, RPLAN vs ResPlan alignment |
| 06 | `06_train_gnn.py` | GCN / GraphSAGE / GAT; GAE, contrastive or both | loss curves, PCA of embeddings |
| 07 | `07_graph_embed_index.py` | graph indexes: `descriptor`, `gnn` | nearest neighbours of sample queries |
| 08 | `08_vision_embed_index.py` | vision indexes: `colorhist`, `resnet18/50`, `dinov2_*`, `clip_b32`, `timm:*` | nearest neighbours of sample queries |
| 09 | `09_retrieval_full.py` | complete-plan queries on every index + hybrid RRF | pipelines side by side |
| 10 | `10_retrieval_partial.py` | masked queries → candidate pool → compatibility re-ranking | is the original plan recovered? |
| 11 | `11_evaluate.py` | architectural metrics, Recall@K, Hit@K, MRR, front door, optional GED | summary table, metric vs rank |
| 12 | `12_compare_pipelines.py` | merges phases 10–11 and other runs | final comparison |
| 13 | `13_constraint_search.py` | k most plausible plans for a request (m², rooms, room types, entrance, adjacency) | top-k grid with explanations, near misses, area calibration |

Common options for every script:

```text
--mode preview|full   --n 500   --dataset combined|rplan|resplan   --env local|cluster
--device auto|cpu|cuda|mps   --set key.subkey=value   --config extra.yaml
```

### Examples

```bash
python scripts/06_train_gnn.py --mode full --set graph.model=gat --set graph.objective=both
python scripts/08_vision_embed_index.py --mode full --encoders dinov2_base resnet50 clip_b32
python scripts/10_retrieval_partial.py --mode full --set partial.strategy=keep_connected --set partial.keep_count=3
python scripts/resplan_geometry_check.py --ids 13997 1022      # ResPlan: original / before / after renders
```

### Constraint search

```bash
python scripts/13_constraint_search.py --spec configs/specs/example_2bed.yaml
python scripts/13_constraint_search.py --area 65-90 --area-target 75 --rooms 4-6 \
    --type bedroom=2 --type bathroom=1+ --type kitchen=1 --type study=0 --hard type:bedroom --k 10
```

Requests can be written in YAML (see `configs/specs/`) or given as flags. Every result shows the area with its source ("≈" = calibrated RPLAN estimate). See [`docs/constraint_search.md`](docs/constraint_search.md).

### Smoke test (no dataset needed)

```bash
python scripts/make_synthetic_dataset.py                 # fake RPLAN + fake ResPlan in data_synthetic/
python scripts/run_all.py --mode preview --n 100 \
  --set paths.data_root=data_synthetic --set paths.artifacts_root=artifacts_synthetic \
  --set paths.outputs_root=outputs_synthetic --set "vision.encoders=[colorhist]"
python -m pytest tests -q
```

---

## 📊 Evaluation

### Retrieval Metrics

- Recall@K, Hit@K, Precision@K, MRR. A plan is relevant when it has the same unified room composition and similar adjacencies.
- Architectural similarity: room types and counts, areas by type, typed adjacency, relation triplets, grid occupancy, constraint satisfaction
- Front-door agreement: entrance-room type and door facing
- Combined gallery: `same_dataset_rate@K` per query dataset (does a pipeline match layouts, or dataset style?)
- Optional graph edit distance on a sample of pairs

### Constraint Search

- Satisfaction of every requirement (met / partly / violated / unknown) and a weighted constraint score
- Plausibility components: room size, room shape, reachability from the entrance, living room as a hub
- Hard-requirement funnel and near misses

---

## 🤝 Working Together

- Never commit `data/`, `artifacts/`, `outputs/` (already in `.gitignore`). Share results by copying the `report.html` files: they are self-contained.
- Shared defaults go in `configs/default.yaml`, experiments use `--set`, and machine-specific paths go in `configs/env/*.yaml`.
- Compare experiments by running them with different output roots (e.g. `--set paths.outputs_root=outputs_gat`), then `python scripts/12_compare_pipelines.py --evaluation outputs_gat/combined/full/11_evaluate:gat`.
- Before pushing: `python -m pytest tests -q` and `ruff check src scripts`.

### Troubleshooting

- **`ModuleNotFoundError: scipy` / `shapely` when loading a pickle**: both are needed just to open the RPLAN and ResPlan files. Run `pip install -r requirements/base.txt`.
- **`taxonomy.level=fine is not allowed for the combined dataset`**: RPLAN and ResPlan fine types are different vocabularies; use `unified` (or `auto`).
- **`The GNN was trained with a different room-type vocabulary`**: a checkpoint is tied to its dataset and taxonomy level. Re-run phase 06.
- **Files named `10154 2.json` in `data/processed/`**: these are copies made by Finder or iCloud. Delete them, or re-run phases 02–04.

---

## 📚 References

- Hu, R. et al. *Graph2Plan: Learning Floorplan Generation from Layout Graphs*. ACM SIGGRAPH 2020.
- Wu, W. et al. *Data-driven Interior Plan Generation for Residential Buildings*. ACM SIGGRAPH Asia 2019 (RPLAN dataset).
- ResPlan dataset: [github.com/m-agour/ResPlan](https://github.com/m-agour/ResPlan).
- Hamilton, W. L. et al. *Inductive Representation Learning on Large Graphs* (GraphSAGE). NeurIPS 2017.
- Oquab, M. et al. *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024.
- Cormack, G. V. et al. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR 2009.
- Carbonell, J. and Goldstein, J. *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*. SIGIR 1998.
- Nauata, N. et al. *House-GAN++: Generative Adversarial Layout Refinement Network towards Intelligent Computational Agent for Professional Architects*. CVPR 2021.
- Chai, S. et al. *LayoutDM: Transformer-based Diffusion Model for Layout Generation*. CVPR 2023.

---

## 🚀 Future Developments

- Floor plan **generation** from partial specifications (diffusion- or transformer-based, constraint-aware decoding), with the constraint search as a baseline
- Integration of the **Maticad** professional dataset
- Human-in-the-loop refinement: generate, correct, update constraints, regenerate
- "More plans like this": use constraint-search results as seeds for the retrieval pipelines
- Learned plausibility (e.g. density of GNN embeddings)
- Interactive CAD integration and natural-language requests

---

## 🤝 Acknowledgments

This project is developed in collaboration with **Maticad**, which provides the industrial application context of the research.

---

## 📄 License

Released under the [MIT License](LICENSE).
