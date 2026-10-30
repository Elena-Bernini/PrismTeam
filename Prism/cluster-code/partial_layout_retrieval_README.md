# Partial Layout Retrieval

This module evaluates retrieval from incomplete floorplan queries.

It is designed to run after the preprocessing stage has produced:

```text
data/processed/rplan/layouts_json/
data/processed/rplan/graphs_json/
```

The script creates partial queries by masking rooms from complete layouts, then retrieves complete layouts that are compatible with the visible rooms and known graph constraints.

---

## Goal

The goal is to test whether the retrieval system can answer questions such as:

> Given an incomplete apartment layout, can we retrieve complete floorplans that are compatible with the visible rooms and spatial constraints?

This is useful for semantic architectural search because a user may provide only a partial layout, a sketch, or a set of known rooms.

---

## Method

For each selected full layout:

1. load its preprocessed `layout_json` and `graph_json`;
2. remove one or more rooms according to a masking strategy;
3. keep only the visible rooms and the edges between visible rooms;
4. use the partial layout as a query;
5. retrieve complete layouts from the dataset;
6. re-rank candidates using explicit compatibility metrics.

The retrieval uses a two-stage strategy:

```text
partial query
    ↓
FAISS preselection over full-layout descriptors
    ↓
explicit compatibility re-ranking
    ↓
top-K complete floorplans
```

---

## Compatibility Metrics

The final ranking is based on interpretable architectural scores:

- room coverage;
- adjacency coverage;
- relation coverage;
- grid-location coverage;
- size compatibility;
- FAISS/vector similarity as a weak tie-breaker.

The original complete layout is used as a proxy ground-truth completion. The script reports whether the original layout is recovered in the top-K results.

---

## Output Structure

The script writes:

```text
outputs/partial_layout_retrieval/
├── results/
│   └── <query_id>_partial_top10.json
├── previews/
│   └── <query_id>_partial_top10.png
├── partial_metrics_per_result.csv
├── partial_metrics_by_query.csv
├── partial_summary.json
├── partial_summary.md
├── partial_summary_bar.png
└── partial_run_metadata.json
```

The most important files are:

```text
partial_summary.md
partial_metrics_by_query.csv
previews/
```

---

## Windows PowerShell Usage

From the project root:

```powershell
cd "C:\Users\pierg\Desktop\Uni\LM\1° Anno\2° Semestre\Computer Vision\Prism"
.\.venv\Scripts\activate
```

Install the required dependency if needed:

```powershell
python -m pip install faiss-cpu numpy pillow matplotlib tqdm
```

---

## Fast Test

Run a small test on 1000 layouts and 10 partial queries:

```powershell
python .\cluster-code\partial_layout_retrieval.py `
  --processed-root data\processed\rplan `
  --dataset-root data\raw\rplan\extracted `
  --outputs-root outputs\partial_layout_test `
  --build `
  --clean `
  --max-items 1000 `
  --num-queries 10 `
  --mask-strategy random `
  --remove-ratio 0.3 `
  --candidate-pool 100 `
  --top-k 10 `
  --preview-limit 10
```

Open:

```text
outputs/partial_layout_test/partial_summary.md
outputs/partial_layout_test/previews/
```

---

## Full Run

Run on the full preprocessed dataset:

```powershell
python .\cluster-code\partial_layout_retrieval.py `
  --processed-root data\processed\rplan `
  --dataset-root data\raw\rplan\extracted `
  --outputs-root outputs\partial_layout_full `
  --build `
  --clean `
  --num-queries 50 `
  --mask-strategy random `
  --remove-ratio 0.3 `
  --candidate-pool 300 `
  --top-k 10 `
  --preview-limit 20
```

---

## Single Query

After the index has been built, a single query can be run without rebuilding:

```powershell
python .\cluster-code\partial_layout_retrieval.py `
  --processed-root data\processed\rplan `
  --dataset-root data\raw\rplan\extracted `
  --outputs-root outputs\partial_layout_full `
  --query-id 75342 `
  --mask-strategy random `
  --remove-ratio 0.3 `
  --candidate-pool 300 `
  --top-k 10 `
  --preview-limit 1
```

---

## Masking Strategies

The script supports several masking strategies.

### Random

Remove a fraction of rooms:

```powershell
--mask-strategy random --remove-ratio 0.3
```

### Remove Bedroom

Remove one bedroom-like room if present:

```powershell
--mask-strategy remove_bedroom
```

### Remove Room Type

Remove all rooms of a specific RPLAN room type id:

```powershell
--mask-strategy remove_type --remove-type 3
```

Example room type ids:

```text
0  LivingRoom
1  MasterRoom
2  Kitchen
3  Bathroom
4  DiningRoom
5  ChildRoom
6  StudyRoom
7  SecondRoom
8  GuestRoom
9  Balcony
10 Entrance
11 Storage
12 Wall-in
```

### Remove Smallest / Largest Room

```powershell
--mask-strategy remove_smallest
```

```powershell
--mask-strategy remove_largest
```

---

## Linux / Cluster Usage

From the project root:

```bash
cd /work/cvcs2026/PrismTeam
source /homes/$USER/cvcs2026/venv/bin/activate
```

Fast test:

```bash
python cluster-code/partial_layout_retrieval.py \
  --processed-root data/processed/rplan \
  --dataset-root data/raw/rplan/extracted \
  --outputs-root outputs/partial_layout_test \
  --build \
  --clean \
  --max-items 1000 \
  --num-queries 10 \
  --mask-strategy random \
  --remove-ratio 0.3 \
  --candidate-pool 100 \
  --top-k 10 \
  --preview-limit 10
```

Full run:

```bash
python cluster-code/partial_layout_retrieval.py \
  --processed-root data/processed/rplan \
  --dataset-root data/raw/rplan/extracted \
  --outputs-root outputs/partial_layout_full \
  --build \
  --clean \
  --num-queries 50 \
  --mask-strategy random \
  --remove-ratio 0.3 \
  --candidate-pool 300 \
  --top-k 10 \
  --preview-limit 20
```

---

## Interpretation

Important metrics:

- `original_recall_at_k`: fraction of partial queries where the original complete layout was retrieved in the top-K;
- `mean_topk_compatibility`: average compatibility of retrieved layouts;
- `mean_top1_compatibility`: compatibility of the best retrieved layout;
- `mean_hard_constraint_rate_topk`: fraction of top-K results satisfying hard compatibility constraints.

The original complete layout is not the only valid answer, but it is a useful proxy for measuring whether the system can recover a plausible completion.
