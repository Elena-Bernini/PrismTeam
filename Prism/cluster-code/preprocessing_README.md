# RPLAN / Graph2Plan Preprocessing Pipeline

This document describes the unified preprocessing script used to convert the preprocessed RPLAN / Graph2Plan dataset into layout JSON files, graph JSON files, dataset statistics, and visual quality-control previews.

The script is designed to be run both locally and on a cluster. It replaces multiple small preprocessing scripts with a single reproducible entry point.

---

## 1. Purpose

The preprocessing pipeline converts the Graph2Plan-preprocessed RPLAN data into a project-friendly representation for graph-based floorplan retrieval.

It produces:

```text
data/processed/rplan/
├── layouts_json/
├── graphs_json/
└── metadata/

outputs/dataset_audit/
├── rplan_file_audit.json
├── train_converted_sample_report.json
├── rplan_processed_statistics.json
├── preprocessing_summary.json
├── preprocessing_report.md
├── layout_previews/
└── graph_previews/
```

The main outputs used by later retrieval stages are:

```text
data/processed/rplan/layouts_json/
data/processed/rplan/graphs_json/
data/processed/rplan/metadata/layouts_metadata.json
```

The preview images and audit reports are for debugging, validation, and documentation. They are not model inputs.

---

## 2. Expected Dataset Structure

The script expects the unzipped Graph2Plan/RPLAN data under:

```text
data/raw/rplan/extracted/
```

The required file is:

```text
data/raw/rplan/extracted/Interface/static/Data/data_train_converted.pkl
```

The optional but recommended snapshot directory is:

```text
data/raw/rplan/extracted/Interface/static/Data/snapshot_train/
```

The snapshot images are used only for side-by-side visual checks.

---

## 3. Installation

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install the required packages:

```bash
pip install numpy scipy pillow matplotlib tqdm
```

No PyTorch, FAISS, or GPU libraries are required for this preprocessing step.

---

## 4. Quick Debug Run

Before converting the full dataset, run a small test on 200 samples:

```bash
python scripts/preprocessing.py --clean --limit 200 --layout-preview-n 30 --graph-preview-n 10
```

This performs the full pipeline on a small subset:

1. Inspects the raw dataset.
2. Converts the first 200 layouts to JSON.
3. Converts the layouts to graph JSON.
4. Computes dataset statistics on the converted subset.
5. Generates layout comparison previews.
6. Generates graph comparison previews.
7. Writes a Markdown preprocessing report.

After the run, inspect:

```text
outputs/dataset_audit/layout_previews/
outputs/dataset_audit/graph_previews/
outputs/dataset_audit/preprocessing_report.md
```

---

## 5. Full Run

When the debug run is correct, run the full preprocessing:

```bash
python scripts/preprocessing.py --clean --layout-preview-n 50 --graph-preview-n 20
```

This converts all available samples but generates only a small number of preview images.

The preview count is intentionally limited because preview images are only for quality control. Generating one image per layout would be slow and would waste disk space.

---

## 6. Running on a Cluster

Place the script in the project scripts directory, for example:

```text
/work/cvcs2026/PrismTeam/scripts/preprocessing.py
```

Then run from the project root:

```bash
cd /work/cvcs2026/PrismTeam
python scripts/preprocessing.py --clean --limit 200 --layout-preview-n 30 --graph-preview-n 10
```

If the dataset is located in a different directory, pass it explicitly:

```bash
python scripts/preprocessing.py \
  --clean \
  --dataset-root /path/to/rplan/extracted \
  --layout-preview-n 50 \
  --graph-preview-n 20
```

---

## 7. Official Room Type Mapping

The script uses the official Graph2Plan/RPLAN room type IDs for room boxes.

| ID | Room type |
|---:|---|
| 0 | LivingRoom |
| 1 | MasterRoom |
| 2 | Kitchen |
| 3 | Bathroom |
| 4 | DiningRoom |
| 5 | ChildRoom |
| 6 | StudyRoom |
| 7 | SecondRoom |
| 8 | GuestRoom |
| 9 | Balcony |
| 10 | Entrance |
| 11 | Storage |
| 12 | Wall-in |

Each room in `layout.json` is saved with:

```json
{
  "id": 0,
  "room_type_id": 7,
  "room_type_name": "SecondRoom"
}
```

The field `id` is the local room/node index inside a single floorplan. The field `room_type_id` is the original Graph2Plan/RPLAN semantic room type ID.

---

## 8. Spatial Relation Mapping

The script uses the official Graph2Plan relation mapping.

An edge has the form:

```text
(source, target, relation_id)
```

The relation describes the source room relative to the target room.

| ID | Relation |
|---:|---|
| 0 | left-above |
| 1 | left-below |
| 2 | left-of |
| 3 | above |
| 4 | inside |
| 5 | surrounding |
| 6 | below |
| 7 | right-of |
| 8 | right-above |
| 9 | right-below |

For example:

```json
{
  "source": 2,
  "target": 4,
  "relation_id": 0,
  "relation": "left-above"
}
```

means that room `2` is left-above room `4`.

---

## 9. Coordinate System

All converted layouts use the Graph2Plan 256 x 256 image coordinate system:

```text
origin = top-left
x-axis = right
y-axis = down
```

This is an image coordinate system, not a standard Cartesian plot.

Therefore:

```text
above      = smaller y
below      = larger y
left-of    = smaller x
right-of   = larger x
```

In Matplotlib visualizations, the y-axis is inverted with:

```python
ax.set_ylim(256, 0)
```

This preserves the same orientation as the original floorplan images.

---

## 10. Layout JSON Schema

Each file in:

```text
data/processed/rplan/layouts_json/
```

contains one floorplan.

Simplified example:

```json
{
  "id": "75342",
  "coordinate_system": {
    "width": 256,
    "height": 256,
    "origin": "top_left",
    "x_axis": "right",
    "y_axis": "down"
  },
  "rooms": [
    {
      "id": 0,
      "room_type_id": 7,
      "room_type_name": "SecondRoom",
      "bbox": [37, 41, 87, 94],
      "width": 50,
      "height": 53,
      "area_pixels": 2650,
      "area_ratio": 0.0404,
      "centroid": [62.0, 67.5],
      "grid_location_5x5": [1, 1],
      "polygon": [[37, 41], [87, 41], [87, 94], [37, 94]]
    }
  ],
  "edges": [
    {
      "id": 0,
      "source": 0,
      "target": 1,
      "relation_id": 7,
      "relation": "right-of"
    }
  ]
}
```

The `polygon` field is used for accurate visualization. The `bbox`, `centroid`, `area_ratio`, and `grid_location_5x5` fields are useful for graph features and retrieval baselines.

---

## 11. Graph JSON Schema

Each file in:

```text
data/processed/rplan/graphs_json/
```

contains the graph representation of one floorplan.

Simplified example:

```json
{
  "id": "75342",
  "nodes": [
    {
      "id": 0,
      "room_type_id": 7,
      "room_type_name": "SecondRoom",
      "area_ratio": 0.0404,
      "bbox": [37, 41, 87, 94],
      "centroid": [62.0, 67.5],
      "grid_location_5x5": [1, 1]
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "relation_id": 7,
      "relation": "right-of"
    }
  ]
}
```

This graph representation is the main input for the graph-based retrieval baseline.

---

## 12. Visual Quality Control

The script generates two types of preview images.

### Layout previews

Saved in:

```text
outputs/dataset_audit/layout_previews/
```

Each image compares:

```text
left: original snapshot image
right: polygon-based layout JSON rendering
```

These previews are used to verify room geometry and semantic room labels.

### Graph previews

Saved in:

```text
outputs/dataset_audit/graph_previews/
```

Each image compares:

```text
left: original snapshot image
right: graph rendering with room nodes and spatial relation edges
```

These previews are used to verify that graph nodes are located in the correct rooms and that relation edges are plausible.

---

## 13. Useful Commands

Run a small debug subset:

```bash
python scripts/preprocessing.py --clean --limit 200 --layout-preview-n 30 --graph-preview-n 10
```

Run the full preprocessing:

```bash
python scripts/preprocessing.py --clean --layout-preview-n 50 --graph-preview-n 20
```

Run without preview images:

```bash
python scripts/preprocessing.py --clean --skip-previews
```

Generate previews for specific IDs:

```bash
python scripts/preprocessing.py \
  --clean \
  --limit 200 \
  --layout-preview-ids 75342,42007 \
  --graph-preview-ids 75342,42007
```

Use a custom dataset path:

```bash
python scripts/preprocessing.py \
  --dataset-root /path/to/rplan/extracted \
  --clean \
  --layout-preview-n 50 \
  --graph-preview-n 20
```

---

## 14. Recommended Git Ignore Rules

The raw dataset and generated outputs should not be committed to Git.

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.pyc

data/raw/
data/processed/
outputs/

*.pkl
*.npy
*.mat
*.index
```

---

## 15. Next Step After Preprocessing

After this preprocessing step, the next project stage is the graph retrieval baseline:

```text
graphs_json -> feature vectors -> nearest neighbors -> top-k similar floorplans
```

The graph retrieval baseline can use:

```text
room_type_id histogram
room_type_name histogram
room count
edge count
relation histogram
area statistics
centroid/grid statistics
```

This provides a simple, interpretable retrieval baseline before introducing GNN encoders or vision encoders.
