# Implementation Roadmap

## Phase 0: Repository setup

Create the project structure:

```text
PrismTeam/
    data/
        raw/
        processed/
        embeddings/
        indexes/
        splits/
    src/
        data/
        graph/
        vision/
        masking/
        retrieval/
        evaluation/
        visualization/
    scripts/
    notebooks/
    outputs/
    checkpoints/
    docs/
    report/
```

Create initial files:

```text
README.md
requirements.txt
.gitignore
configs/default.yaml
```

## Phase 1: Dataset audit

Tasks:

```text
load RPLAN
inspect file structure
visualize sample images
inspect annotation channels
map room labels
count room categories
compute room count distribution
compute room area distribution
```

Deliverables:

```text
notebooks/01_dataset_audit.ipynb
outputs/audit/dataset_stats.json
outputs/audit/sample_grid.png
outputs/audit/room_type_mapping.json
```

## Phase 2: RPLAN preprocessing

Tasks:

```text
extract room instances
extract room labels
extract building boundary
compute bbox, centroid, area, area ratio
create layout.json for each sample
create preview images
```

Deliverables:

```text
data/processed/rplan/<sample_id>/layout.json
data/processed/rplan/<sample_id>/preview.png
scripts/build_layout_json.py
```

## Phase 3: Graph extraction

Tasks:

```text
construct room nodes
compute node features
compute adjacency edges
compute spatial relation labels
export graph.json
export graph.pt
render graph overlay
```

Deliverables:

```text
data/processed/rplan/<sample_id>/graph.json
data/processed/rplan/<sample_id>/graph.pt
outputs/graph_debug/<sample_id>_graph_overlay.png
scripts/build_graphs.py
```

## Phase 4: Vision retrieval baseline

Tasks:

```text
preprocess images
extract embeddings with pretrained encoder
build FAISS index
implement top-k search
create top-10 visualization grid
evaluate complete image queries
```

Deliverables:

```text
data/embeddings/vision_embeddings.npy
data/indexes/vision.faiss
scripts/build_vision_index.py
scripts/retrieve_vision.py
outputs/vision_top10/
outputs/metrics/vision_metrics.json
```

## Phase 5: Symbolic graph retrieval baseline

Tasks:

```text
extract handcrafted graph features
normalize graph features
build FAISS index
implement graph top-k search
evaluate complete graph queries
```

Deliverables:

```text
data/embeddings/graph_features.npy
data/indexes/graph_features.faiss
scripts/build_graph_feature_index.py
scripts/retrieve_graph.py
outputs/graph_top10/
outputs/metrics/graph_feature_metrics.json
```

## Phase 6: Partial layout masking

Tasks:

```text
generate raster masks
generate graph masks
generate semantic constraint queries
save mask metadata
create partial retrieval benchmark split
```

Deliverables:

```text
data/processed/rplan/<sample_id>/masks/
data/splits/partial_queries_train.json
data/splits/partial_queries_val.json
data/splits/partial_queries_test.json
scripts/generate_masks.py
```

## Phase 7: Partial retrieval evaluation

Tasks:

```text
evaluate masked raster retrieval
evaluate masked graph retrieval
evaluate semantic constraint retrieval
report metrics by mask level
create qualitative grids
```

Deliverables:

```text
outputs/metrics/partial_vision_metrics.json
outputs/metrics/partial_graph_metrics.json
outputs/partial_top10_grids/
```

## Phase 8: GNN graph retrieval

Tasks:

```text
load graph.pt dataset
implement GraphSAGE or GCN encoder
train with full and partial graphs
extract graph embeddings
build FAISS index
evaluate complete and partial graph retrieval
```

Deliverables:

```text
checkpoints/gnn_graph_encoder.pt
data/embeddings/gnn_graph_embeddings.npy
data/indexes/gnn_graph.faiss
scripts/train_gnn_retriever.py
outputs/metrics/gnn_graph_metrics.json
```

## Phase 9: Hybrid retrieval

Tasks:

```text
load vision candidates
load graph candidates
load GNN candidates
merge candidate sets
normalize scores
compute semantic and adjacency scores
re-rank candidates
visualize final top-10
```

Deliverables:

```text
scripts/retrieve_hybrid.py
outputs/hybrid_top10/
outputs/metrics/hybrid_metrics.json
```

## Phase 10: Final evaluation and report

Tasks:

```text
compare vision, graph, GNN, and hybrid retrieval
compare complete and partial queries
produce final tables
produce qualitative figures
write final report
prepare GitHub README
```

Deliverables:

```text
outputs/final_tables/
outputs/final_figures/
report/final_report.pdf
README.md
docs/
```

## First target milestone

The first milestone is a working vision retrieval demo:

```text
query RPLAN image -> FAISS -> top-10 similar floor plans
```

The second milestone is a working graph retrieval demo:

```text
query RPLAN graph -> FAISS -> top-10 structurally similar floor plans
```

The third milestone is partial layout retrieval:

```text
partial query -> retrieve complete compatible layouts
```
