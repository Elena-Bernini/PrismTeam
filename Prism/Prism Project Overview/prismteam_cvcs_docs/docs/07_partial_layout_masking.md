# Partial Layout Masking

## Purpose

Partial layout retrieval is central to the project. It simulates the architectural workflow where designers start from incomplete sketches or partially specified spatial constraints.

The system must retrieve complete and compatible floor plans from incomplete queries.

## General protocol

For each complete floor plan:

```text
full floor plan -> degraded query -> retrieval -> check whether compatible full plans are retrieved
```

The original full plan is treated as a positive target. Other structurally compatible plans can also be considered semantically relevant.

## Raster masking

Raster masks operate on floor plan images.

Mask types:

```text
room removal
rectangular occlusion
semantic class removal
boundary-preserving room deletion
structural component removal
```

Output:

```text
data/processed/rplan/<sample_id>/masks/raster_mask_<type>_<level>.png
```

Mask levels:

```text
10 percent
30 percent
50 percent
70 percent
```

## Graph masking

Graph masks operate on the layout graph.

Mask types:

```text
node removal
edge removal
room type masking
room size masking
room location masking
subgraph query
```

Example:

```text
Full graph:
LivingRoom -- Kitchen
LivingRoom -- Bedroom
Bedroom -- Bathroom
LivingRoom -- Balcony

Partial graph:
LivingRoom -- Kitchen
Bedroom -- unknown
```

## Semantic masking

Semantic masking creates queries based on design constraints.

Examples:

```text
keep only living area and kitchen
keep only bedrooms and bathrooms
keep only room counts and types
keep only required adjacencies
remove all room positions
```

This is useful for testing whether the graph retrieval system can retrieve floor plans based on high-level architectural constraints.

## Mask metadata

Every generated mask must be saved with metadata:

```json
{
  "sample_id": "000001",
  "mask_type": "node_removal",
  "mask_level": 0.3,
  "removed_rooms": [2, 4],
  "removed_edges": [[1, 2]],
  "query_file": "graph_mask_node_removal_30.json",
  "target_file": "graph.json"
}
```

## Positive and negative pairs

For contrastive training:

Positive pair:

```text
masked query from sample i
complete representation of sample i
```

Negative pairs:

```text
complete representations from other samples
```

## Retrieval evaluation on masks

For each mask level, evaluate:

```text
Recall@1
Recall@5
Recall@10
MRR
nDCG
semantic similarity
adjacency similarity
functional constraint satisfaction
```

## Expected result tables

```text
Method          Mask 10%   Mask 30%   Mask 50%   Mask 70%
Vision only     ...        ...        ...        ...
Graph features  ...        ...        ...        ...
GNN graph       ...        ...        ...        ...
Hybrid          ...        ...        ...        ...
```

## Debug output

For each test query, generate:

```text
query_complete.png
query_masked.png
top10_grid.png
results.json
scores.csv
```
