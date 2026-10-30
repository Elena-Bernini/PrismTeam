# Graph-Based Retrieval

## Purpose

The graph retrieval branch retrieves floor plans using room topology and semantic structure rather than image appearance alone.

## Inputs

```text
layout.json
graph.json
graph.pt
```

Queries may be complete or partial graphs.

## Baseline 1: symbolic graph feature retrieval

Before training a GNN, extract a fixed-length feature vector from each graph.

Recommended features:

```text
number of rooms
room type histogram
area histogram
mean room area
area per room type
number of edges
degree histogram
average degree
spatial relation histogram
edge type histogram
boundary descriptor if available
```

Pipeline:

```text
graph.json
    -> handcrafted feature vector
    -> normalization
    -> FAISS index
    -> top-k retrieval
```

This baseline is interpretable and should be implemented first.

## Baseline 2: graph matching / constraint filtering

For partial graph queries, implement a symbolic filter before neural retrieval:

```text
filter by room count constraints
filter by required room types
filter by required adjacencies
rank by area and relation similarity
```

This produces a transparent graph retrieval baseline.

## Model: GNN graph encoder

Recommended models:

```text
GCN
GraphSAGE
GIN
GAT
```

Initial choice:

```text
GraphSAGE or GCN
```

Pipeline:

```text
graph.pt
    -> GNN encoder
    -> global mean/max pooling
    -> projection head
    -> normalized graph embedding
    -> FAISS index
    -> top-k retrieval
```

## Node features

Suggested node feature vector:

```text
one-hot room type
area ratio
centroid x/y
bbox x/y/w/h
aspect ratio
5x5 grid one-hot location
```

## Edge features

Suggested edge feature vector:

```text
one-hot edge type
one-hot spatial relation
centroid distance
shared boundary length
```

## Training with partial graphs

Training should use full graphs and masked versions of the same graphs.

Positive pair:

```text
partial graph of sample i
full graph of sample i
```

Negative pairs:

```text
full graphs of other samples
```

Loss options:

```text
InfoNCE
Triplet loss
Supervised contrastive loss
```

## Graph retrieval script

Expected command:

```bash
python scripts/retrieve_graph.py \
  --query_id 000123 \
  --model graph_features \
  --top_k 10 \
  --output outputs/graph_top10/000123
```

For GNN embeddings:

```bash
python scripts/retrieve_graph.py \
  --query_id 000123 \
  --model graphsage \
  --checkpoint checkpoints/graphsage.pt \
  --top_k 10 \
  --output outputs/gnn_top10/000123
```

## Evaluation

Evaluate:

- complete graph retrieval;
- partial graph retrieval;
- graph feature baseline vs GNN retrieval;
- graph-only vs vision-only vs hybrid retrieval.

Metrics:

```text
Recall@K
MRR
nDCG
room type histogram similarity
adjacency precision/recall/F1
graph edit distance
required adjacency satisfaction
```
