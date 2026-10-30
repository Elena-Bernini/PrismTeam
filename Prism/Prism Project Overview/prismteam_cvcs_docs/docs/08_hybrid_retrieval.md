# Hybrid Retrieval and Re-ranking

## Purpose

Hybrid retrieval combines the strengths of vision-based and graph-based retrieval.

Vision retrieval captures visual and geometric similarity. Graph retrieval captures semantic and topological compatibility. Re-ranking combines both with explicit architectural constraints.

## Candidate generation

For each query, retrieve candidates from each branch:

```text
top-k image candidates
top-k graph feature candidates
top-k GNN candidates
```

Then create the union:

```text
candidate_set = union(image_candidates, graph_feature_candidates, gnn_candidates)
```

## Score normalization

Each branch may produce scores on a different scale. Normalize all scores before fusion.

Recommended methods:

```text
min-max normalization per query
z-score normalization per query
rank-based score normalization
```

## Re-ranking score

Use a weighted final score:

```text
score_final =
    alpha * visual_similarity
  + beta  * graph_similarity
  + gamma * room_composition_score
  + delta * adjacency_score
  + eta   * area_score
  + theta * boundary_score
```

Initial weights can be set manually and later tuned on validation data.

## Constraint scores

### Room composition score

Measures how similar the retrieved layout is to the query in room types and counts.

Components:

```text
room count difference
room type histogram cosine similarity
bedroom count match
bathroom count match
kitchen/living presence
```

### Adjacency score

Measures how many query adjacencies are preserved.

```text
adjacency_precision
adjacency_recall
adjacency_F1
required_adjacency_satisfaction
```

### Area score

Measures whether room areas and global area distribution are compatible.

```text
absolute area error
area ratio error
area histogram similarity
```

### Boundary score

Measures compatibility of building footprint shape.

Possible descriptors:

```text
aspect ratio
boundary area
boundary compactness
turning function distance
```

The turning function idea is useful because Graph2Plan uses boundary similarity to rank retrieved graph layouts.

## Output format

For each query, save:

```json
{
  "query_id": "000123",
  "top_k": 10,
  "results": [
    {
      "rank": 1,
      "sample_id": "004532",
      "final_score": 0.91,
      "visual_score": 0.88,
      "graph_score": 0.84,
      "room_composition_score": 0.95,
      "adjacency_score": 0.90,
      "area_score": 0.86
    }
  ]
}
```

## Visualization

Create a grid for each query:

```text
query image / query graph
rank 1 retrieved plan
rank 2 retrieved plan
...
rank 10 retrieved plan
```

Each retrieved plan should display:

```text
sample ID
final score
visual score
graph score
room match
adjacency match
```

## Hybrid retrieval script

Expected command:

```bash
python scripts/retrieve_hybrid.py \
  --query_id 000123 \
  --top_k 10 \
  --output outputs/hybrid_top10/000123
```
