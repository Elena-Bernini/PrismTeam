# Retrieval Evaluation Report

This report evaluates graph-based and vision-based floorplan retrieval results after preprocessing.
The evaluation does not use the embedding similarity alone. Instead, each retrieved floorplan is compared with the query floorplan using architectural descriptors extracted from the preprocessed `layout_json` and `graph_json` files.

## Inputs

- Processed dataset root: `data\processed\rplan`
- Retrieval results root: `outputs\retrieval_vit_full`
- Evaluation depth: top-10

## Main Metrics

- `architectural_score`: weighted aggregate of room-type similarity, room-count similarity, area similarity, adjacency similarity, relation similarity, grid similarity, and constraint satisfaction.
- `room_type_count_similarity`: generalized Jaccard similarity between room-type histograms.
- `adjacency_type_similarity`: generalized Jaccard similarity between semantic adjacency multisets, where edges are represented as room-type pairs.
- `relation_id_similarity`: generalized Jaccard similarity between Graph2Plan relation-id histograms.
- `constraint_score`: average satisfaction of simple query-derived constraints such as matching kitchen count, matching bathroom count, bedroom count within ±1, room count within ±1, and living-room coverage.
- `hard_constraint_result_rate`: fraction of retrieved results satisfying all simple constraints.

## Pipeline Summary

| Pipeline | Queries | Results | Architectural | Room types | Adjacency | Constraints | Hard constraints |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vision_pipeline | 10 | 100 | 0.6768 | 0.8055 | 0.5646 | 0.9160 | 0.6600 |

## Rank-Level Behavior

The rank-level table is saved as `metrics_by_rank.csv`. It can be used to check whether architectural quality decreases as retrieval rank increases.

## Query-Level Behavior

The query-level table is saved as `metrics_by_query.csv`. It reports mean and best top-k architectural scores for each query and pipeline.

## Output Files

- `metrics_per_result.csv`: one row per query-result pair.
- `metrics_by_query.csv`: top-k aggregates for each query and pipeline.
- `metrics_by_rank.csv`: rank-wise aggregate metrics.
- `pipeline_summary.json`: machine-readable summary.
- `pipeline_summary.md`: this report.
- `plots/summary_bar.png`: compact pipeline comparison plot.
- `plots/rank_curve.png`: rank-wise architectural score plot.
