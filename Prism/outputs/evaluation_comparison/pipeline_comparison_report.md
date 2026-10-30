# Retrieval Pipeline Evaluation Comparison

This report compares retrieval pipelines using the evaluation outputs produced after preprocessing and retrieval.
The comparison is based on per-result architectural metrics computed from the preprocessed layout and graph JSON files.

## Input Evaluations

- `resnet50`: `outputs\evaluation_resnet50`
- `dinov2_vitb14`: `outputs\evaluation_dinov2`
- `vit_base_patch16_224`: `outputs\evaluation_vit`

## Summary Table

| Pipeline | Architectural Score | Constraint Score | Faiss Similarity |
| --- | --- | --- | --- |
| Graph | 0.7998 | 0.9940 | 0.8352 |
| Vision DINOv2 | 0.6979 | 0.9520 | 0.9886 |
| Vision ResNet50 | 0.6872 | 0.9360 | 0.9744 |
| Vision ViT | 0.6768 | 0.9160 | 0.9587 |

## Interpretation Guide

- `architectural_score` is the aggregate score used to summarize architectural retrieval quality.
- `room_type_similarity` measures whether the retrieved layout has a similar room-type composition.
- `adjacency_similarity` measures whether the retrieved layout preserves similar semantic room adjacencies.
- `relation_similarity` measures whether spatial relation distributions are similar.
- `constraint_score` and `hard_constraint_satisfied` measure whether simple architectural constraints are respected.
- `faiss_similarity` is the vector-space similarity used for retrieval; it should not be interpreted as architectural quality by itself.

## Recommended Use

Use this comparison to decide which representation is more suitable for semantic architectural search.
A visual encoder may produce high vector similarity while still retrieving layouts with different room composition or topology.
A graph-based representation is expected to perform better when the target is room-level semantic and topological compatibility.
