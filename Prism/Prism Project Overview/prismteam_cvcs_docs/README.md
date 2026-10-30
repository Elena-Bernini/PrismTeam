# PrismTeam - Semantic Architectural Search

This repository contains the implementation plan and documentation for the CVCS Maticad project on semantic architectural search for residential floor plans.

The project focuses on apartment-level floor plan retrieval in Phase 1. Given a complete or partial floor plan query, the system retrieves the top-k most compatible layouts from a dataset. The work compares vision-based and graph-based representations, then combines them in a hybrid retrieval system.

The initial dataset is RPLAN because it provides raster floor plan images and annotations suitable for fast vision retrieval and graph extraction. ResPlan is used as a schema and benchmarking reference for vector-graph representations. Graph2Plan is used as the main methodological reference for extracting and using layout graphs from RPLAN. Maticad data will be integrated later, especially for room-level retrieval based on furniture layout and spatial arrangement.

## Main objectives

1. Build a vision-based floor plan retrieval baseline using pretrained visual encoders and FAISS.
2. Convert RPLAN floor plans into structured layout graphs.
3. Build a graph-based retrieval baseline using handcrafted graph features and FAISS.
4. Train or evaluate a GNN-based graph encoder for retrieval.
5. Implement partial layout retrieval by degrading complete plans through raster masks and graph masks.
6. Evaluate vision, graph, and hybrid retrieval using retrieval, semantic, and topological metrics.
7. Prepare the second stage for Maticad single-room retrieval when the proprietary dataset becomes available.

## Documentation structure

- `docs/01_project_overview.md`: project goals, scope, and final system architecture.
- `docs/02_dataset_strategy.md`: RPLAN, ResPlan, Maticad, and indoor scene datasets.
- `docs/03_rplan_preprocessing.md`: RPLAN audit, parsing, normalization, and internal schema.
- `docs/04_graph_extraction.md`: layout graph construction inspired by Graph2Plan and ResPlan.
- `docs/05_vision_retrieval.md`: image encoder baseline and FAISS retrieval.
- `docs/06_graph_retrieval.md`: symbolic graph retrieval and GNN retrieval.
- `docs/07_partial_layout_masking.md`: raster and graph masking protocol.
- `docs/08_hybrid_retrieval.md`: score fusion, candidate union, and re-ranking.
- `docs/09_evaluation.md`: metrics, experimental protocol, and result tables.
- `docs/10_maticad_room_retrieval.md`: future single-room retrieval stage.
- `docs/11_implementation_roadmap.md`: step-by-step implementation roadmap.

## Expected first demo

The first working demo should take one RPLAN floor plan as query and display the top-10 retrieved layouts.

Expected command:

```bash
python scripts/retrieve_vision.py \
  --query_id 000123 \
  --index data/indexes/vision_dino.faiss \
  --top_k 10 \
  --output outputs/vision_top10/000123
```

The second demo should use a graph query instead of an image query:

```bash
python scripts/retrieve_graph.py \
  --query_id 000123 \
  --index data/indexes/graph_features.faiss \
  --top_k 10 \
  --output outputs/graph_top10/000123
```

The main Phase 1 demo should compare complete and partial queries across vision, graph, and hybrid retrieval.
