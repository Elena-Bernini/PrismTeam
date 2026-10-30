# Where did the old code go?

## PrismCodePier (Pier)

| Old file | New location | Notes |
|---|---|---|
| `scripts/01_inspect_rplan.py`, `cluster-code/preprocessing.py` (step 1) | `scripts/01_inspect_dataset.py` | + edge-column histogram, front-door check, polygon alignment |
| `scripts/02_convert_rplan_to_layout_json.py`, `preprocessing.py` (step 2) | `src/prism/datasets/rplan.py`, `scripts/02_convert_to_layout_json.py` | same field names; official taxonomy (the old `configs/rplan_type_mapping.json` had wrong names such as `PublicAreaType4`); + `unified_type`, `polygon_area_ratio`, `front_door` |
| `scripts/04_convert_layouts_to_graph_json.py`, `preprocessing.py` (step 3) | `src/prism/graph/build.py`, `scripts/03_build_graphs.py` | **relations recomputed** (see `docs/relations.md`); edges in both directions; FrontDoor node |
| `scripts/03_visualize_layout_json.py`, `06_visualize_graph_json.py` | `src/prism/common/viz.py` (+ the reports of phases 02/03) | |
| `scripts/05_rplan_statistics.py`, `preprocessing.py` (step 4) | `scripts/05_statistics_and_splits.py` | + splits, packing |
| `cluster-code/retrieval_pipelines.py` — graph descriptor | `src/prism/graph/descriptor.py`, `scripts/07_graph_embed_index.py` | + door features, connection and typed-adjacency features |
| `retrieval_pipelines.py` — vision encoders, colorhist | `src/prism/vision/encoders.py`, `scripts/08_vision_embed_index.py` | DINOv2 now through Hugging Face (as in Elena's code) |
| `retrieval_pipelines.py` — FAISS | `src/prism/retrieval/index.py` | NumPy fallback when FAISS is missing |
| `retrieval_pipelines.py` — queries, grids | `scripts/09_retrieval_full.py`, `src/prism/retrieval/visual.py` | |
| `cluster-code/partial_layout_retrieval.py` | `src/prism/partial/{masking,compatibility}.py`, `scripts/10_retrieval_partial.py` | runs on **every** pipeline (graph, GNN, vision via partial renders, hybrid); + door compatibility |
| `cluster-code/evaluate_retrieval.py` | `src/prism/evaluation/metrics.py`, `scripts/11_evaluate.py` | same metrics and architectural score; + Recall@K with an explicit relevance, door metrics, optional GED |
| `cluster-code/compare_pipeline_evaluations.py` | `scripts/12_compare_pipelines.py` | |

## Archivio (Elena)

| Old file | New location | Notes |
|---|---|---|
| `build_dataset_index.py` (`dataset_index.pkl` from `.mat`) | replaced by phases 02–05 | the `.mat` splits are not used yet; deterministic splits are in `splits.json` |
| `pipeline_vision/extract_vision_embeddings*.py` | `src/prism/vision/encoders.py` (`HFEncoder`, `dinov2_base`) | MPS / CUDA / CPU selected automatically |
| `pipeline_vision/build_faiss_index.py` | `src/prism/retrieval/index.py` | cosine = L2-normalise + inner product, as before |
| `pipeline_graph/build_graphs.py` | `src/prism/graph/features.py` | node features: type one-hot + area + centroid (as before) + bbox size + door features; **edge features** added |
| `pipeline_graph/train_gnn_gae.py` | `src/prism/graph/{models,train}.py`, `scripts/06_train_gnn.py` | GraphSAGE GAE kept as `graph.objective=gae`; + GCN, GAT, contrastive objective, train/val split, early stopping |
| `pipeline_graph/build_faiss_index_graphs.py` | `scripts/07_graph_embed_index.py` | |
| `incomplete_retrieval.py` | `src/prism/partial/masking.py` (`keep_connected`), `compatibility.py` (`f1_type_adjacency`) | |
| `hybrid_retrieval.py` | `src/prism/retrieval/fusion.py` (RRF, k=60) | used in phases 09 and 10 |
| `compare_pipelines.py`, `compare_visual.py`, `compare_hybrid_visual.py`, `visualize_*` | phases 09, 11, 12 reports | |
| `*.pkl`, `*.faiss`, `gnn_encoder_weights.pt`, `colored_images/` | not carried over | regenerated on the full dataset |
