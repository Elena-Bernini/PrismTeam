# RPLAN Post-Preprocessing Retrieval Report

This report summarizes graph and vision retrieval artifacts generated from the preprocessed RPLAN / Graph2Plan data.

## Configuration

- Dataset root: `data\raw\rplan\extracted`
- Processed root: `data\processed\rplan`
- Outputs root: `outputs\retrieval_vit_full`
- Pipeline: `vision`
- FAISS metric: `cosine`
- FAISS index type: `flat`
- Top-k: `10`

## Vision Retrieval Pipeline

- Items: `67453`
- Feature dimension: `768`
- Encoder: `{'backbone': 'timm:vit_base_patch16_224', 'weights': 'imagenet', 'device': 'cuda'}`
- Artifact directory: `data\processed\rplan\retrieval\vision_pipeline`
- Input representation: original rendered floorplan snapshots.

## Output Files

- Graph retrieval examples: `outputs\retrieval_vit_full\graph_pipeline`
- Vision retrieval examples: `outputs\retrieval_vit_full\vision_pipeline`
- Pipeline comparisons: `outputs\retrieval_vit_full\pipeline_comparison`

Each query produces a JSON file with ranking scores and a PNG grid with the query floorplan and its nearest neighbors.
