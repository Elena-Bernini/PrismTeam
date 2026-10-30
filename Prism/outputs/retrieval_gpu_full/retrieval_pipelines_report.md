# RPLAN Post-Preprocessing Retrieval Report

This report summarizes graph and vision retrieval artifacts generated from the preprocessed RPLAN / Graph2Plan data.

## Configuration

- Dataset root: `data\raw\rplan\extracted`
- Processed root: `data\processed\rplan`
- Outputs root: `outputs\retrieval_gpu_full`
- Pipeline: `both`
- FAISS metric: `cosine`
- FAISS index type: `flat`
- Top-k: `10`

## Graph Retrieval Pipeline

- Items: `74995`
- Feature dimension: `100`
- Artifact directory: `data\processed\rplan\retrieval\graph_pipeline`
- Input representation: room nodes, typed edges, area statistics, centroid distribution, boundary descriptors.

## Vision Retrieval Pipeline

- Items: `67453`
- Feature dimension: `2048`
- Encoder: `{'backbone': 'resnet50', 'weights': 'imagenet', 'device': 'cuda'}`
- Artifact directory: `data\processed\rplan\retrieval\vision_pipeline`
- Input representation: original rendered floorplan snapshots.

## Output Files

- Graph retrieval examples: `outputs\retrieval_gpu_full\graph_pipeline`
- Vision retrieval examples: `outputs\retrieval_gpu_full\vision_pipeline`
- Pipeline comparisons: `outputs\retrieval_gpu_full\pipeline_comparison`

Each query produces a JSON file with ranking scores and a PNG grid with the query floorplan and its nearest neighbors.
