"""Prism: vision vs graph representations for semantic floor-plan retrieval.

The package is organised by responsibility:

- ``prism.common``      configuration, paths, devices, I/O, HTML reports
- ``prism.datasets``    dataset adapters (RPLAN, ResPlan) and the unified schema
- ``prism.geometry``    polygons, adjacency, spatial relations, front door
- ``prism.graph``       graph construction, descriptors, GNN models and training
- ``prism.vision``      rendering of semantic images and vision encoders
- ``prism.retrieval``   vector indexes (FAISS or NumPy) and rank fusion
- ``prism.partial``     masking strategies and compatibility re-ranking
- ``prism.evaluation``  architectural metrics, Recall@K, aggregation

Heavy dependencies (torch, faiss, transformers) are imported lazily so that the
preprocessing phases run with NumPy/Pillow/Matplotlib only.
"""

__version__ = "0.1.0"
