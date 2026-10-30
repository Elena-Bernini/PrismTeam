# Project Overview

## Project title

Semantic Architectural Search: Vision and Graph Representations for Floor Plan Retrieval

## Objective

The project builds a retrieval system for residential floor plans. Given a complete or partial floor plan query, the system returns the top-k most compatible layouts from the dataset.

The core task is apartment-level retrieval. The system must compare two representation families:

1. Vision-based representations, where the input is a raster floor plan image.
2. Graph-based representations, where rooms are nodes and spatial relationships are edges.

The final system combines both representations through hybrid retrieval and re-ranking.

## Main use case

A designer starts from a full or incomplete floor plan. The system retrieves examples from a dataset that are visually, semantically, and topologically compatible with the query.

Typical query types:

- complete raster floor plan;
- masked raster floor plan;
- complete layout graph;
- partial layout graph;
- room-count and adjacency constraints;
- hybrid query with both raster and graph information.

Expected output:

- top-10 retrieved floor plans;
- visual similarity score;
- graph similarity score;
- room composition score;
- adjacency compatibility score;
- final hybrid score.

## Required Phase 1 pipeline

The project follows the CVCS Maticad Phase 1 requirement: compare vision and graph representations for semantic architectural search.

```text
RPLAN dataset
    -> dataset audit
    -> raster preprocessing
    -> vision retrieval baseline
    -> graph extraction
    -> graph retrieval baseline
    -> partial layout masking
    -> partial layout retrieval
    -> hybrid retrieval
    -> semantic and topological evaluation
```

## Final Phase 1 architecture

```text
Query floor plan
    |
    |-- Raster branch
    |       -> image preprocessing
    |       -> CLIP / ViT / DINO encoder
    |       -> image embedding
    |       -> FAISS image index
    |
    |-- Graph branch
    |       -> room extraction
    |       -> layout graph construction
    |       -> graph feature vector or GNN embedding
    |       -> FAISS graph index
    |
    |-- Constraint branch
            -> room counts
            -> room type histogram
            -> area ratios
            -> required adjacency checks

Candidate union
    -> score normalization
    -> semantic and topological re-ranking
    -> top-10 retrieved floor plans
```

## Key design principles

The system should not retrieve layouts only because they look similar as images. It should also consider architectural compatibility:

- number of rooms;
- room categories;
- area distribution;
- room location;
- adjacency between rooms;
- spatial relations between rooms;
- compatibility with partial queries.

## Main deliverables

- RPLAN parser and visualization tools;
- internal JSON schema for floor plans;
- layout graph extraction pipeline;
- vision retrieval baseline;
- symbolic graph retrieval baseline;
- GNN retrieval model;
- partial layout masking module;
- hybrid re-ranking module;
- evaluation scripts;
- qualitative top-k visualization grids;
- final PDF report and repository documentation.
