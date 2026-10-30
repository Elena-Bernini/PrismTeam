# Dataset Strategy

## Purpose

This document defines which datasets are used, why they are relevant, and how they fit into the project pipeline.

## Primary dataset: RPLAN

RPLAN is the starting dataset for Phase 1.

Reasons:

- it provides raster floor plan images;
- it is widely used in floor plan generation and retrieval literature;
- it is used by Graph2Plan;
- it enables a fast vision retrieval baseline;
- it can be transformed into layout graphs for graph retrieval.

RPLAN is used for:

- image retrieval;
- room extraction;
- layout graph construction;
- partial layout masking;
- vision-vs-graph comparison;
- hybrid retrieval evaluation.

## Methodological reference: Graph2Plan

Graph2Plan is the main reference for graph extraction from RPLAN.

Useful elements:

- each room is represented as a node;
- node features include room type, room size, and room location;
- location is discretized using a 5 x 5 grid;
- room size is represented as an area ratio;
- edges encode adjacency between rooms;
- edge attributes encode spatial relations such as left-of, right-of, above, below, and diagonal variants;
- sparse user constraints are represented as partial layout graphs.

The project adapts these ideas for retrieval rather than generation.

## Schema reference: ResPlan

ResPlan is used as a reference for a cleaner vector-graph representation.

Useful elements:

- vector-based geometry;
- room polygons;
- room semantic labels;
- graph connectivity;
- typed edges such as door, open-arch, and shared-wall relations;
- JSON-serializable structure;
- compatibility between geometric and graph representations.

ResPlan can also be used as a secondary dataset if available in a usable form.

## Structural representation reference: HomeWorld

HomeWorld is useful for its structured floor plan representation, especially the K-D tree JSON format. This representation is not the first implementation priority, but it can inform later structural serialization or masked structure completion experiments.

Relevant ideas:

- floor plans can be serialized into structured JSON;
- hierarchical partitions can make layouts easier to validate;
- masked or corrupted layout states can be useful for learning robust correction or retrieval behavior.

## Future proprietary dataset: Maticad

The Maticad dataset is expected to become the most relevant resource for the second part of the project.

Expected use:

- room-level retrieval;
- furniture layout retrieval;
- spatial arrangement search;
- industrial validation of retrieved layouts.

## Other datasets to inspect for room-level retrieval

For the single-room/furniture-level stage, the project should inspect:

- 3D-FRONT;
- 3D-FUTURE;
- Structured3D;
- ZInD;
- InteriorNet;
- CubiCasa5K.

The goal is to identify which datasets provide furniture categories, object positions, top-down layouts, room types, and usable room-level structure.

## Dataset decision table

| Dataset | Immediate role | Strength | Limitation |
|---|---|---|---|
| RPLAN | Primary Phase 1 dataset | Raster images and annotations | Requires graph extraction |
| ResPlan | Schema and benchmark reference | Vector + graph representation | Smaller than RPLAN |
| Graph2Plan/RPLAN pipeline | Method reference | Concrete graph extraction strategy | Generation-focused paper |
| HomeWorld | Structural representation reference | K-D tree JSON and hierarchical representation | Whole-home generation scope |
| Maticad | Future industrial dataset | Most relevant to application | Not immediately available |
| 3D-FRONT / 3D-FUTURE | Future room-level study | Furniture and indoor scenes | Different task/domain |
