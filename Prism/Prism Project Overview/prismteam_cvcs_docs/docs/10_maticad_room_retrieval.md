# Maticad Room-Level Retrieval Preparation

## Purpose

The Maticad dataset will be used for the second part of the project, focused on single-room retrieval based on furniture layout and spatial arrangement.

This stage should be prepared while Phase 1 apartment-level retrieval is being implemented.

## Expected task

Given a room layout or furniture arrangement, retrieve similar rooms from the database.

Input may include:

```text
room raster or top-view image
CAD room representation
room type
room dimensions
furniture categories
furniture positions
spatial arrangement graph
```

Output:

```text
top-10 similar rooms
furniture layout similarity score
room geometry similarity score
functional arrangement score
```

## Candidate representation

For each room, store:

```json
{
  "room_id": "room_0001",
  "room_type": "bedroom",
  "boundary": [[x0, y0], [x1, y1], [x2, y2]],
  "area": 18.5,
  "objects": [
    {
      "id": 0,
      "category": "bed",
      "bbox": [x, y, w, h],
      "centroid": [cx, cy],
      "rotation": 90
    }
  ],
  "relations": [
    {
      "source": 0,
      "target": 1,
      "relation": "near"
    }
  ]
}
```

## Room-level graph

Room-level retrieval can use a graph similar to the floor plan graph:

```text
nodes = furniture objects
edges = spatial relations between objects
room boundary = global context
```

Node features:

```text
object category
bbox
centroid
rotation
size
support relation if available
```

Edge features:

```text
near
left_of
right_of
above
below
facing
aligned_with
on_top_of
against_wall
```

## Datasets to inspect before Maticad arrival

```text
3D-FRONT
3D-FUTURE
Structured3D
ZInD
InteriorNet
CubiCasa5K
```

For each dataset, check:

```text
room-level annotations
furniture categories
object bounding boxes
object rotations
top-down render availability
3D coordinates
license and accessibility
conversion difficulty
```

## First room-level baseline

Once data is available:

```text
room top-view image
    -> vision encoder
    -> embedding
    -> FAISS
    -> top-10 similar rooms
```

Second baseline:

```text
furniture graph
    -> graph features or GNN
    -> embedding
    -> FAISS
    -> top-10 similar rooms
```

Hybrid room retrieval:

```text
room image similarity
+ room geometry similarity
+ furniture graph similarity
+ functional zone compatibility
```

## Connection to apartment-level Phase 1

The same infrastructure can be reused:

```text
FAISS indexing
embedding extraction
masking protocol
retrieval metrics
hybrid re-ranking
qualitative grids
```

The main difference is the graph semantics:

```text
apartment-level: rooms are nodes
room-level: furniture objects are nodes
```
