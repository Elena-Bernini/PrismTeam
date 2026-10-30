# Graph Extraction

## Purpose

This document defines how to convert each floor plan into a layout graph. The graph representation follows the ideas of Graph2Plan and the vector-graph schema described in ResPlan.

## Graph definition

Each floor plan is represented as:

```text
G = (V, E)
```

where:

- `V` is the set of rooms;
- `E` is the set of spatial or connectivity relationships between rooms.

## Node representation

Each room is a node.

Recommended node attributes:

```text
room_id
room_type
area
area_ratio
centroid_x
centroid_y
bbox_x
bbox_y
bbox_w
bbox_h
aspect_ratio
grid_location_5x5
```

Graph2Plan encodes room type, room location, and room size. The project should preserve this design because it gives a compact but semantically meaningful graph representation.

## Room type normalization

RPLAN may contain fine-grained room types. The initial implementation should normalize them into a stable taxonomy.

Suggested taxonomy:

```text
LivingRoom
Bedroom
Bathroom
Kitchen
DiningRoom
Balcony
Storage
Entrance
StudyRoom
Other
```

A second taxonomy can preserve more fine-grained bedroom types if available.

## Location encoding

Use two location encodings:

1. Continuous centroid coordinates normalized to `[0, 1]`.
2. Discrete 5 x 5 grid location inspired by Graph2Plan.

The grid encoding is useful for symbolic retrieval and for GNN node features.

## Size encoding

For each room:

```text
area_ratio = room_area / total_interior_area
```

Also store binned size class for symbolic retrieval:

```text
small / medium / large
```

## Edge extraction

Edges are created between rooms that are spatially adjacent or directly connected.

Possible edge sources:

1. Door-based adjacency: two rooms share a door.
2. Boundary contact: two room masks touch or are closer than a threshold.
3. Shared wall: two rooms share a wall segment but have no direct opening.
4. Open passage: two spaces connect through a large opening.

The first implementation can use mask proximity and shared boundary length. If door annotations are available, door-based edge typing should be added.

## Edge attributes

Recommended edge attributes:

```text
source
target
edge_type
spatial_relation
centroid_distance
shared_boundary_length
relative_angle
```

Suggested `edge_type` values:

```text
door
open_arch
shared_wall
proximity
unknown
```

Suggested `spatial_relation` values:

```text
left_of
right_of
above
below
left_above
right_above
left_below
right_below
inside
surrounding
```

## Spatial relation computation

Given source centroid `(cx_i, cy_i)` and target centroid `(cx_j, cy_j)`:

```text
dx = cx_j - cx_i
dy = cy_j - cy_i
```

Then assign:

- `right_of` if `dx > threshold` and `abs(dy)` is small;
- `left_of` if `dx < -threshold` and `abs(dy)` is small;
- `below` or `above` depending on image coordinate convention;
- diagonal classes when both `abs(dx)` and `abs(dy)` are significant.

The exact coordinate convention must be documented in the code.

## Graph JSON schema

```json
{
  "id": "000001",
  "nodes": [
    {
      "id": 0,
      "room_type": "Bedroom",
      "area_ratio": 0.18,
      "centroid": [0.42, 0.35],
      "bbox": [0.30, 0.20, 0.25, 0.22],
      "grid_location_5x5": [2, 1]
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "edge_type": "proximity",
      "spatial_relation": "left_of"
    }
  ]
}
```

## PyTorch Geometric object

Each graph should also be exported as a PyTorch Geometric `Data` object.

Expected fields:

```python
Data(
    x=node_feature_tensor,
    edge_index=edge_index_tensor,
    edge_attr=edge_feature_tensor,
    y=optional_label,
    sample_id=sample_id
)
```

## Debug visualization

For each processed graph, generate a visualization with:

- floor plan raster;
- room nodes at centroids;
- edges drawn between adjacent rooms;
- node labels as room types;
- edge colors or labels for relation types.

Output:

```text
outputs/graph_debug/<sample_id>_graph_overlay.png
```
