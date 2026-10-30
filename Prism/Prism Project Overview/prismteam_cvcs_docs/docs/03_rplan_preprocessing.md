# RPLAN Preprocessing

## Purpose

This document defines how RPLAN should be parsed and converted into the internal representation used by the retrieval system.

## Input data

RPLAN provides raster floor plan images and annotations. The preprocessing stage must identify:

- building boundary;
- interior area;
- room instances;
- room semantic labels;
- doors, if available;
- walls and structural components, if available;
- front door or entrance information, if available.

## Output data structure

Each processed floor plan should be stored in a dedicated sample folder:

```text
data/processed/rplan/<sample_id>/
    image.png
    layout.json
    graph.json
    graph.pt
    preview.png
    masks/
```

## Internal layout JSON schema

```json
{
  "id": "000001",
  "source": "RPLAN",
  "image_path": "image.png",
  "width": 256,
  "height": 256,
  "boundary": [[x0, y0], [x1, y1], [x2, y2]],
  "rooms": [
    {
      "id": 0,
      "type": "Bedroom",
      "area": 1234,
      "area_ratio": 0.18,
      "bbox": [x, y, w, h],
      "centroid": [cx, cy],
      "grid_location_5x5": [gx, gy],
      "polygon": [[x0, y0], [x1, y1], [x2, y2]]
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "relation": "adjacent",
      "spatial_relation": "left_of",
      "edge_type": "door_or_proximity"
    }
  ],
  "metadata": {
    "num_rooms": 5,
    "room_type_histogram": {
      "Bedroom": 2,
      "Bathroom": 1,
      "Kitchen": 1,
      "LivingRoom": 1
    }
  }
}
```

## Room extraction tasks

1. Load annotation channels.
2. Extract connected components for room instances.
3. Assign a semantic room type to each instance.
4. Compute bbox, centroid, area, and area ratio.
5. Map the centroid to a 5 x 5 grid cell.
6. Extract polygon approximation if possible.
7. Save room-level features in `layout.json`.

## Boundary extraction tasks

1. Extract the building boundary mask.
2. Convert boundary mask to polygon or contour.
3. Normalize coordinates.
4. Save boundary information in `layout.json`.
5. If entrance/front-door information is present, save it separately.

## Visualization checks

For every preprocessing run, generate preview images showing:

- original raster floor plan;
- room instance masks;
- room labels;
- bounding boxes;
- centroids;
- graph edges overlaid on the floor plan.

Expected output:

```text
outputs/preprocessing_debug/<sample_id>_preview.png
```

## Dataset audit outputs

The first audit should produce:

```text
outputs/audit/dataset_stats.json
outputs/audit/room_type_mapping.json
outputs/audit/room_count_distribution.csv
outputs/audit/room_area_distribution.csv
outputs/audit/sample_grid.png
```

## Quality checks

The parser should verify:

- every room has one semantic label;
- every room has positive area;
- no room has an empty mask;
- bbox coordinates remain inside the image;
- the sum of room areas is compatible with the interior area;
- graph nodes have matching room IDs;
- graph edges do not reference missing nodes.
