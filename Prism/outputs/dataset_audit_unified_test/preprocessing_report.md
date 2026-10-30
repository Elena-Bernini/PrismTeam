# RPLAN / Graph2Plan Preprocessing Report

Generated at: `2026-08-03T18:11:03`

## Input and Output Paths

- Dataset root: `data\raw\rplan\extracted`
- Processed root: `data\processed\rplan_unified_test`
- Audit root: `outputs\dataset_audit_unified_test`

## Converted Data

- Available PKL samples: `74995`
- Converted layouts: `200`
- Converted graphs: `200`
- Layout JSON directory: `data\processed\rplan_unified_test\layouts_json`
- Graph JSON directory: `data\processed\rplan_unified_test\graphs_json`
- Metadata directory: `data\processed\rplan_unified_test\metadata`

## Coordinate System

The converted layouts use a 256 x 256 image coordinate system.

- Origin: top-left
- x-axis: grows to the right
- y-axis: grows downward
- Edge relation: source room relative to target room

This is why visualizations invert the y-axis with `ylim(256, 0)`.

## Room Type Mapping

| ID | Room type |
|---:|---|
| 0 | LivingRoom |
| 1 | MasterRoom |
| 2 | Kitchen |
| 3 | Bathroom |
| 4 | DiningRoom |
| 5 | ChildRoom |
| 6 | StudyRoom |
| 7 | SecondRoom |
| 8 | GuestRoom |
| 9 | Balcony |
| 10 | Entrance |
| 11 | Storage |
| 12 | Wall-in |

## Spatial Relation Mapping

| ID | Relation |
|---:|---|
| 0 | left-above |
| 1 | left-below |
| 2 | left-of |
| 3 | above |
| 4 | inside |
| 5 | surrounding |
| 6 | below |
| 7 | right-of |
| 8 | right-above |
| 9 | right-below |

## Dataset Statistics

- Number of layouts: `200`
- Room count per plan: `{'min': 4, 'max': 8, 'mean': 6.820000171661377, 'median': 7.0, 'std': 0.9474176168441772}`
- Edge count per plan: `{'min': 5, 'max': 15, 'mean': 10.199999809265137, 'median': 10.0, 'std': 2.2671568393707275}`

### Room type counts

```json
{
  "SecondRoom": 255,
  "Bathroom": 246,
  "Kitchen": 191,
  "MasterRoom": 197,
  "Balcony": 222,
  "LivingRoom": 200,
  "Entrance": 1,
  "StudyRoom": 33,
  "ChildRoom": 8,
  "DiningRoom": 3,
  "Storage": 5,
  "GuestRoom": 2,
  "Wall-in": 1
}
```

### Relation counts

```json
{
  "right-of": 292,
  "right-above": 114,
  "inside": 209,
  "above": 389,
  "below": 303,
  "left-of": 337,
  "left-above": 136,
  "left-below": 141,
  "right-below": 109,
  "surrounding": 10
}
```

## Visual Quality Control

- Layout preview count: `30`
- Graph preview count: `10`
- Layout preview directory: `outputs\dataset_audit_unified_test\layout_previews`
- Graph preview directory: `outputs\dataset_audit_unified_test\graph_previews`

Preview images are not model inputs. They are only generated for debugging and documentation.

## Main Output Files

- `outputs\dataset_audit_unified_test\rplan_file_audit.json`
- `outputs\dataset_audit_unified_test\train_converted_sample_report.json`
- `outputs\dataset_audit_unified_test\rplan_processed_statistics.json`
- `outputs\dataset_audit_unified_test\preprocessing_summary.json`
- `outputs\dataset_audit_unified_test\preprocessing_report.md`
