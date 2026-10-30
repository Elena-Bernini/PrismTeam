# RPLAN / Graph2Plan Preprocessing Report

Generated at: `2026-08-03T18:20:44`

## Input and Output Paths

- Dataset root: `data\raw\rplan\extracted`
- Processed root: `data\processed\rplan_unified`
- Audit root: `outputs\dataset_audit_unified`

## Converted Data

- Available PKL samples: `74995`
- Converted layouts: `74995`
- Converted graphs: `74995`
- Layout JSON directory: `data\processed\rplan_unified\layouts_json`
- Graph JSON directory: `data\processed\rplan_unified\graphs_json`
- Metadata directory: `data\processed\rplan_unified\metadata`

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

- Number of layouts: `74995`
- Room count per plan: `{'min': 4, 'max': 8, 'mean': 6.788679122924805, 'median': 7.0, 'std': 0.9123387932777405}`
- Edge count per plan: `{'min': 3, 'max': 18, 'mean': 10.18250560760498, 'median': 10.0, 'std': 2.2162599563598633}`

### Room type counts

```json
{
  "MasterRoom": 74685,
  "Bathroom": 90148,
  "LivingRoom": 74995,
  "SecondRoom": 92851,
  "Kitchen": 72188,
  "Balcony": 80293,
  "ChildRoom": 3657,
  "StudyRoom": 13913,
  "Storage": 3127,
  "DiningRoom": 1220,
  "Wall-in": 971,
  "GuestRoom": 798,
  "Entrance": 271
}
```

### Relation counts

```json
{
  "left-of": 134178,
  "below": 113162,
  "left-above": 53482,
  "surrounding": 2897,
  "right-below": 38860,
  "above": 153524,
  "left-below": 49091,
  "right-of": 106156,
  "inside": 66520,
  "right-above": 45767
}
```

## Visual Quality Control

- Layout preview count: `50`
- Graph preview count: `20`
- Layout preview directory: `outputs\dataset_audit_unified\layout_previews`
- Graph preview directory: `outputs\dataset_audit_unified\graph_previews`

Preview images are not model inputs. They are only generated for debugging and documentation.

## Main Output Files

- `outputs\dataset_audit_unified\rplan_file_audit.json`
- `outputs\dataset_audit_unified\train_converted_sample_report.json`
- `outputs\dataset_audit_unified\rplan_processed_statistics.json`
- `outputs\dataset_audit_unified\preprocessing_summary.json`
- `outputs\dataset_audit_unified\preprocessing_report.md`
