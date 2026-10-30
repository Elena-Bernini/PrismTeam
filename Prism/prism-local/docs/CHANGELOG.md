# Changelog relative to the uploaded Prism code

## Structural changes

- Split the flat five-script repository into two independent projects: `prism-local` and `prism-cluster`.
- Renamed scripts into numbered stage files under `scripts/`.
- Added `config/` with YAML files per stage and `room_zones.yaml`.
- Added `src/` shared helpers copied identically into both projects.
- Added local PowerShell wrappers and cluster SLURM job scripts.
- Added `.gitignore` and `requirements.txt`.

## Data/schema changes

- Added config-driven semantic `zone` per room.
- Added `scale_meters_per_pixel`, `area_sqm`, and `total_area_sqm` fields, with square-meter values left `null` when scale is unknown.
- Added layout-level total room area in pixels and ratio.
- Preserved existing similarity/retrieval algorithms; no metric formula was intentionally changed.

## Fixes and documented limitations

- Fixed README/doc references from the obsolete `post_preprocessing_retrieval.py` name to `02_retrieval_pipelines.py`.
- Kept the DINOv2 `torch.hub.load(..., trust_repo=True, skip_validation=True)` fix.
- Made removed rooms in partial-layout previews explicit with a `REMOVED` label.
- For `remove_type`, query selection now prefers layouts that actually contain the requested type, reducing fallback-random ambiguity.
- Investigated doors/windows and documented that reliable geometry is unavailable in the current RPLAN/Graph2Plan converted inputs. No fabricated window/door positions were added.

## Logging

- The legacy progress output is routed through Python logging in stage scripts for cleaner cluster logs while preserving the existing messages.
