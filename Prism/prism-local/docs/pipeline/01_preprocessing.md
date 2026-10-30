# 01 Preprocessing

Converts raw Graph2Plan/RPLAN pickle data to per-layout JSON, graph JSON, metadata, statistics, QC previews, and a report.

Reads `config/01_preprocessing.yaml`. Safe smoke-test override: `--limit 200 --layout-preview-n 10 --graph-preview-n 5`.

New schema fields: per-room `zone`, per-room `area_sqm`, layout-level `scale_meters_per_pixel`, `total_room_area_pixels`, `total_room_area_ratio`, `total_area_sqm`. Square-meter fields stay null for default RPLAN config.
