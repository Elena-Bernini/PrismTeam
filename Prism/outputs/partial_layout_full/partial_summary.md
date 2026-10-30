# Partial Layout Retrieval Summary

This report evaluates retrieval from incomplete floorplan queries.

## Aggregate Metrics

- Number of queries: `50`
- Original recall@K: `0.3600`
- Mean top-K compatibility: `0.8202`
- Mean top-1 compatibility: `0.8960`
- Mean hard-constraint rate@K: `0.9440`

## Notes

- The original complete layout is used as a proxy ground-truth completion.
- Compatibility is computed from known room types, known semantic adjacencies, spatial relations, grid locations, and candidate size.
- A high recall@K means that the system can recover the original complete layout from its partial version.
- A high compatibility score means that retrieved layouts satisfy the visible partial constraints.
