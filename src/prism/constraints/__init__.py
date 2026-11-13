"""Constraint search (phase 13): the k most plausible floor plans for a user's requirements.

- :mod:`prism.constraints.spec`     requests (YAML / CLI), shorthands, validation
- :mod:`prism.constraints.catalog`  per-plan attribute table, m² (dataset / calibrated), priors
- :mod:`prism.constraints.scoring`  requirement satisfaction, plausibility, final score
- :mod:`prism.constraints.search`   hard filtering, ranking, MMR diversity, near misses

CPU only: no torch or FAISS needed.
"""
