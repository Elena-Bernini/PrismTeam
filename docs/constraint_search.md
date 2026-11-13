# Constraint search (phase 13)

Given a user's **requirements**, phase 13 returns the **k most plausible floor
plans** of the gallery (RPLAN, ResPlan or both), each with an explanation of how
well it meets every requirement.

```bash
python scripts/13_constraint_search.py --spec configs/specs/example_2bed.yaml
python scripts/13_constraint_search.py --area 65-90 --area-target 75 --rooms 4-6 \
    --type bedroom=2 --type bathroom=1+ --type kitchen=1 --type study=0 --k 10
python scripts/13_constraint_search.py --spec configs/specs/example_family.yaml --dataset resplan --mode full
```

It is CPU-only (no torch or FAISS) and needs phases 02–05 of the same
workspace (phase 07's `graph_descriptor` index is used for diversity when it
exists). Outputs go to `outputs/<dataset>/<mode>/13_constraint_search/<name>/`:
`spec_normalised.json`, `results.json`, `results.csv`, `near_misses.json`,
`report.html`. Cached data (catalog, calibration, priors) goes to
`artifacts/<dataset>/<mode>/constraints/`.

## Requirements

| Requirement | Meaning |
|---|---|
| `area_m2` | floor area in m²: `target` and/or `min` / `max`; `basis: net` (sum of rooms) or `gross` (outline) |
| `rooms` | total number of rooms (front door excluded) |
| `room_types` | count per **unified** room type (`bedroom`, `bathroom`, `kitchen`, `living`, `dining`, `study`, `balcony`, `entrance`, `storage`, `stair`, `other`); `0` = must not be present. Fine names (e.g. `MasterRoom`) only with `--dataset rplan` / `resplan` |
| `entrance` | unified type of the room behind the front door |
| `adjacent` | pairs of unified types that should share an edge in the room graph |

Each requirement is **soft** (scored, `weight`) or **hard** (`hard: true`:
plans that fail it are never returned, only shown as near misses).

### Example request (`configs/specs/example_2bed.yaml`)

```yaml
name: two_bed_flat
k: 10
datasets: [rplan, resplan]
split: all
area_m2: {basis: net, target: 75, min: 65, max: 90, hard: false, weight: 2.0}
rooms: {min: 4, max: 6, hard: true}
room_types:
  bedroom:  {min: 2, max: 2, hard: true, weight: 2.0}
  bathroom: "1+"
  kitchen:  1
  balcony:  {min: 1, weight: 0.5}
  study:    0
entrance: {type: living, weight: 0.5}
adjacent:
  - {pair: [kitchen, living], weight: 0.5}
ranking: {plausibility_weight: 0.3, diversity: 0.2, near_misses: 5}
```

A second example, `configs/specs/example_family.yaml`, asks for 3 bedrooms
(hard), at least 2 bathrooms (hard) and a balcony.

### Shorthands

| Written | Meaning |
|---|---|
| `2` | exactly 2 (`{min: 2, max: 2}`) |
| `"2+"` | at least 2 |
| `"2-3"` | between 2 and 3 |
| `"<=1"` | at most 1 |
| `0` | must not be present |

Defaults: `hard: false`, `weight: 1.0`, `area_m2.basis: net`,
`area_m2.tolerance_m2` = 10% of the target (or of the interval midpoint)
(`constraints.area_tolerance_ratio`).

### CLI flags

`--spec FILE`, `--k`, `--area 65-90`, `--area-target 75`, `--area-basis net|gross`,
`--rooms 4-6`, `--type bedroom=2` (repeatable), `--entrance living`,
`--adjacent kitchen-living` (repeatable), `--hard area|rooms|type:bedroom|entrance|adjacent:kitchen-living`
(repeatable), `--name`, `--rebuild-catalog`. CLI flags override the matching
keys of `--spec`. Without `--spec` and without requirement flags, the request in
`constraints.default_spec` is used (this is what `run_all.py` does).

Invalid requests stop with a clear message: unknown room types (the valid ones
are listed), `min > max`, negative counts, `k < 1`, room-type minimums that
cannot fit in a hard `rooms.max`.

## Scores

All scores are in [0, 1].

```
interval_satisfaction(x, lo, hi, s) = exp(-d / s),  d = distance from x to [lo, hi]  (1 inside)
target_closeness(x, t, s)           = exp(-0.5 ((x - t) / s)²)
```

- counts (`rooms`, `room_types`): `interval_satisfaction` with `s = 1` room;
- area: interval and target → `0.5 interval + 0.5 closeness`; interval only →
  interval; target only → closeness; `s = tolerance_m2`. A hard area given only
  as a target means `[target - tolerance, target + tolerance]`;
- entrance: 1 if the entrance room has the wanted type, else 0 (no front door: 0.5, "unknown");
- adjacent: 1 if the pair is an edge of the room graph, else 0.

```
constraint_score = Σ w_c s_c / Σ w_c                (evaluable requirements only)
plausibility     = Σ w_i p_i / Σ w_i
final            = (1 - plausibility_weight) · constraint_score + plausibility_weight · plausibility
```

**Plausibility** is independent of the request and uses the gallery's own
statistics (`constraints.plausibility_components`, defaults in brackets):

| Component | Definition |
|---|---|
| `size` (0.4) | mean over rooms of `exp(-0.5 z²)`, `z` = log room area standardised with the mean / std of its type |
| `shape` (0.2) | same on log(aspect ratio of the room's bounding box) |
| `reachability` (0.3) | share of rooms reachable from the entrance room through the room graph |
| `living_hub` (0.1) | 1 if a living room touches ≥ 50% of the other rooms or is the entrance room, else that share |

Components are a registry (`prism.constraints.scoring.PLAUSIBILITY`), so new
ones can be added.

**Ranking.** Plans passing every hard requirement are sorted by `final` (ties:
constraint score, then id). With `diversity` λ > 0, the top `20 k` are re-picked
with Maximal Marginal Relevance on the graph-descriptor vectors,
`argmax (1 - λ) final_i - λ max_j cos(v_i, v_j)`, so the top-k are not
near-duplicates; the result order is then the MMR pick order. If fewer than k
plans pass, all of them are returned and the report says so: violating plans are
never used to fill the list. Everything is deterministic (no RNG).

## Areas in m² (read this before trusting a number)

Every plan records its `area_source`.

- **ResPlan** (`"dataset"`). The real `ResPlan.pkl` (2026-07-28 release) stores
  coordinates normalised **per plan** (every plan is scaled into a ~256-unit
  box), so the geometry alone has no metric scale. The plan's `area` field is
  consistent m² (48–511 m²; the wall thickness it implies is 0.22 m ± 15%
  across plans), while `net_area` is not (0 in more than 25% of plans, up to
  2.5·10⁷ in others). Therefore `gross_area_m2 = area`, the plan's m² per canvas
  px² is `area / outline px²`, each room is its original wall-free polygon
  (`source_polygons`) × that factor, and `net_area_m2` is the sum of the rooms.
  `net_area` is used only when `area` is missing and `net_area` looks valid;
  without both, and with `constraints.resplan_units: m`, the geometry is read as
  metres (`"geometry_metric"`). The report shows this cross-check.
- **RPLAN** (`"calibrated"`). The Graph2Plan pickle has no metric scale. RPLAN
  m² are **estimates**: canvas px² × a scale calibrated on ResPlan room sizes,
  `scale = median over calibration_types of (median ResPlan m²(t) / median RPLAN px²(t))`
  (default types: bedroom, bathroom, kitchen). They are always shown as "≈" and
  can be overridden with `--set constraints.rplan_m2_per_px2=<value>`. RPLAN
  rooms tile the outline (no walls), so RPLAN net and gross areas coincide.
  With `--dataset rplan` the ResPlan workspace of the same mode is read for the
  calibration; without ResPlan data and without a manual scale, RPLAN areas are
  unknown.
- **Unknown area**: a soft area requirement is skipped (its weight leaves the
  average) and marked "unknown"; a hard one keeps the plan unless
  `require_known_area: true`.

## Reading the report

1. **Request** and KPIs: gallery size, plans passing every hard requirement,
   results, results with unknown area, share per dataset.
2. **Hard-requirement funnel**: plans removed by each hard requirement, alone
   and in order. If nothing matches, **Why nothing matches** lists the
   requirements that the most plans fail - relax those first.
3. **Top k**: renders with rank, id, final score, area ("≈" = estimated) and
   the requested room counts, then one explanation table per result
   (`status`: met / partly / violated / unknown).
4. **Near misses**: the best plans that fail a hard requirement, with what they violate.
5. **Area in m²**: area sources, the RPLAN calibration (scale, per-type
   ratios), the ResPlan units check and the area histogram per dataset.
6. Score distribution of the candidates and plausibility components of the top k.

## Limitations

- RPLAN areas are calibrated estimates; ResPlan per-room m² inherit the
  approximations of the partition outline (≈ 3–5% below the full wall outline).
- Plausibility is a heuristic prior (room sizes, shapes, reachability, living
  hub) fitted on the gallery itself, not a learned model.
- In preview mode the catalog and the priors cover only the preview subset.
- `adjacent` uses unified types and the room graph of phase 03 (ResPlan's own
  graph for ResPlan); it does not know where doors are for RPLAN.
