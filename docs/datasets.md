# Working with RPLAN and ResPlan together

The default is `dataset: combined`: one gallery with both datasets. Every phase
accepts `--dataset combined | rplan | resplan`.

## How a run is organised

| Phases | With `--dataset combined` |
|---|---|
| 01–04 (inspect, convert, graphs, renders) | run once per dataset, into `data/processed/<dataset>/<mode>/` |
| 05 (statistics, splits) | each dataset's split first (RPLAN: hash of the id; ResPlan: official `split.json`), then the merged split and an **alignment report** in `outputs/combined/<mode>/05_statistics_and_splits/` |
| 06–12 (GNN, indexes, retrieval, evaluation) | on the merged gallery; artifacts in `artifacts/combined/<mode>/`, reports in `outputs/combined/<mode>/` |

In preview mode `--n` is the sample size **per dataset** (default 200 + 200).

Layout ids are always prefixed with the dataset (`rplan_75342`,
`resplan_5012`), so they never collide and every result says where a plan
comes from.

## What was done to make the two datasets comparable

| Aspect | RPLAN | ResPlan | Shared representation |
|---|---|---|---|
| Coordinates | 256 px canvas (Graph2Plan) | metres, y up | ResPlan fitted into the same 256 px canvas (`resplan.fit_ratio`), y flipped |
| Room types | 13 ids (MasterRoom, SecondRoom, ...) | category keys (bedroom, living, ...) | both stored; **unified** types (`room_types.yaml`) used whenever both datasets are mixed (`taxonomy.level: auto`) |
| Living room | one polygon | several polygons possible | one room with `extra_polygons` (as ResPlan's own `living_0`) |
| Room geometry | rooms tile the outline, meeting on wall centre lines | rooms separated by real walls, balconies outside `inner` | ResPlan turned into the same partition (`resplan.room_partition`, `src/prism/datasets/resplan_geometry.py`); original polygons kept in `source_polygons` |
| Walls / doors / windows | not available | available | ResPlan edges labelled door / arch / window / wall; connection features **off** in combined mode |
| Front door | rows 0-1 of `boundary` | `front_door` polygon | same `front_door` block, FrontDoor node, yellow segment in renders |
| Adjacency radius | `geometry.wall_dilation_px` | same rule on the partition; door / arch / window labels use the real walls | identical adjacency rule |
| Split | none → hash of the id | official `split.json` | merged; augmented ResPlan copies skipped by default (`resplan.include_augmented`) |
| Renders | unified palette | unified palette | identical colours per unified type |

## ResPlan geometry (why the renders now look like RPLAN)

ResPlan stores rooms as the free space between walls; drawn as they are they
give double walls, balconies floating next to the plan and outlines with small
notches. Phase 02 therefore builds an RPLAN-style **partition**
(`src/prism/datasets/resplan_geometry.py`):

1. rooms grow simultaneously into the walls until neighbours meet half-way;
2. the outline is closed, small unlabelled pockets inside it are given to a
   neighbour (light wells larger than `partition.fill_holes` stay holes and are
   drawn as outside);
3. narrow protrusions / notches are removed, x and y coordinates closer than
   `partition.snap x wall_depth` are merged (small steps disappear);
4. the cells are rebuilt as an edge-matched coverage and simplified together.

`python scripts/resplan_geometry_check.py` shows original / before / after for a
sample of plans next to RPLAN renders and measures the grey share (gaps between
rooms), detached pieces and overlaps. On 300 real plans: grey share 0.0067 ->
0.0 (RPLAN 0.0), plans with detached pieces 16 -> 0, overlaps 0, ~30 ms / plan.
Garden, parking and pool are not rooms of the apartment and are skipped
(`resplan.include_outdoor`). The whole step can be switched off with
`--set resplan.room_partition=false`.

## What to check in the reports

1. **Phase 01 (`--dataset resplan`)**: keys flagged as unknown. If they are room
   categories, add them to `configs/room_types.yaml -> resplan`. The units
   section shows whether coordinates are metres.
2. **Phase 03 (ResPlan)**: walls/doors/windows next to the graph, and edge
   colours by connection type.
3. **Phase 05 (combined)**: the alignment table and charts. If the outline size
   or room-area medians differ a lot between datasets, tune `resplan.fit_ratio`
   and re-run 02–05.
4. **Phase 11 (combined)**: `same_dataset_rate@K` by query dataset. If a pipeline
   almost never retrieves from the other dataset while relevant plans exist
   there, it is matching dataset style (scale, drawing) rather than layout.

## Running on one dataset only

`--dataset rplan` or `--dataset resplan` gives a separate workspace (fine room
types, connection features on for ResPlan). Both can coexist with the combined
run because every workspace has its own folders.
