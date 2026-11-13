"""Unit tests for adjacency, spatial relations, front door and partial graphs.

Run with:  python -m pytest tests -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.common.config import load_config  # noqa: E402
from prism.datasets.schema import INVERSE_RELATION, RELATIONS  # noqa: E402
from prism.geometry.adjacency import RoomMasks, classify_connections, geometric_adjacency  # noqa: E402
from prism.geometry.front_door import describe_front_door, door_facing  # noqa: E402
from prism.geometry.polygon import bbox_to_polygon, dilate_square, point_in_polygon, polygon_area  # noqa: E402
from prism.geometry.relations import spatial_relation  # noqa: E402

import numpy as np  # noqa: E402


def rect(x0, y0, x1, y1):
    """Rectangle polygon helper."""
    return bbox_to_polygon([x0, y0, x1, y1])


def test_relations_axis_and_diagonal():
    """Side-by-side, stacked and diagonal rooms get the expected labels."""
    a, b = [10, 10, 50, 50], [54, 12, 90, 48]          # b to the right of a, same rows
    assert spatial_relation(a, b) == "left_of"
    assert spatial_relation(b, a) == "right_of"
    c = [12, 54, 48, 90]                                 # c below a
    assert spatial_relation(a, c) == "above"
    assert spatial_relation(c, a) == "below"
    d = [54, 54, 90, 90]                                 # d diagonal bottom-right of a
    assert spatial_relation(a, d) == "left_above"
    assert spatial_relation(d, a) == "right_below"


def test_relations_containment():
    """A small box inside the bbox of a large (L-shaped) room is inside / surrounding."""
    big, small = [0, 0, 100, 100], [60, 60, 90, 90]
    assert spatial_relation(small, big) == "inside"
    assert spatial_relation(big, small) == "surrounding"


def test_inverse_is_consistent():
    """The inverse relation table is an involution covering every relation."""
    for r in RELATIONS:
        assert INVERSE_RELATION[INVERSE_RELATION[r]] == r


def test_adjacency_shared_wall_vs_corner():
    """Rooms across a 4 px wall are adjacent; rooms touching only at a corner are not."""
    rooms = [rect(10, 10, 50, 50), rect(54, 10, 90, 50), rect(54, 54, 90, 90)]
    boxes = [[10, 10, 50, 50], [54, 10, 90, 50], [54, 54, 90, 90]]
    pairs = geometric_adjacency(RoomMasks([[r] for r in rooms], 256, 3), boxes, min_contact_px=5)
    assert (0, 1) in pairs and (1, 2) in pairs
    assert (0, 2) not in pairs          # diagonal neighbours only meet at a corner
    assert pairs[(0, 1)] >= 30          # shared wall ~ 40 px


def test_dilation_square():
    """Dilating a single pixel by r gives a (2r+1)^2 square."""
    m = np.zeros((11, 11), dtype=bool)
    m[5, 5] = True
    assert dilate_square(m, 2).sum() == 25


def test_front_door_facing_and_entrance():
    """Door on the bottom wall faces south and opens into the room above it."""
    boundary = rect(10, 10, 100, 100)
    seg = [[40, 100], [55, 100]]
    assert door_facing(seg, boundary) == "south"
    rooms = [rect(12, 12, 98, 50), rect(12, 54, 98, 98)]
    fd = describe_front_door(seg, boundary, rooms, ["Bedroom", "LivingRoom"])
    assert fd["entrance_room_id"] == 1 and fd["orientation"] == "horizontal"


def test_polygon_basics():
    """Area and point-in-polygon on a square."""
    sq = rect(0, 0, 10, 10)
    assert polygon_area(sq) == 100
    assert point_in_polygon((5, 5), sq) and not point_in_polygon((15, 5), sq)


def test_graph_build_and_subgraph():
    """End-to-end graph construction on a synthetic plan, then a partial subgraph."""
    import random

    from prism.datasets.rplan import convert_sample
    from prism.datasets.schema import Taxonomy
    from prism.datasets.synthetic import make_sample
    from prism.graph.build import build_graph, subgraph
    from prism.graph.features import feature_dims, graph_to_arrays

    cfg = load_config("local")
    tax = Taxonomy(cfg, "rplan")
    layout = convert_sample(make_sample(3, random.Random(1)), tax, cfg)
    assert layout["front_door"] is not None
    graph = build_graph(layout, cfg)
    kinds = [n["kind"] for n in graph["nodes"]]
    assert kinds.count("front_door") == 1
    # every edge appears in both directions with inverse relations
    seen = {(e["source"], e["target"]): e["relation"] for e in graph["edges"]}
    for (u, v), r in seen.items():
        assert seen[(v, u)] == INVERSE_RELATION[r]
    arr = graph_to_arrays(graph, tax, True)
    nd, ed = feature_dims(tax)
    assert arr["x"].shape == (len(graph["nodes"]), nd) and arr["edge_attr"].shape[1] == ed
    sub = subgraph(graph, [0, 1])
    assert {n["id"] for n in sub["nodes"] if n["kind"] == "room"} == {0, 1}


def test_connection_types_door_arch_wall():
    """A door in the shared wall -> door; no wall between rooms -> arch; plain wall -> wall."""
    rooms = [rect(10, 10, 50, 50), rect(54, 10, 90, 50), rect(10, 54, 50, 90)]
    masks = RoomMasks([[r] for r in rooms], 128, 3)
    pairs = [(0, 1), (0, 2)]
    walls = [{"exterior": rect(6, 6, 94, 94), "holes": rooms}]
    doors = [rect(50, 25, 54, 35)]                         # door in the wall between rooms 0 and 1
    conn = classify_connections(pairs, masks, walls, doors, [], wall_thickness_px=4)
    assert conn[(0, 1)] == "door" and conn[(0, 2)] == "wall"
    open_walls = [{"exterior": rect(6, 6, 94, 94), "holes": [rect(10, 10, 90, 50), rooms[2]]}]  # no wall 0|1
    conn = classify_connections(pairs, masks, open_walls, [], [], wall_thickness_px=4)
    assert conn[(0, 1)] == "arch"
    assert classify_connections(pairs, masks, None, None, None, 4)[(0, 1)] == "wall"   # RPLAN: no geometry


def test_resplan_conversion_and_combined_taxonomy():
    """Synthetic ResPlan plan: namespaced id, front door, unified taxonomy for combined."""
    import random

    from prism.datasets.resplan import convert_plan, normalize_keys
    from prism.datasets.schema import Taxonomy, dataset_of
    from prism.datasets.synthetic import make_resplan_plan
    from prism.graph.build import build_graph

    cfg = load_config("local")
    plan = normalize_keys(make_resplan_plan(0, random.Random(3)))
    layout = convert_plan(plan, 0, Taxonomy(cfg, "resplan"), cfg)
    assert dataset_of(layout["id"]) == "resplan" and layout["front_door"]["entrance_room_id"] is not None
    assert all(0 <= p[0] <= 256 and 0 <= p[1] <= 256 for r in layout["rooms"] for p in r["polygon"])
    graph = build_graph(layout, cfg)
    assert any(e["connection"] == "door" for e in graph["edges"])
    tax = Taxonomy(cfg, "combined")
    assert tax.level == "unified" and tax.key({"room_type_name": "MasterRoom", "unified_type": "bedroom"}) == "bedroom"


def test_resplan_partition_tiles_the_outline():
    """Partitioned ResPlan rooms tile the outline (no gaps, no overlaps) like RPLAN rooms."""
    import random

    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from prism.datasets.resplan import convert_plan, normalize_keys
    from prism.datasets.schema import Taxonomy
    from prism.datasets.synthetic import make_resplan_plan

    cfg = load_config("local")
    for seed in range(3):
        plan = normalize_keys(make_resplan_plan(seed, random.Random(seed)))
        layout = convert_plan(plan, 0, Taxonomy(cfg, "resplan"), cfg)
        assert layout["geometry_hints"]["room_partition"]
        rooms = [unary_union([Polygon(p) for p in [r["polygon"]] + r["extra_polygons"]]).buffer(0)
                 for r in layout["rooms"]]
        outline = Polygon(layout["boundary"]).buffer(0)
        assert abs(unary_union(rooms).area / outline.area - 1) < 0.01       # rooms cover the outline
        assert sum(r.area for r in rooms) / unary_union(rooms).area < 1.01  # and do not overlap
        assert all(r.get("source_polygons") for r in layout["rooms"])     # originals kept for the labels


def test_resplan_graph_is_the_adjacency_source():
    """ResPlan's own graph drives adjacency and connection types; fallback edges are dropped."""
    import random

    from prism.datasets.resplan import convert_plan, normalize_keys
    from prism.datasets.schema import Taxonomy
    from prism.datasets.synthetic import make_resplan_plan
    from prism.graph.build import build_graph

    cfg = load_config("local")
    tax = Taxonomy(cfg, "resplan")
    plan = normalize_keys(make_resplan_plan(0, random.Random(5)))     # idx 0 -> has a fallback edge
    layout = convert_plan(plan, 0, tax, cfg)
    info = layout["source_graph_info"]
    assert info["has_graph"] and info["unmatched_nodes"] == 0 and info["front_door_edges"] == 1
    assert all(e["source_type"] != "fallback" for e in layout["source_edges"])
    assert layout["front_door"]["entrance_source"] == "resplan_graph"
    graph = build_graph(layout, cfg)
    assert graph["adjacency"]["source"] == "source"
    src_pairs = {(e["source"], e["target"]) for e in layout["source_edges"]}
    room_pairs = {(e["source"], e["target"]) for e in graph["edges"]
                  if e["connection"] != "entrance" and e["source"] < e["target"]}
    assert room_pairs == src_pairs
    assert all("geometric_connection" in e for e in graph["edges"])
    # Without a graph the same plan falls back to geometric adjacency.
    plan.pop("graph")
    layout2 = convert_plan(plan, 0, tax, cfg)
    assert build_graph(layout2, cfg)["adjacency"]["source"] == "geometric"
    assert layout2["front_door"]["entrance_source"] == "closest_room"


def test_resplan_node_matched_by_geometry():
    """A graph node with a non-standard name is matched to a room through its geometry."""
    import random

    from prism.datasets.resplan import convert_plan, normalize_keys
    from prism.datasets.schema import Taxonomy
    from prism.datasets.synthetic import make_resplan_plan

    cfg = load_config("local")
    for idx in range(3, 60, 7):                                       # idx % 7 == 3 renames a bathroom node
        plan = normalize_keys(make_resplan_plan(idx, random.Random(idx)))
        if "room_without_standard_name" in plan["graph"].nodes():
            info = convert_plan(plan, idx, Taxonomy(cfg, "resplan"), cfg)["source_graph_info"]
            assert info["matched_by_geometry"] == 1 and info["unmatched_nodes"] == 0
            return
    raise AssertionError("no synthetic plan with a renamed node")
