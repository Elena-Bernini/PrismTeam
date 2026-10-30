"""Unit tests for adjacency, spatial relations, front door and partial graphs.

Run with:  python -m pytest tests -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.common.config import load_config  # noqa: E402
from prism.datasets.schema import INVERSE_RELATION, RELATIONS  # noqa: E402
from prism.geometry.adjacency import geometric_adjacency  # noqa: E402
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
    pairs = geometric_adjacency(rooms, boxes, 256, wall_dilation_px=3, min_contact_px=5)
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
    arr = graph_to_arrays(graph, tax, "fine", True)
    nd, ed = feature_dims(tax, "fine")
    assert arr["x"].shape == (len(graph["nodes"]), nd) and arr["edge_attr"].shape[1] == ed
    sub = subgraph(graph, [0, 1])
    assert {n["id"] for n in sub["nodes"] if n["kind"] == "room"} == {0, 1}
