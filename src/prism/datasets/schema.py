"""Unified data schema shared by every dataset adapter.

Layout JSON (one file per floor plan, ``layouts_json/<id>.json``)::

    {
      "schema_version": "1.1",
      "id": "rplan_75342",                       # global id = <dataset>_<source id>
      "source_id": "75342",                      # id inside the original dataset
      "dataset": "rplan",                        # rplan | resplan
      "coordinate_system": {"width": 256, "height": 256, "origin": "top_left",
                            "x_axis": "right", "y_axis": "down", "units": "px"},
      "boundary": [[x, y], ...],                 # exterior outline (closed implicitly)
      "boundary_area_ratio": 0.53,
      "front_door": {                            # null if the dataset has none
          "segment": [[x1, y1], [x2, y2]],
          "center": [x, y], "width_px": 12.0,
          "orientation": "horizontal" | "vertical",
          "facing": "north" | "south" | "east" | "west" | "unknown",
          "entrance_room_id": 3, "entrance_distance_px": 2.0,
          "source": "g2p_boundary_rows_0_1"
      },
      "rooms": [{
          "id": 0, "room_type_id": 7, "room_type_name": "SecondRoom",
          "unified_type": "bedroom",
          "bbox": [x0, y0, x1, y1], "width": 50, "height": 53,
          "area_pixels": 2650, "area_ratio": 0.0404,     # bbox area / canvas area (as before)
          "polygon_area_ratio": 0.0398,                  # true polygon area / canvas area
          "centroid": [cx, cy], "grid_location_5x5": [gx, gy],
          "polygon": [[x, y], ...],
          "is_entrance_room": false
      }, ...],
      "source_edges": [{"source": 0, "target": 1, "raw_value": 7}, ...],
      "raw": {...}
    }

ResPlan layouts additionally carry ``walls`` (polygons with holes), ``doors``,
``windows``, ``geometry_hints.wall_thickness_px``, ``extra_polygons`` on rooms
made of several parts and ``source_key`` (``bedroom_0`` ...) on each room.

Graph JSON (``graphs_json/<id>.json``) - see :mod:`prism.graph.build`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.1"

# -----------------------------------------------------------------------------
# Spatial relations (ours, recomputed geometrically - see docs/relations.md).
# relation(u, v) describes the SOURCE room u relative to the TARGET room v,
# in image coordinates (y grows downward, so "above" means smaller y).
# -----------------------------------------------------------------------------
RELATIONS: List[str] = [
    "left_of",      # 0
    "right_of",     # 1
    "above",        # 2
    "below",        # 3
    "left_above",   # 4
    "right_above",  # 5
    "left_below",   # 6
    "right_below",  # 7
    "inside",       # 8
    "surrounding",  # 9
]
RELATION_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(RELATIONS)}

INVERSE_RELATION: Dict[str, str] = {
    "left_of": "right_of", "right_of": "left_of",
    "above": "below", "below": "above",
    "left_above": "right_below", "right_below": "left_above",
    "right_above": "left_below", "left_below": "right_above",
    "inside": "surrounding", "surrounding": "inside",
}

# Edge connection types. RPLAN only gives "wall" (rooms adjacent across a wall,
# door presence unknown). ResPlan has wall/door/window geometry, so its edges are
# classified as door (a door polygon in the shared wall), arch (open passage, no
# wall), window (e.g. room -> balcony through a window) or wall. The virtual
# front-door node is connected with "entrance".
CONNECTIONS: List[str] = ["wall", "door", "arch", "window", "entrance", "unknown"]

# Datasets that can be merged in the "combined" workspace. Layout ids are always
# namespaced as "<dataset>_<source id>" so that they never collide.
DATASETS: List[str] = ["rplan", "resplan"]
COMBINED = "combined"


def make_id(dataset: str, source_id: Any) -> str:
    """Global layout id, e.g. ``rplan_75342`` or ``resplan_123``."""
    return f"{dataset}_{source_id}"


def dataset_of(layout_id: str) -> str:
    """Dataset prefix of a global layout id (``rplan_75342`` -> ``rplan``)."""
    for name in DATASETS:
        if str(layout_id).startswith(name + "_"):
            return name
    raise ValueError(f"Layout id '{layout_id}' has no known dataset prefix {DATASETS}")


def source_id_of(layout_id: str) -> str:
    """Original id inside its dataset (``rplan_75342`` -> ``75342``)."""
    return str(layout_id)[len(dataset_of(layout_id)) + 1:]
CONNECTION_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CONNECTIONS)}

# Name used for the virtual node that represents the front door.
FRONT_DOOR_NODE_TYPE = "FrontDoor"

# Graph2Plan relation table as reported in Pier's preprocessing README. It is
# kept ONLY to label the raw third column of ``sample.edge`` in diagnostic
# reports. Graph2Plan's own code ignores that column and recomputes the
# relation from boxes, so we do not use it as ground truth.
G2P_RELATION_GUESS: Dict[int, str] = {
    0: "left-above", 1: "left-below", 2: "left-of", 3: "above", 4: "inside",
    5: "surrounding", 6: "below", 7: "right-of", 8: "right-above", 9: "right-below",
}


class Taxonomy:
    """Room-type lookups, built from ``configs/room_types.yaml``.

    Two levels of room types exist:

    - ``fine``: dataset-specific names (RPLAN ``MasterRoom``, ``SecondRoom`` ...;
      ResPlan category keys ``bedroom``, ``living`` ...);
    - ``unified``: the cross-dataset taxonomy (``bedroom``, ``bathroom`` ...).

    Every layout room and graph node stores both (``room_type_name`` and
    ``unified_type``). Code that compares, counts or one-hot-encodes room types
    uses :meth:`key`, which returns the name at the active level. The level is
    ``taxonomy.level`` in the config; ``auto`` = fine for one dataset, unified for
    ``combined`` (fine names of different datasets are not comparable).
    """

    def __init__(self, cfg: Dict[str, Any], dataset: str = "rplan", level: Optional[str] = None) -> None:
        """Read both datasets' taxonomies and resolve the active level."""
        rt = cfg["room_types"]
        self.dataset = dataset
        self.unified_colors: Dict[str, str] = dict(rt["unified_types"])
        self.unified_names: List[str] = list(self.unified_colors.keys())
        # RPLAN official ids -> fine name / unified type / colour.
        self.rplan: Dict[int, Dict[str, str]] = {
            int(k): {"name": v[0], "unified": v[1], "color": v[2]} for k, v in rt["rplan"].items()}
        self.rplan_extra: Dict[int, str] = {int(k): v for k, v in rt.get("rplan_extra", {}).items()}
        # ResPlan category key -> unified type.
        self.resplan: Dict[str, str] = dict(rt["resplan"])
        self.resplan_non_room: List[str] = list(rt.get("resplan_non_room", []))
        self.fine_colors: Dict[str, str] = {v["name"]: v["color"] for v in self.rplan.values()}
        for name, unified in self.resplan.items():
            self.fine_colors.setdefault(name, self.unified_colors.get(unified, self.unified_colors["other"]))

        requested = level or cfg.get("taxonomy", {}).get("level", "auto")
        if requested == "auto":
            requested = "unified" if dataset == COMBINED else "fine"
        if requested not in {"fine", "unified"}:
            raise ValueError(f"Unknown taxonomy level '{requested}'")
        if dataset == COMBINED and requested == "fine":
            raise ValueError("taxonomy.level=fine is not allowed for the combined dataset: RPLAN and ResPlan "
                             "fine types are different vocabularies. Use unified (or auto).")
        self.level = requested

    # ----------------------------------------------------- dataset conversion
    def rplan_name(self, type_id: int) -> str:
        """RPLAN fine name of an official id (``Unknown_<id>`` if not a room id)."""
        if type_id in self.rplan:
            return self.rplan[type_id]["name"]
        return self.rplan_extra.get(type_id, f"Unknown_{type_id}")

    def rplan_unified(self, type_id: int) -> str:
        """Unified type of an RPLAN id (``other`` if unknown)."""
        return self.rplan.get(type_id, {}).get("unified", "other")

    def resplan_unified(self, category: str) -> str:
        """Unified type of a ResPlan category key (``other`` if unknown)."""
        return self.resplan.get(category, "other")

    # --------------------------------------------------------- level-aware API
    def vocab(self) -> List[str]:
        """Room-type vocabulary at the active level (used for counts and one-hot features)."""
        if self.level == "unified":
            return list(self.unified_names)
        if self.dataset == "rplan":
            return [self.rplan[i]["name"] for i in sorted(self.rplan)]
        if self.dataset == "resplan":
            return list(self.resplan.keys())
        raise ValueError(f"No fine vocabulary for dataset '{self.dataset}'")

    def key(self, item: Dict[str, Any]) -> str:
        """Type name of a room / node at the active level."""
        if self.level == "unified":
            return str(item.get("unified_type", "other"))
        return str(item.get("room_type_name", item.get("unified_type", "other")))

    def is_bedroom(self, item: Dict[str, Any]) -> bool:
        """True for every bedroom-like room (RPLAN Master/Second/Guest/Child room, ResPlan bedroom)."""
        return item.get("unified_type") == "bedroom"

    def color(self, item: Dict[str, Any], palette: str = "unified") -> str:
        """Render colour of a room: by unified type (default, same for every dataset) or by fine type."""
        if palette == "fine" and item.get("room_type_name") in self.fine_colors:
            return self.fine_colors[item["room_type_name"]]
        return self.unified_colors.get(item.get("unified_type", "other"), self.unified_colors["other"])
