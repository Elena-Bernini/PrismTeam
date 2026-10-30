"""Unified data schema shared by every dataset adapter.

Layout JSON (one file per floor plan, ``layouts_json/<id>.json``)::

    {
      "schema_version": "1.1",
      "id": "75342",
      "dataset": "rplan",
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

Graph JSON (``graphs_json/<id>.json``) - see :mod:`prism.graph.build`.
"""

from __future__ import annotations

from typing import Any, Dict, List

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
# door presence unknown); ResPlan distinguishes door / arch / wall; the virtual
# front-door node is connected with "entrance".
CONNECTIONS: List[str] = ["wall", "door", "arch", "entrance", "unknown"]
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
    """Room-type lookups for one dataset, built from ``configs/room_types.yaml``."""

    def __init__(self, cfg: Dict[str, Any], dataset: str = "rplan") -> None:
        """Read the fine and unified taxonomies for ``dataset``."""
        rt = cfg["room_types"]
        self.dataset = dataset
        self.unified_colors: Dict[str, str] = dict(rt["unified_types"])
        self.unified_names: List[str] = list(self.unified_colors.keys())
        self.fine: Dict[int, Dict[str, str]] = {}
        if dataset == "rplan":
            for key, (name, unified, color) in rt["rplan"].items():
                self.fine[int(key)] = {"name": name, "unified": unified, "color": color}
            self.extra = {int(k): v for k, v in rt.get("rplan_extra", {}).items()}
        elif dataset == "resplan":
            # ResPlan has no numeric fine ids: we enumerate its categories.
            for i, (name, unified) in enumerate(rt["resplan"].items()):
                self.fine[i] = {"name": name, "unified": unified,
                                "color": self.unified_colors.get(unified, "#d9d9d9")}
            self.extra = {}
        else:
            raise ValueError(f"Unknown dataset '{dataset}'")
        self.fine_ids: List[int] = sorted(self.fine.keys())
        self.name_to_id: Dict[str, int] = {v["name"]: k for k, v in self.fine.items()}

    def name(self, type_id: int) -> str:
        """Fine type name for an id (``Unknown_<id>`` if not in the taxonomy)."""
        if type_id in self.fine:
            return self.fine[type_id]["name"]
        return self.extra.get(type_id, f"Unknown_{type_id}")

    def unified(self, type_id: int) -> str:
        """Unified type for a fine id (``other`` if unknown)."""
        return self.fine.get(type_id, {}).get("unified", "other")

    def color(self, type_id: int) -> str:
        """Render colour for a fine id."""
        return self.fine.get(type_id, {}).get("color", self.unified_colors["other"])

    def unified_color(self, unified: str) -> str:
        """Render colour for a unified type."""
        return self.unified_colors.get(unified, self.unified_colors["other"])

    def is_bedroom(self, type_id: int) -> bool:
        """True for every bedroom-like fine type (Master/Second/Guest/Child room in RPLAN)."""
        return self.unified(type_id) == "bedroom"

    def vocab(self, kind: str) -> List[str]:
        """Return the node-type vocabulary used as one-hot features (``fine`` or ``unified``)."""
        if kind == "fine":
            return [self.name(i) for i in self.fine_ids]
        if kind == "unified":
            return list(self.unified_names)
        raise ValueError(f"Unknown vocabulary kind '{kind}'")
