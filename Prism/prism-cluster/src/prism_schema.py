# SHARED between prism-local and prism-cluster — keep in sync manually.
"""Schema, area, and semantic-zone helpers for Prism RPLAN layouts.

The helpers in this module deliberately do not assume a real-world scale for
RPLAN/Graph2Plan samples. The converted RPLAN data is normalized to a 256x256
canvas; square-meter fields are therefore only populated when an explicit,
verified meters-per-pixel scale is supplied by configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

DEFAULT_ROOM_ZONES: Dict[str, str] = {
    "MasterRoom": "Private Zone",
    "ChildRoom": "Private Zone",
    "SecondRoom": "Private Zone",
    "GuestRoom": "Private Zone",
    "Bathroom": "Private Zone",
    "Wall-in": "Private Zone",
    "LivingRoom": "Social Zone",
    "DiningRoom": "Social Zone",
    "Kitchen": "Social Zone",
    "StudyRoom": "Social Zone",
    "Balcony": "External Zone",
    "Entrance": "Service/Circulation Zone",
    "Storage": "Service/Circulation Zone",
}


def _minimal_zone_yaml_parser(text: str) -> Dict[str, str]:
    """Parse the restricted room_zones.yaml structure without PyYAML."""
    mapping: Dict[str, str] = {}
    current_zone: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith((" ", "\t")) and line.endswith(":"):
            current_zone = line[:-1].strip().strip('"\'')
            continue
        stripped = line.strip()
        if stripped.startswith("-") and current_zone:
            room_name = stripped[1:].strip().strip('"\'')
            if room_name:
                mapping[room_name] = current_zone
    return mapping


def load_room_zone_mapping(path: Optional[Path]) -> Dict[str, str]:
    """Load a room_type_name -> zone mapping from YAML, falling back to defaults."""
    if path is None or not Path(path).exists():
        return dict(DEFAULT_ROOM_ZONES)

    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
        mapping: Dict[str, str] = {}
        for zone, room_names in data.items():
            if isinstance(room_names, (list, tuple)):
                for room_name in room_names:
                    mapping[str(room_name)] = str(zone)
        return mapping or dict(DEFAULT_ROOM_ZONES)
    except Exception:
        parsed = _minimal_zone_yaml_parser(text)
        return parsed or dict(DEFAULT_ROOM_ZONES)


def get_room_zone(room_type_name: str, mapping: Mapping[str, str]) -> str:
    """Return the configured semantic zone for a fine-grained room type."""
    return mapping.get(room_type_name, "Unknown Zone")


def compute_area_sqm(area_pixels: float, scale_meters_per_pixel: Optional[float]) -> Optional[float]:
    """Convert pixel area to square meters only when a trusted scale is provided."""
    if scale_meters_per_pixel is None:
        return None
    if scale_meters_per_pixel <= 0:
        raise ValueError("scale_meters_per_pixel must be positive or null/omitted.")
    return float(area_pixels) * float(scale_meters_per_pixel) ** 2


def get_room_type_id(room: Mapping[str, Any]) -> int:
    """Read a room type id from either the final or legacy schema."""
    if "room_type_id" in room:
        return int(room["room_type_id"])
    if "raw_type_id" in room:
        return int(room["raw_type_id"])
    raise KeyError("Room is missing both 'room_type_id' and legacy 'raw_type_id'.")


def get_room_type_name(room: Mapping[str, Any], fallback_by_id: Optional[Mapping[int, str]] = None) -> str:
    """Read a room type name from either the final or legacy schema."""
    if room.get("room_type_name") is not None:
        return str(room["room_type_name"])
    if room.get("type_name") is not None:
        return str(room["type_name"])
    if fallback_by_id is not None:
        return str(fallback_by_id.get(get_room_type_id(room), f"Unknown_{get_room_type_id(room)}"))
    return f"Unknown_{get_room_type_id(room)}"
