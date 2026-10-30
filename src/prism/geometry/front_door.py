"""Front-door extraction and description.

RPLAN (Graph2Plan pickle): the front door is the segment formed by the first two
rows of ``sample.boundary`` (columns 0-1 are x, y). This is exactly how
Graph2Plan's own ``floorplan.py`` reads it (``door = boundary[:2, :2]``) and it is
the yellow segment drawn on the exterior wall of the RPLAN snapshots.

ResPlan: the front door is a separate ``front_door`` polygon; the adapter turns
it into a segment along its long axis with :func:`segment_from_door_polygon`.

The description stored in the layout JSON contains the segment, its centre and
width, its orientation, the direction the door faces (outward normal of the
exterior wall) and the entrance room (the room the door opens into).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.prism.geometry.polygon import point_in_polygon, segment_polygon_distance


def door_segment_from_g2p_boundary(boundary: Any) -> Optional[List[List[float]]]:
    """Return ``[[x1, y1], [x2, y2]]`` from a Graph2Plan boundary array (rows 0 and 1)."""
    try:
        arr = np.asarray(boundary, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    return [[float(arr[0, 0]), float(arr[0, 1])], [float(arr[1, 0]), float(arr[1, 1])]]


def segment_from_door_polygon(points: Sequence[Sequence[float]]) -> Optional[List[List[float]]]:
    """Collapse a thin door polygon (ResPlan) into the segment along its long axis."""
    if len(points) < 2:
        return None
    p = np.asarray(points, dtype=np.float64)[:, :2]
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    if (x1 - x0) >= (y1 - y0):
        yc = (y0 + y1) / 2.0
        return [[float(x0), float(yc)], [float(x1), float(yc)]]
    xc = (x0 + x1) / 2.0
    return [[float(xc), float(y0)], [float(xc), float(y1)]]


def door_facing(segment: Sequence[Sequence[float]], boundary: Sequence[Sequence[float]], probe_px: float = 4.0) -> str:
    """Direction the door faces (``north``/``south``/``east``/``west``/``unknown``).

    We probe a point ``probe_px`` away from the door centre on each side of the
    segment; the side that falls outside the building outline is "out". North is
    the top of the image (smaller y).
    """
    (x1, y1), (x2, y2) = segment[0][:2], segment[1][:2]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    horizontal = abs(x2 - x1) >= abs(y2 - y1)
    if horizontal:
        candidates = {"north": (cx, cy - probe_px), "south": (cx, cy + probe_px)}
    else:
        candidates = {"west": (cx - probe_px, cy), "east": (cx + probe_px, cy)}
    outside = [name for name, pt in candidates.items() if not point_in_polygon(pt, boundary)]
    return outside[0] if len(outside) == 1 else "unknown"


def find_entrance_room(segment: Sequence[Sequence[float]], room_polygons: Sequence[Sequence[Sequence[float]]],
                       room_type_names: Sequence[str]) -> tuple:
    """Return ``(room_index, distance_px)`` of the room closest to the door segment.

    Ties within 1 px prefer rooms named ``Entrance`` then ``LivingRoom``.
    """
    best_idx, best_dist = None, float("inf")
    preference = {"Entrance": 0, "entrance": 0, "LivingRoom": 1, "living": 1}
    for i, poly in enumerate(room_polygons):
        if len(poly) < 3:
            continue
        d = segment_polygon_distance(segment[0], segment[1], poly)
        if d < best_dist - 1.0:
            best_idx, best_dist = i, d
        elif abs(d - best_dist) <= 1.0 and best_idx is not None:
            if preference.get(room_type_names[i], 9) < preference.get(room_type_names[best_idx], 9):
                best_idx, best_dist = i, min(d, best_dist)
    return best_idx, best_dist


def describe_front_door(
    segment: Optional[Sequence[Sequence[float]]],
    boundary: Sequence[Sequence[float]],
    room_polygons: Sequence[Sequence[Sequence[float]]],
    room_type_names: Sequence[str],
    max_dist_px: float = 12.0,
    min_width_px: float = 2.0,
    source: str = "unknown",
) -> Optional[Dict[str, Any]]:
    """Build the ``front_door`` block of the layout JSON (``None`` if no valid door)."""
    if segment is None:
        return None
    (x1, y1), (x2, y2) = segment[0][:2], segment[1][:2]
    width = float(np.hypot(x2 - x1, y2 - y1))
    if width < min_width_px:
        return None
    entrance_idx, dist = find_entrance_room(segment, room_polygons, room_type_names)
    entrance_ok = entrance_idx is not None and dist <= max_dist_px
    return {
        "segment": [[float(x1), float(y1)], [float(x2), float(y2)]],
        "center": [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)],
        "width_px": width,
        "orientation": "horizontal" if abs(x2 - x1) >= abs(y2 - y1) else "vertical",
        "facing": door_facing(segment, boundary) if len(boundary) >= 3 else "unknown",
        "entrance_room_id": int(entrance_idx) if entrance_ok else None,
        "entrance_distance_px": float(dist) if np.isfinite(dist) else None,
        "source": source,
    }


def detect_yellow_door_pixels(image: np.ndarray) -> Optional[np.ndarray]:
    """Find strongly yellow pixels in an RGB snapshot (diagnostic only).

    Returns an ``(N, 2)`` array of ``(x, y)`` pixel coordinates or ``None``.
    Used by phase 01/02 to check that the door segment matches the yellow line
    drawn in the Graph2Plan snapshots.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    r, g, b = (image[..., c].astype(np.int32) for c in range(3))
    mask = (r > 190) & (g > 160) & (b < 90) & (np.abs(r - g) < 90)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.stack([xs, ys], axis=1)
