"""Render a layout JSON into a semantic RGB image (Pillow only).

The same renderer is used for full layouts (phase 04) and for partial queries
(phase 10), so the vision encoders always see the same visual language:

- white background outside the building;
- building interior filled with ``render.unknown_fill`` (grey) - this is what a
  masked room looks like in a partial query;
- every room filled with the colour of its type (``configs/room_types.yaml``);
- thin dark room outlines (walls), a thicker exterior outline;
- the FRONT DOOR drawn as a thick yellow segment on the exterior wall, as in
  the original RPLAN images.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from PIL import Image, ImageDraw

from src.prism.datasets.schema import Taxonomy
from src.prism.geometry.polygon import bbox_to_polygon

FRONT_DOOR_COLOR = (255, 214, 0)
WALL_COLOR = (40, 40, 40)
BACKGROUND = (255, 255, 255)


def hex_to_rgb(color: str) -> tuple:
    """Convert ``#rrggbb`` into an ``(r, g, b)`` tuple."""
    c = color.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def render_layout(layout: Dict[str, Any], taxonomy: Taxonomy, rcfg: Dict[str, Any],
                  visible_room_ids: Optional[Sequence[int]] = None, size: Optional[int] = None) -> Image.Image:
    """Return the semantic render of ``layout``.

    ``visible_room_ids`` restricts the coloured rooms (partial queries); the
    other rooms stay grey inside the building outline.
    """
    canvas = float(layout["coordinate_system"]["width"])
    out_size = int(size or rcfg["size"])
    scale = out_size / canvas
    img = Image.new("RGB", (out_size, out_size), BACKGROUND)
    draw = ImageDraw.Draw(img)

    def sc(points):
        """Scale polygon points from layout to image coordinates."""
        return [(float(x) * scale, float(y) * scale) for x, y in points]

    boundary = layout.get("boundary") or []
    unknown = hex_to_rgb(rcfg.get("unknown_fill", "#bdbdbd"))
    if len(boundary) >= 3:
        draw.polygon(sc(boundary), fill=unknown)

    visible = None if visible_room_ids is None else {int(i) for i in visible_room_ids}
    outline_w = max(1, int(round(rcfg.get("outline_width", 2) * scale)))
    for room in layout["rooms"]:
        if visible is not None and int(room["id"]) not in visible:
            continue
        poly = room.get("polygon") or bbox_to_polygon(room["bbox"])
        if len(poly) < 3:
            continue
        fill = hex_to_rgb(taxonomy.color(int(room["room_type_id"])))
        draw.polygon(sc(poly), fill=fill)
        if rcfg.get("draw_room_outlines", True):
            pts = sc(poly)
            draw.line(pts + [pts[0]], fill=WALL_COLOR, width=outline_w)

    if len(boundary) >= 3:
        pts = sc(boundary)
        draw.line(pts + [pts[0]], fill=WALL_COLOR, width=max(1, int(round(rcfg.get("boundary_width", 3) * scale))))

    door = layout.get("front_door")
    if door and rcfg.get("draw_front_door", True):
        draw.line(sc(door["segment"]), fill=FRONT_DOOR_COLOR,
                  width=max(2, int(round(rcfg.get("front_door_width", 4) * scale))))
    return img
