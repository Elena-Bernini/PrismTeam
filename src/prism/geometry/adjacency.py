"""Geometric room adjacency and connection types.

**Adjacency.** Two rooms are adjacent when they share a wall. Room polygons are
separated by walls a few pixels thick, so we:

1. rasterise each room (all its polygons) into a 256x256 mask;
2. dilate every mask by ``radius`` pixels (square element), so the two dilated
   masks overlap across a shared wall. ``radius`` is ``geometry.wall_dilation_px``
   for RPLAN and is derived from the plan's own wall thickness for ResPlan;
3. measure the shared-wall length as the longest side of the bounding box of the
   overlap region, minus the ``2 * radius`` that the dilation adds at both ends;
4. call the pair adjacent if the shared-wall length is at least
   ``min_contact_px``. Rooms that only touch at a corner have a shared length of
   ~0 and are therefore NOT adjacent.

**Connection type** (only when the layout has wall / door / window geometry,
i.e. ResPlan). The "gap" between two adjacent rooms is the overlap of their
dilated masks outside both rooms - the strip where the shared wall should be:

- ``door``    a door polygon covers at least ``door_min_px`` of the gap length
              (door pixels in the gap / wall thickness);
- ``arch``    a stretch of the gap of at least ``open_passage_min_px`` has no wall
              (open passage), or the rooms touch directly where no wall is drawn;
- ``window``  a window polygon lies in the gap (e.g. room -> balcony);
- ``wall``    otherwise.

RPLAN layouts have no such geometry, so their edges are all ``wall``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from prism.geometry.polygon import dilate_square, rasterize_shapes


def bbox_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """Chebyshev gap between two boxes (0 if they overlap or touch)."""
    gx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    gy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(gx, gy)


def contact_length(mask_a: np.ndarray, mask_b: np.ndarray, radius: int) -> Tuple[float, int]:
    """Return ``(shared_wall_px, overlap_pixels)`` for two dilated masks.

    ``shared_wall_px`` is the longest side of the overlap bounding box minus the
    ``2 * radius`` extension introduced by dilation (clipped at 0).
    """
    overlap = mask_a & mask_b
    count = int(overlap.sum())
    if count == 0:
        return 0.0, 0
    ys, xs = np.nonzero(overlap)
    extent = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return float(max(0, extent - 2 * radius)), count


class RoomMasks:
    """Raw and dilated masks of every room of a layout (computed once, reused)."""

    def __init__(self, room_shapes: Sequence[Sequence[Any]], canvas: int, radius: int) -> None:
        """Rasterise each room (list of polygons) and dilate it by ``radius``."""
        self.canvas, self.radius = canvas, radius
        self.raw = [rasterize_shapes(shapes, canvas) for shapes in room_shapes]
        self.dilated = [dilate_square(m, radius) for m in self.raw]


def geometric_adjacency(masks: RoomMasks, bboxes: Sequence[Sequence[float]], min_contact_px: int = 5) -> Dict[Tuple[int, int], float]:
    """Adjacent room pairs ``{(i, j): shared_wall_px}`` with ``i < j``.

    Pairs whose bounding boxes are further apart than twice the dilation radius
    are skipped without comparing masks.
    """
    n = len(masks.raw)
    r = masks.radius
    pairs: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if bbox_gap(bboxes[i], bboxes[j]) > 2 * r + 1:
                continue
            length, _ = contact_length(masks.dilated[i], masks.dilated[j], r)
            if length >= min_contact_px:
                pairs[(i, j)] = length
    return pairs


def classify_connections(
    pairs: Sequence[Tuple[int, int]],
    masks: RoomMasks,
    walls: Optional[List[Any]],
    doors: Optional[List[Any]],
    windows: Optional[List[Any]],
    wall_thickness_px: float,
    open_passage_min_px: float = 6.0,
    door_min_px: float = 4.0,
) -> Dict[Tuple[int, int], str]:
    """Connection type of each adjacent pair (see module docstring).

    Returns ``"wall"`` for every pair when the layout has no wall geometry.
    """
    if not walls:
        return {p: "wall" for p in pairs}
    size = masks.canvas
    wall_mask = dilate_square(rasterize_shapes(walls, size), 1)
    door_mask = dilate_square(rasterize_shapes(doors or [], size), 1)
    window_mask = dilate_square(rasterize_shapes(windows or [], size), 1)
    thickness = max(1.0, float(wall_thickness_px))
    out: Dict[Tuple[int, int], str] = {}
    for i, j in pairs:
        raw_i, raw_j = masks.raw[i], masks.raw[j]
        gap = masks.dilated[i] & masks.dilated[j] & ~raw_i & ~raw_j
        # Direct contact only counts where no wall is drawn: with thin walls (~3 px after
        # rescaling) rasterised rooms on both sides of a real wall can touch, which made
        # ~19% of ResPlan's plain "adjacency" pairs look like open passages.
        touching = int((dilate_square(raw_i, 1) & raw_j & ~wall_mask).sum())
        # Door pixels / wall thickness = length of the shared wall covered by a door. A
        # fixed pixel count (3) labelled ~60% of ResPlan's plain "adjacency" pairs as doors,
        # because door polygons of nearby rooms reach into the corners of the gap.
        if int((gap & door_mask).sum()) / thickness >= door_min_px:
            out[(i, j)] = "door"
            continue
        open_len = int((gap & ~wall_mask & ~door_mask & ~window_mask).sum()) / thickness
        if open_len >= open_passage_min_px or touching >= open_passage_min_px:
            out[(i, j)] = "arch"
        elif int((gap & window_mask).sum()) >= 3:
            out[(i, j)] = "window"
        else:
            out[(i, j)] = "wall"
    return out
