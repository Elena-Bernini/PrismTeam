"""Geometric room adjacency.

Two rooms are adjacent when they share a wall. RPLAN room polygons are separated
by walls a few pixels thick, so we:

1. rasterise each room polygon into a 256x256 mask;
2. dilate every mask by ``wall_dilation_px`` pixels (square element), so the two
   dilated masks overlap across a shared wall;
3. measure the shared-wall length as the longest side of the bounding box of the
   overlap region, minus the ``2 * wall_dilation_px`` that the dilation adds at
   both ends of the wall;
4. call the pair adjacent if the shared-wall length is at least
   ``min_contact_px``. Rooms that only touch at a corner have a shared length of
   ~0 and are therefore NOT adjacent.

This mirrors the Graph2Plan paper ("rooms on the two sides of a door, plus rooms
closer than a threshold") but is deterministic, documented and tunable.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.prism.geometry.polygon import dilate_square, rasterize_polygon


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


def geometric_adjacency(
    polygons: Sequence[Sequence[Sequence[float]]],
    bboxes: Sequence[Sequence[float]],
    canvas: int = 256,
    wall_dilation_px: int = 3,
    min_contact_px: int = 5,
) -> Dict[Tuple[int, int], float]:
    """Compute adjacent room pairs from polygons.

    Returns a dictionary ``{(i, j): shared_wall_px}`` with ``i < j`` for every
    adjacent pair. Pairs whose bounding boxes are further apart than twice the
    dilation radius are skipped without rasterising.
    """
    n = len(polygons)
    dilated: List[np.ndarray] = []
    for poly in polygons:
        dilated.append(dilate_square(rasterize_polygon(poly, canvas), wall_dilation_px))

    pairs: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if bbox_gap(bboxes[i], bboxes[j]) > 2 * wall_dilation_px + 1:
                continue
            length, _ = contact_length(dilated[i], dilated[j], wall_dilation_px)
            if length >= min_contact_px:
                pairs[(i, j)] = length
    return pairs
