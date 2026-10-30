"""Polygon primitives implemented with NumPy and Pillow only (no shapely needed).

Coordinates are image coordinates: origin top-left, x to the right, y downward.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

Point = Tuple[float, float]


def as_points(raw: Any) -> List[List[float]]:
    """Convert a raw polygon (list/array of ``[x, y, ...]``) into ``[[x, y], ...]``.

    Extra columns are dropped, malformed points are skipped and a closing point
    equal to the first one is removed.
    """
    if raw is None:
        return []
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return []
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return []
    pts = arr[:, :2]
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return [[float(x), float(y)] for x, y in pts]


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Absolute polygon area with the shoelace formula."""
    if len(points) < 3:
        return 0.0
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(points: Sequence[Sequence[float]]) -> Optional[Point]:
    """Area-weighted centroid of a simple polygon (falls back to the vertex mean)."""
    if len(points) < 3:
        return None
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return float(x.mean()), float(y.mean())
    cx = ((x + x1) * cross).sum() / (6.0 * a)
    cy = ((y + y1) * cross).sum() / (6.0 * a)
    return float(cx), float(cy)


def polygon_bbox(points: Sequence[Sequence[float]]) -> Optional[List[float]]:
    """Axis-aligned bounding box ``[x0, y0, x1, y1]`` of a polygon."""
    if not points:
        return None
    p = np.asarray(points, dtype=np.float64)
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``[x0, y0, x1, y1]`` boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_to_polygon(bbox: Sequence[float]) -> List[List[float]]:
    """Rectangle polygon from a bounding box (used when a room has no polygon)."""
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def point_in_polygon(point: Sequence[float], points: Sequence[Sequence[float]]) -> bool:
    """Even-odd ray-casting test (points exactly on an edge may go either way)."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(points)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = points[i][0], points[i][1]
        xj, yj = points[j][0], points[j][1]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def point_segment_distance(p: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance from point ``p`` to segment ``ab``."""
    p_, a_, b_ = (np.asarray(v, dtype=np.float64)[:2] for v in (p, a, b))
    ab = b_ - a_
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p_ - a_))
    t = float(np.clip(np.dot(p_ - a_, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p_ - (a_ + t * ab)))


def point_polygon_distance(point: Sequence[float], points: Sequence[Sequence[float]]) -> float:
    """Distance from a point to a polygon (0 when the point is inside)."""
    if len(points) < 2:
        return float("inf")
    if len(points) >= 3 and point_in_polygon(point, points):
        return 0.0
    n = len(points)
    return min(point_segment_distance(point, points[i], points[(i + 1) % n]) for i in range(n))


def segment_polygon_distance(a: Sequence[float], b: Sequence[float], points: Sequence[Sequence[float]],
                             samples: int = 9) -> float:
    """Approximate distance between a segment and a polygon by sampling the segment."""
    best = float("inf")
    for t in np.linspace(0.0, 1.0, samples):
        q = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        best = min(best, point_polygon_distance(q, points))
    return best


def rasterize_polygon(points: Sequence[Sequence[float]], size: int) -> np.ndarray:
    """Rasterise a polygon into a ``size x size`` boolean mask with Pillow."""
    mask = Image.new("L", (size, size), 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in points], fill=1, outline=1)
    return np.asarray(mask, dtype=bool)


def dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a ``(2r+1) x (2r+1)`` square structuring element.

    Implemented as two separable max-filters with array shifts, so it needs only
    NumPy. A square (not round) element keeps rectilinear walls rectilinear.
    """
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    # Horizontal pass.
    tmp = out.copy()
    for s in range(1, radius + 1):
        tmp[:, s:] |= out[:, :-s]
        tmp[:, :-s] |= out[:, s:]
    out = tmp
    # Vertical pass.
    tmp = out.copy()
    for s in range(1, radius + 1):
        tmp[s:, :] |= out[:-s, :]
        tmp[:-s, :] |= out[s:, :]
    return tmp
