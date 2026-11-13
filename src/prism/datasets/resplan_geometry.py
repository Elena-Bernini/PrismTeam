"""ResPlan-specific geometry normalisation: wall-separated rooms -> RPLAN-style room partition.

Why
---
RPLAN layouts are a **partition**: the room polygons tile the building outline
exactly (no gaps, no overlaps) and neighbouring rooms meet on the centre line of
the wall between them, so the renderer draws one wall line per wall.

ResPlan stores the rooms as the free space *between* walls (interior walls are
~0.7 x ``wall_depth`` thick, exterior walls ~1 x), balconies lie entirely outside
the ``inner`` outline and room outlines have small notches (door recesses).
Drawn as they are, this gives double walls, detached rooms (balconies floating
next to the plan) and outlines that do not match.

What
----
:func:`partition_rooms` turns the wall-separated rooms into an RPLAN-style
partition, in the original ResPlan coordinates:

0. **overlaps** - ResPlan rooms can overlap (a storage or stair polygon inside the
   living room); the overlap is given to the smaller room;
1. **simultaneous growth** - every room grows into the surrounding walls in
   ``steps`` small increments up to ``growth x wall_depth``; at each step a room
   only takes territory nobody owns yet, so neighbours meet half-way across the
   wall (a discrete Voronoi split with error <= one step);
2. **footprint** - the union of the grown rooms, closed morphologically
   (``closing x wall_depth``) to remove notches and slivers; unlabelled pockets
   inside the apartment (up to ``fill_holes`` x its area) are filled;
3. **leftovers** - every piece of the footprint still unowned is given to the
   room that shares the longest boundary with it (else the nearest one);
4. **regularisation** - notches and protrusions narrower than ~``regularize x
   wall_depth`` (door recesses, wall stubs) are removed with a per-room
   opening / closing that keeps the partition;
5. **axis snapping** - x (and y) coordinates closer than ``snap x wall_depth``
   are merged, so small steps and slightly offset walls line up (L-shaped corner
   cut-outs survive opening / closing, so they need this step);
6. **clean-up** - tiny parts go to their neighbour; the cells are rebuilt from
   the noded boundaries (identical vertices on shared edges) and simplified all
   together with ``shapely.coverage_simplify``, so neighbours keep one common
   edge (no double lines, no zero-width spikes).

The result is used for the layout polygons and the boundary (so areas, grid
cells, renders and adjacency follow the same conventions as RPLAN). The original
room polygons are kept separately (``source_polygons``) because door / arch /
window classification needs the real wall strips.

Only shapely is needed (already required to unpickle ``ResPlan.pkl``); it is
imported lazily.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


def _mitre(geom: Any, dist: float) -> Any:
    """Buffer with mitred joins, so rectilinear rooms stay rectilinear."""
    import numpy as np

    with np.errstate(all="ignore"):          # degenerate slivers make GEOS divide by zero (harmless)
        return geom.buffer(dist, join_style=2, mitre_limit=2.0)


def _fill_small_holes(geom: Any, max_area: float) -> Any:
    """Fill the interior holes of ``geom`` whose area is at most ``max_area``."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    out = []
    for part in _parts(geom):
        keep = [h for h in part.interiors if Polygon(h).area > max_area]
        out.append(Polygon(part.exterior, keep))
    return _clean(unary_union(out)) if out else geom


def _clean(geom: Any) -> Any:
    """Valid, purely polygonal geometry (lines / points left by set operations are dropped)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if geom is None or geom.is_empty:
        return Polygon()
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = _parts(geom)
    return unary_union(parts) if parts else Polygon()


def _parts(geom: Any) -> List[Any]:
    """Polygons of a (Multi)Polygon / GeometryCollection (empty parts removed)."""
    if geom is None or geom.is_empty:
        return []
    if hasattr(geom, "geoms"):
        out: List[Any] = []
        for g in geom.geoms:
            out.extend(_parts(g))
        return out
    return [geom] if geom.geom_type == "Polygon" and geom.area > 0 else []


def _largest_neighbour(cells: List[Any], piece: Any, exclude: int = -1) -> int:
    """Index of the cell sharing the longest boundary with ``piece`` (else the nearest one)."""
    idx = [i for i in range(len(cells)) if i != exclude and not cells[i].is_empty]
    if not idx:
        return exclude
    shared = {i: cells[i].boundary.intersection(piece.boundary).length for i in idx}
    best = max(idx, key=lambda i: shared[i])
    if shared[best] <= 0:
        best = min(idx, key=lambda i: cells[i].distance(piece))
    return best


def _assign_leftovers(cells: List[Any], footprint: Any) -> List[Any]:
    """Give every unowned piece of ``footprint`` to the cell it shares the longest boundary with."""
    from shapely.ops import unary_union

    for piece in _parts(footprint.difference(unary_union(cells))):
        best = _largest_neighbour(cells, piece)
        cells[best] = _clean(cells[best].union(piece))
    return cells


def _regularize(cells: List[Any], footprint: Any, radius: float) -> List[Any]:
    """Remove notches and protrusions narrower than ``2 * radius`` while keeping a partition.

    Each cell is opened (thin protrusions removed); then, from the smallest room to
    the largest, each cell is closed and takes the notches it fills if nobody owns
    them yet. Remaining pieces go to the neighbour with the longest shared boundary.
    Opening / closing use mitred buffers, so rectilinear rooms stay rectilinear.
    """
    from shapely.ops import unary_union

    if radius <= 0:
        return cells
    opened = [_clean(_clean(_mitre(_mitre(c, -radius), radius)).intersection(c)) for c in cells]
    opened = [o if not o.is_empty else c for o, c in zip(opened, cells)]   # never delete a room
    owned = unary_union(opened)
    out = list(opened)
    for i in sorted(range(len(cells)), key=lambda i: cells[i].area):
        closed = _clean(_clean(_mitre(_mitre(cells[i], radius), -radius)).intersection(footprint))
        gain = _clean(closed.difference(owned))
        if not gain.is_empty:
            out[i] = _clean(out[i].union(gain))
            owned = owned.union(gain)
    return _assign_leftovers(out, footprint)


def _edge_matched(cells: List[Any]) -> List[Any]:
    """Rebuild the cells from the noded union of their boundaries (identical vertices on shared edges)."""
    from shapely.ops import polygonize, unary_union

    faces = list(polygonize(unary_union([c.boundary for c in cells if not c.is_empty])))
    groups: List[List[Any]] = [[] for _ in cells]
    for face in faces:
        pt = face.representative_point()
        owner = next((i for i, c in enumerate(cells) if c.contains(pt)), None)
        if owner is None:
            owner = max(range(len(cells)), key=lambda i: cells[i].intersection(face).area)
            if cells[owner].intersection(face).area <= 0:
                continue                                  # a hole of the footprint
        groups[owner].append(face)
    return [_clean(unary_union(g)) if g else cells[i] for i, g in enumerate(groups)]


def _axis_clusters(values: Any, tol: float) -> Any:
    """Monotone map of coordinates: values closer than ``tol`` are merged into their weighted median.

    Clusters are built on the sorted values (a new cluster starts after a gap
    larger than ``tol`` or when the cluster would span more than ``1.5 tol``), so the
    map never changes the order of two coordinates.
    """
    import numpy as np

    vals = np.sort(np.asarray(values, dtype=np.float64))
    if vals.size == 0:
        return {}
    mapping = {}
    start = 0
    for k in range(1, vals.size + 1):
        if k == vals.size or vals[k] - vals[k - 1] > tol or vals[k] - vals[start] > 1.5 * tol:
            rep = float(np.median(vals[start:k]))
            for v in vals[start:k]:
                mapping[float(v)] = rep
            start = k
    return mapping


def _snap_axes(cells: List[Any], footprint: Any, tol: float) -> List[Any]:
    """Merge nearly equal x (and y) coordinates of the whole plan: steps smaller than ``tol`` disappear.

    The same monotone map is applied to every room, so shared edges stay shared;
    slivers created by the snap are cleaned and any overlap or gap is resolved
    again (overlaps go to the earlier room, gaps to the longest shared boundary).
    """
    import numpy as np
    import shapely
    from shapely.ops import unary_union

    if tol <= 0:
        return cells
    coords = np.concatenate([shapely.get_coordinates(c) for c in cells if not c.is_empty])
    mx, my = _axis_clusters(coords[:, 0], tol), _axis_clusters(coords[:, 1], tol)

    def snap(xy: Any) -> Any:
        """Apply the axis maps to an (N, 2) coordinate array."""
        out = np.array(xy, dtype=np.float64)
        out[:, 0] = [mx.get(float(v), float(v)) for v in out[:, 0]]
        out[:, 1] = [my.get(float(v), float(v)) for v in out[:, 1]]
        return out

    snapped = [_clean(shapely.transform(c, snap)) for c in cells]
    # Overlaps created by the snap go to the SMALLER room, so a closet cannot be eaten by its neighbour.
    resolved: List[Any] = list(snapped)
    owned = None
    for i in sorted(range(len(snapped)), key=lambda i: snapped[i].area):
        piece = snapped[i] if owned is None else _clean(snapped[i].difference(owned))
        resolved[i] = piece if not piece.is_empty else snapped[i]
        owned = resolved[i] if owned is None else owned.union(resolved[i])
    new_footprint = _clean(shapely.transform(footprint, snap))
    return _assign_leftovers(resolved, unary_union([new_footprint, unary_union(resolved)]))


def partition_rooms(rooms: Sequence[Any], wall_depth: float, growth: float = 0.55, steps: int = 6,
                    closing: float = 0.75, regularize: float = 0.4, snap: float = 1.5, simplify: float = 0.25,
                    fill_holes: float = 0.05, min_part_ratio: float = 0.02) -> Tuple[List[Any], Any, Dict[str, float]]:
    """Grow wall-separated rooms into a gap-free, overlap-free partition of the building.

    ``rooms`` are shapely (Multi)Polygons in ResPlan coordinates; every distance is
    a multiple of ``wall_depth``. Returns ``(cells, footprint, stats)`` where
    ``cells[i]`` is the (Multi)Polygon of room ``i`` and ``footprint`` is their
    union (the building outline, balconies included).
    """
    import shapely
    from shapely.ops import unary_union

    wd = max(float(wall_depth), 1e-6)
    # 0) ResPlan rooms may overlap: storage / stair polygons often lie INSIDE the living
    #    room or the kitchen (~6% of plans). The overlap goes to the smaller room (the
    #    closet stays a closet, the living room gets a hole around it).
    src = [_clean(r) for r in rooms]
    taken = None
    for i in sorted(range(len(src)), key=lambda i: src[i].area):
        if taken is not None:
            rest = _clean(src[i].difference(taken))
            if not rest.is_empty:
                src[i] = rest
        taken = src[i] if taken is None else taken.union(src[i])
    cells = list(src)

    # 1) simultaneous growth: neighbours meet half-way across each wall.
    for k in range(1, steps + 1):
        radius = growth * wd * k / steps
        claimed = unary_union(cells)
        for i, room in enumerate(src):
            new = _clean(_clean(_mitre(room, radius)).difference(claimed))
            if not new.is_empty:
                cells[i] = _clean(cells[i].union(new))
                claimed = claimed.union(new)

    # 2) footprint = closed union of the grown rooms (fills notches and slivers).
    union = unary_union(cells)
    c = closing * wd
    footprint = _clean(_mitre(_mitre(union, c), -c)).union(union)
    # Unlabelled pockets inside the apartment (shafts, closets without a category) up to
    # ``fill_holes`` x footprint area are filled: in RPLAN the rooms always tile the
    # outline, and a grey pocket would look like a masked room of a partial query.
    footprint = _fill_small_holes(footprint, fill_holes * footprint.area)

    # 3) leftovers go to the room sharing the longest boundary.
    cells = _assign_leftovers(cells, footprint)

    # 4) regularisation: notches / protrusions smaller than ~a wall disappear.
    cells = _regularize(cells, footprint, regularize * wd)

    # 5) tiny parts are merged into the neighbour they touch most.
    for i in range(len(cells)):
        parts = _parts(cells[i])
        if len(parts) <= 1:
            continue
        biggest = max(p.area for p in parts)
        keep = [p for p in parts if p.area >= min_part_ratio * biggest]
        for p in parts:
            if p.area < min_part_ratio * biggest:
                j = _largest_neighbour(cells, p, exclude=i)
                if j != i:
                    cells[j] = _clean(cells[j].union(p))
        cells[i] = _clean(unary_union(keep))

    # 6) axis snapping: steps smaller than ``snap x wall_depth`` (corner cut-outs, wall offsets)
    #    disappear; opening / closing cannot remove them because they are L-shaped.
    cells = _snap_axes(cells, unary_union(cells), snap * wd)

    # 7) edge-matched coverage, then topology-preserving simplification of all edges at once.
    cells = _edge_matched(cells)
    if simplify > 0 and hasattr(shapely, "coverage_simplify"):
        simplified = list(shapely.coverage_simplify(cells, simplify * wd))
        if all(s.is_valid and not s.is_empty for s in simplified):
            cells = simplified
    cells = [_clean(cell) for cell in cells]

    footprint = unary_union(cells)
    total = sum(cell.area for cell in cells)
    stats = {
        "overlap_ratio": max(0.0, total / max(footprint.area, 1e-9) - 1.0),
        "footprint_parts": float(len(_parts(footprint))),
        "growth_area_ratio": total / max(sum(r.area for r in src), 1e-9),
        "vertices": float(sum(len(ring.coords) for cell in cells for p in _parts(cell) for ring in [p.exterior])),
    }
    return cells, footprint, stats


def polygon_rings(geom: Any) -> List[List[List[float]]]:
    """Exterior rings (closing point dropped) of every part, largest first."""
    parts = sorted(_parts(geom), key=lambda p: p.area, reverse=True)
    return [[[float(x), float(y)] for x, y, *_ in list(p.exterior.coords)[:-1]] for p in parts]


def outline_ring(footprint: Any) -> List[List[float]]:
    """Exterior ring of the building footprint (largest part)."""
    rings = polygon_rings(footprint)
    return rings[0] if rings else []


def outline_holes(footprint: Any) -> List[List[List[float]]]:
    """Interior rings (light wells, courtyards) of the largest part of the footprint."""
    parts = sorted(_parts(footprint), key=lambda p: p.area, reverse=True)
    if not parts:
        return []
    return [[[float(x), float(y)] for x, y, *_ in list(h.coords)[:-1]] for h in parts[0].interiors]
