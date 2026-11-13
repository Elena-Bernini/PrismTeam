"""Tiny synthetic dataset in the Graph2Plan pickle format (for smoke tests only).

It lets the whole pipeline run without the real RPLAN files (CI, new laptops,
debugging). Buildings are rectangles split by random guillotine cuts into
rectangular rooms separated by 4 px walls; the front door is put on the
exterior wall of the living room and also drawn in yellow on a fake snapshot.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

WALL = 4  # wall thickness in pixels


class G2PSample:
    """Minimal stand-in for a Graph2Plan sample (attributes only)."""

    def __init__(self, **kwargs: Any) -> None:
        """Store every keyword argument as an attribute."""
        self.__dict__.update(kwargs)


def _split(cell: Tuple[int, int, int, int], n: int, rng: random.Random) -> List[Tuple[int, int, int, int]]:
    """Recursively split a rectangle into ``n`` cells with guillotine cuts."""
    if n <= 1:
        return [cell]
    x0, y0, x1, y1 = cell
    w, h = x1 - x0, y1 - y0
    n_left = rng.randint(1, n - 1)
    frac = min(0.75, max(0.25, n_left / n + rng.uniform(-0.1, 0.1)))
    if w >= h:
        cut = int(x0 + w * frac)
        return _split((x0, y0, cut, y1), n_left, rng) + _split((cut, y0, x1, y1), n - n_left, rng)
    cut = int(y0 + h * frac)
    return _split((x0, y0, x1, cut), n_left, rng) + _split((x0, cut, x1, y1), n - n_left, rng)


def make_sample(idx: int, rng: random.Random) -> G2PSample:
    """Create one synthetic floor plan."""
    bx0, by0 = rng.randint(16, 60), rng.randint(16, 60)
    bx1, by1 = rng.randint(196, 240), rng.randint(196, 240)
    n_rooms = rng.randint(4, 8)
    cells = _split((bx0, by0, bx1, by1), n_rooms, rng)
    # Shrink each cell by half a wall on every side -> rooms separated by WALL px.
    rooms = [(x0 + WALL // 2, y0 + WALL // 2, x1 - WALL // 2, y1 - WALL // 2) for x0, y0, x1, y1 in cells]
    areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in rooms]
    living = int(np.argmax(areas))
    types = []
    pool = [1, 7, 8, 5, 3, 2, 9, 11, 6, 4]
    for i in range(len(rooms)):
        types.append(0 if i == living else pool[(i + idx) % len(pool)])

    # Front door on the exterior wall closest to the living room (bottom or top).
    lx0, ly0, lx1, ly1 = rooms[living]
    door_w = min(14, max(6, (lx1 - lx0) // 3))
    dx0 = (lx0 + lx1) // 2 - door_w // 2
    if (by1 - ly1) <= (ly0 - by0):
        door = [(dx0, by1), (dx0 + door_w, by1)]
        outline = [door[0], door[1], (bx1, by1), (bx1, by0), (bx0, by0), (bx0, by1)]
    else:
        door = [(dx0 + door_w, by0), (dx0, by0)]
        outline = [door[0], door[1], (bx0, by0), (bx0, by1), (bx1, by1), (bx1, by0)]
    boundary = np.array([[x, y, 0, 1 if k < 2 else 0] for k, (x, y) in enumerate(outline)], dtype=np.int32)

    # Graph2Plan-like edge list: pairs of cells sharing a wall, with an opaque third column.
    edges = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            a, b = cells[i], cells[j]
            share_x = min(a[2], b[2]) - max(a[0], b[0])
            share_y = min(a[3], b[3]) - max(a[1], b[1])
            if (share_x > 6 and (a[3] == b[1] or b[3] == a[1])) or (share_y > 6 and (a[2] == b[0] or b[2] == a[0])):
                edges.append([i, j, rng.randint(0, 9)])
    box = np.array([[r[0], r[1], r[2], r[3], t] for r, t in zip(rooms, types)], dtype=np.int32)
    polys = [np.array([[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], dtype=np.int32) for r in rooms]
    return G2PSample(name=str(1000 + idx), boundary=boundary, box=box,
                     edge=np.array(edges, dtype=np.int32).reshape(-1, 3), rBoundary=polys,
                     order=np.arange(len(rooms)))


def draw_snapshot(sample: G2PSample, path: Path) -> None:
    """Fake RPLAN snapshot: coloured rooms, black walls, yellow front door."""
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    d = ImageDraw.Draw(img)
    outline = [tuple(p[:2]) for p in sample.boundary.tolist()]
    d.polygon(outline, fill=(60, 60, 60))
    palette = {0: (220, 220, 210), 2: (240, 160, 160), 3: (150, 200, 240), 9: (160, 210, 130)}
    for b in sample.box.tolist():
        d.rectangle(b[:4], fill=palette.get(b[4], (235, 215, 160)))
    d.line([tuple(sample.boundary[0][:2]), tuple(sample.boundary[1][:2])], fill=(255, 220, 0), width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def write_synthetic_rplan(pkl_path: Path, snapshot_dir: Path, n: int = 120, seed: int = 0) -> Dict[str, Any]:
    """Write ``n`` synthetic samples to ``pkl_path`` and their snapshots to ``snapshot_dir``."""
    rng = random.Random(seed)
    samples = [make_sample(i, rng) for i in range(n)]
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump({"data": samples, "note": "synthetic Prism smoke-test data"}, f)
    for s in samples:
        draw_snapshot(s, snapshot_dir / f"{s.name}.png")
    return {"pkl": str(pkl_path), "snapshots": str(snapshot_dir), "n": n}


# =============================================================================
# Synthetic ResPlan (same structure as ResPlan.pkl, shapely-like objects)
# =============================================================================
class FakeRing:
    """Minimal stand-in for a shapely LinearRing (``coords`` incl. the closing point)."""

    def __init__(self, pts: List[Tuple[float, float]]) -> None:
        """Store the ring, closing it."""
        self.coords = list(pts) + [pts[0]]


class FakePolygon:
    """Minimal stand-in for a shapely Polygon (exterior, interiors, area, is_empty)."""

    def __init__(self, exterior: List[Tuple[float, float]], holes: List[List[Tuple[float, float]]] = ()) -> None:
        """Store exterior and holes."""
        self.exterior = FakeRing(exterior)
        self.interiors = [FakeRing(h) for h in holes]
        self.is_empty = False

    @property
    def area(self) -> float:
        """Shoelace area of the exterior minus holes."""
        def a(pts):
            """Area of one ring."""
            p = np.asarray(pts[:-1], dtype=float)
            return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
        return a(self.exterior.coords) - sum(a(r.coords) for r in self.interiors)


class FakeMultiPolygon:
    """Minimal stand-in for a shapely MultiPolygon."""

    def __init__(self, polys: List[FakePolygon]) -> None:
        """Store the parts."""
        self.geoms = list(polys)
        self.is_empty = not polys


def _rect(x0: float, y0: float, x1: float, y1: float) -> List[Tuple[float, float]]:
    """Rectangle as a point list."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def make_resplan_plan(idx: int, rng: random.Random) -> Dict[str, Any]:
    """One synthetic ResPlan-like plan in metres (y up), with walls, doors, windows and front door."""
    wd = 0.2
    W, H = rng.uniform(9, 16), rng.uniform(8, 13)
    n = rng.randint(4, 8)
    cells = _split((0, 0, int(W * 100), int(H * 100)), n, rng)
    cells = [(a / 100, b / 100, c / 100, d / 100) for a, b, c, d in cells]
    rooms = [(x0 + wd / 2, y0 + wd / 2, x1 - wd / 2, y1 - wd / 2) for x0, y0, x1, y1 in cells]
    areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in rooms]
    living = int(np.argmax(areas))
    cats = ["bedroom", "bathroom", "kitchen", "bedroom", "storage", "balcony", "bedroom", "bathroom"]
    types = ["living" if i == living else cats[(i + idx) % len(cats)] for i in range(len(rooms))]

    def shared(a, b):
        """Shared wall segment between two cells (or None)."""
        if abs(a[2] - b[0]) < 1e-6 or abs(b[2] - a[0]) < 1e-6:
            x = a[2] if abs(a[2] - b[0]) < 1e-6 else a[0]
            lo, hi = max(a[1], b[1]), min(a[3], b[3])
            return ("v", x, lo, hi) if hi - lo > 0.8 else None
        if abs(a[3] - b[1]) < 1e-6 or abs(b[3] - a[1]) < 1e-6:
            y = a[3] if abs(a[3] - b[1]) < 1e-6 else a[1]
            lo, hi = max(a[0], b[0]), min(a[2], b[2])
            return ("h", y, lo, hi) if hi - lo > 0.8 else None
        return None

    doors, windows = [], []
    for j in range(len(cells)):
        if j == living:
            continue
        s = shared(cells[living], cells[j])
        if s is None:
            continue
        kind, c, lo, hi = s
        mid = (lo + hi) / 2
        if types[j] == "kitchen":
            continue                                   # open passage, handled below (no wall piece)
        poly = (_rect(c - wd / 2, mid - 0.45, c + wd / 2, mid + 0.45) if kind == "v"
                else _rect(mid - 0.45, c - wd / 2, mid + 0.45, c + wd / 2))
        (windows if types[j] == "balcony" else doors).append(FakePolygon(poly))

    # Kitchen next to the living room: remove the wall between them (rooms grow into the gap).
    room_polys = [list(r) for r in rooms]
    for j in range(len(cells)):
        if types[j] == "kitchen" and shared(cells[living], cells[j]) is not None:
            kind, c, lo, hi = shared(cells[living], cells[j])
            for k in (living, j):
                r = room_polys[k]
                if kind == "v":
                    if abs(r[2] - (c - wd / 2)) < 1e-6:
                        r[2] = c
                    if abs(r[0] - (c + wd / 2)) < 1e-6:
                        r[0] = c
                else:
                    if abs(r[3] - (c - wd / 2)) < 1e-6:
                        r[3] = c
                    if abs(r[1] - (c + wd / 2)) < 1e-6:
                        r[1] = c

    outer = _rect(-wd / 2, -wd / 2, W + wd / 2, H + wd / 2)
    wall = FakePolygon(outer, [_rect(*r) for r in room_polys])
    by_cat: Dict[str, List[FakePolygon]] = {}
    for t, r in zip(types, room_polys):
        by_cat.setdefault(t, []).append(FakePolygon(_rect(*r)))
    lx0, ly0, lx1, ly1 = room_polys[living]
    fx = (lx0 + lx1) / 2
    front = FakePolygon(_rect(fx - 0.5, -wd / 2, fx + 0.5, wd / 2)) if ly0 < 0.5 else \
        FakePolygon(_rect(fx - 0.5, H - wd / 2, fx + 0.5, H + wd / 2))
    plan: Dict[str, Any] = {k: (v[0] if len(v) == 1 else FakeMultiPolygon(v)) for k, v in by_cat.items()}
    if "balcony" in plan:
        plan["balacony"] = plan.pop("balcony")       # ResPlan's own misspelling
    plan.update({"id": 5000 + idx, "unitType": "apartment", "area": round(W * H, 2), "net_area": round(sum(areas), 2),
                 "wall_depth": wd, "wall": wall, "door": FakeMultiPolygon(doors), "window": FakeMultiPolygon(windows),
                 "front_door": front, "inner": FakePolygon(_rect(0, 0, W, H))})
    return plan


def write_synthetic_resplan(root: Path, n: int = 120, seed: int = 1) -> Dict[str, Any]:
    """Write ``ResPlan.pkl`` and ``split.json`` with ``n`` synthetic plans under ``root``."""
    import json

    rng = random.Random(seed)
    plans = [make_resplan_plan(i, rng) for i in range(n)]
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "ResPlan.pkl", "wb") as f:
        pickle.dump(plans, f)
    ids = [str(p["id"]) for p in plans]
    n_aug = max(1, n // 25)
    split = {"train": ids[: int(0.8 * n)], "val": ids[int(0.8 * n): int(0.9 * n)],
             "test": ids[int(0.9 * n): n - n_aug], "augmented": ids[n - n_aug:]}
    (root / "split.json").write_text(json.dumps(split), encoding="utf-8")
    return {"pkl": str(root / "ResPlan.pkl"), "n": n}
