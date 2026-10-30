"""Deterministic spatial relations between two rooms.

``spatial_relation(box_u, box_v)`` describes the SOURCE room ``u`` relative to
the TARGET room ``v`` (image coordinates: y grows downward). Rules, applied in
order (thresholds come from ``configs/default.yaml -> geometry``):

1. Containment. If at least ``containment_threshold`` (0.9) of u's bounding box
   lies inside v's bounding box, u is ``inside`` v. If at least 0.9 of v's box
   lies inside u's box, u is ``surrounding`` v. (This happens for rooms nested
   in the bend of an L-shaped living room.)
2. Axis overlap. Let ``ox`` be the overlap of the two x-intervals divided by the
   narrower width, ``oy`` the same for y-intervals.
   - ``ox >= axis_overlap_threshold`` (0.5): the rooms are stacked vertically and
     share a horizontal wall -> ``above`` / ``below`` from the centroid y.
   - ``oy >= 0.5``: side by side, sharing a vertical wall -> ``left_of`` /
     ``right_of`` from the centroid x.
   - both: the axis with the larger centroid offset (normalised by the box
     sizes) decides.
3. Otherwise the rooms are diagonal: ``left_above``, ``right_above``,
   ``left_below`` or ``right_below`` from the signs of the centroid offset.

Every edge is stored in both directions with the inverse relation, so nothing
is sampled at random (unlike Graph2Plan, which keeps one random direction).
"""

from __future__ import annotations

from typing import Dict, Sequence

from src.prism.datasets.schema import INVERSE_RELATION


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the overlap between intervals ``[a0, a1]`` and ``[b0, b1]``."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def relation_features(box_u: Sequence[float], box_v: Sequence[float], canvas: float = 256.0) -> Dict[str, float]:
    """Continuous geometric features of the pair (used as GNN edge attributes).

    Returns normalised centroid offsets ``dx, dy`` (u minus v, divided by the
    canvas size), projection overlaps ``overlap_x, overlap_y`` and containment
    ratios ``u_in_v, v_in_u``.
    """
    ux0, uy0, ux1, uy1 = [float(v) for v in box_u[:4]]
    vx0, vy0, vx1, vy1 = [float(v) for v in box_v[:4]]
    uw, uh = max(ux1 - ux0, 1e-6), max(uy1 - uy0, 1e-6)
    vw, vh = max(vx1 - vx0, 1e-6), max(vy1 - vy0, 1e-6)
    ix = interval_overlap(ux0, ux1, vx0, vx1)
    iy = interval_overlap(uy0, uy1, vy0, vy1)
    inter = ix * iy
    ucx, ucy = (ux0 + ux1) / 2.0, (uy0 + uy1) / 2.0
    vcx, vcy = (vx0 + vx1) / 2.0, (vy0 + vy1) / 2.0
    return {
        "dx": (ucx - vcx) / canvas,
        "dy": (ucy - vcy) / canvas,
        "overlap_x": ix / min(uw, vw),
        "overlap_y": iy / min(uh, vh),
        "u_in_v": inter / (uw * uh),
        "v_in_u": inter / (vw * vh),
        "norm_dx": (ucx - vcx) / ((uw + vw) / 2.0),
        "norm_dy": (ucy - vcy) / ((uh + vh) / 2.0),
    }


def spatial_relation(
    box_u: Sequence[float],
    box_v: Sequence[float],
    containment_threshold: float = 0.9,
    axis_overlap_threshold: float = 0.5,
) -> str:
    """Return the relation of room ``u`` with respect to room ``v`` (see module docstring)."""
    f = relation_features(box_u, box_v)

    # 1) Containment.
    if f["u_in_v"] >= containment_threshold and f["u_in_v"] >= f["v_in_u"]:
        return "inside"
    if f["v_in_u"] >= containment_threshold:
        return "surrounding"

    vertical_stack = f["overlap_x"] >= axis_overlap_threshold
    horizontal_stack = f["overlap_y"] >= axis_overlap_threshold
    dx, dy = f["norm_dx"], f["norm_dy"]

    # 2) Pure axis relations.
    if vertical_stack and horizontal_stack:
        if abs(dy) >= abs(dx):
            vertical_stack, horizontal_stack = True, False
        else:
            vertical_stack, horizontal_stack = False, True
    if vertical_stack:
        return "above" if dy < 0 else "below"
    if horizontal_stack:
        return "left_of" if dx < 0 else "right_of"

    # 3) Diagonal relations.
    horizontal = "left" if dx < 0 else "right"
    vertical = "above" if dy < 0 else "below"
    return f"{horizontal}_{vertical}"


def inverse_relation(relation: str) -> str:
    """Relation of ``v`` w.r.t. ``u`` given the relation of ``u`` w.r.t. ``v``."""
    return INVERSE_RELATION[relation]
