"""Matplotlib helpers for the preview figures embedded in the HTML reports."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend: works on the cluster and inside scripts
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from src.prism.datasets.schema import Taxonomy  # noqa: E402
from src.prism.geometry.polygon import bbox_to_polygon  # noqa: E402

DOOR_COLOR = "#ffd600"
RELATION_SHORT = {
    "left_of": "L", "right_of": "R", "above": "A", "below": "B",
    "left_above": "LA", "right_above": "RA", "left_below": "LB", "right_below": "RB",
    "inside": "IN", "surrounding": "SUR",
}


def setup_axes(ax: Any, title: str = "", canvas: float = 256.0) -> None:
    """Image-coordinate axes (origin top-left, y downward), no ticks."""
    ax.set_xlim(0, canvas)
    ax.set_ylim(canvas, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)


def draw_layout(ax: Any, layout: Dict[str, Any], taxonomy: Taxonomy, labels: bool = True,
                visible: Optional[Sequence[int]] = None, title: str = "", door: bool = True) -> None:
    """Draw rooms (polygons), boundary and front door of a layout."""
    canvas = float(layout["coordinate_system"]["width"])
    vis = None if visible is None else {int(v) for v in visible}
    boundary = layout.get("boundary") or []
    if len(boundary) >= 3:
        ax.add_patch(MplPolygon(boundary, closed=True, facecolor="#e0e0e0", edgecolor="none", zorder=1))
    for r in layout["rooms"]:
        poly = r.get("polygon") or bbox_to_polygon(r["bbox"])
        hidden = vis is not None and int(r["id"]) not in vis
        ax.add_patch(MplPolygon(poly, closed=True, zorder=2,
                                facecolor="#bdbdbd" if hidden else taxonomy.color(int(r["room_type_id"])),
                                edgecolor="#333333", linewidth=0.8, hatch="///" if hidden else None))
        if labels and not hidden:
            cx, cy = r["centroid"]
            ax.text(cx, cy, f"{r['id']}\n{r['room_type_name']}", ha="center", va="center", fontsize=5, zorder=6)
    if len(boundary) >= 3:
        ax.add_patch(MplPolygon(boundary, closed=True, facecolor="none", edgecolor="black", linewidth=1.8, zorder=4))
    fd = layout.get("front_door")
    if door and fd:
        (x1, y1), (x2, y2) = fd["segment"]
        ax.plot([x1, x2], [y1, y2], color=DOOR_COLOR, linewidth=4, solid_capstyle="butt", zorder=7)
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.6, zorder=8)
    setup_axes(ax, title, canvas)


def draw_graph(ax: Any, layout: Dict[str, Any], graph: Dict[str, Any], taxonomy: Taxonomy, title: str = "",
               relation_labels: bool = True) -> None:
    """Draw the layout faintly with graph nodes and relation-labelled edges on top.

    Each undirected adjacency is drawn once, labelled with the relation of the
    lower-id node w.r.t. the higher-id node (e.g. ``0 L 3`` = room 0 is left of 3).
    """
    draw_layout(ax, layout, taxonomy, labels=False, title=title)
    pos = {int(nd["id"]): nd["centroid"] for nd in graph["nodes"]}
    for e in graph["edges"]:
        u, v = int(e["source"]), int(e["target"])
        if u > v:
            continue
        (x1, y1), (x2, y2) = pos[u], pos[v]
        is_door = e.get("connection") == "entrance"
        ax.plot([x1, x2], [y1, y2], color="#d9822b" if is_door else "#2f6fde", linewidth=1.2,
                linestyle="--" if is_door else "-", zorder=9)
        if relation_labels:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{u}{RELATION_SHORT.get(e['relation'], '?')}{v}",
                    fontsize=4.5, ha="center", va="center", zorder=11,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
    for nd in graph["nodes"]:
        x, y = nd["centroid"]
        is_door = nd.get("kind") == "front_door"
        ax.scatter([x], [y], s=28 if not is_door else 40, marker="s" if is_door else "o",
                   c=DOOR_COLOR if is_door else "white", edgecolors="black", linewidths=0.8, zorder=10)
        ax.text(x, y - 7, str(nd["id"]), fontsize=5.5, ha="center", zorder=12)


def image_grid(images: List[Any], titles: List[str], ncols: int = 6, cell: float = 2.0, suptitle: str = "") -> Any:
    """Grid of PIL images / arrays with titles; returns the Matplotlib figure."""
    n = max(1, len(images))
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows + (0.4 if suptitle else 0)))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, img, t in zip(axes, images, titles):
        if img is not None:
            ax.imshow(img)
        ax.set_title(t, fontsize=7)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def bar_chart(labels: List[str], values: List[float], title: str, ylabel: str = "", horizontal: bool = False) -> Any:
    """Simple single-series bar chart."""
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(labels) + 2), 3.2) if not horizontal else (6, 0.3 * len(labels) + 1.2))
    color = "#2f6fde"
    if horizontal:
        ax.barh(labels, values, color=color)
        ax.invert_yaxis()
    else:
        ax.bar(labels, values, color=color)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for t in ax.get_xticklabels():
            t.set_ha("right")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def grouped_bars(groups: List[str], series: Dict[str, List[float]], title: str, ylabel: str = "") -> Any:
    """Grouped bar chart: one group per metric, one bar per pipeline."""
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(groups) + 2), 3.6))
    n = max(1, len(series))
    width = 0.8 / n
    x = np.arange(len(groups))
    palette = ["#2f6fde", "#d9822b", "#3a9e5f", "#a64ac9", "#c94a4a", "#7a7a7a", "#1fa2b8", "#b8a21f"]
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=name, color=palette[i % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
