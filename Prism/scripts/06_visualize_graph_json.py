from pathlib import Path
import argparse
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

LAYOUT_DIR = Path("data/processed/rplan/layouts_json")
GRAPH_DIR = Path("data/processed/rplan/graphs_json")
SNAPSHOT_DIR = Path("data/raw/rplan/extracted/Interface/static/Data/snapshot_train")
OUT = Path("outputs/dataset_audit/graph_previews")
OUT.mkdir(parents=True, exist_ok=True)

ROOM_COLORS = {
    "LivingRoom": "#eeeeee",
    "PublicArea": "#eeeeee",
    "Bedroom": "#fff2a8",
    "Bathroom": "#bde7ff",
    "Kitchen": "#ffd1dc",
    "Balcony": "#c9e8a2",
    "Storage": "#ddd0c8",
    "Unknown": "#ffffff",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_boundary(ax, layout):
    boundary = layout.get("boundary", None)

    if not boundary:
        return

    points = [[p[0], p[1]] for p in boundary if len(p) >= 2]

    if len(points) >= 3:
        patch = Polygon(
            points,
            closed=True,
            facecolor="none",
            edgecolor="black",
            linewidth=2.5,
        )
        ax.add_patch(patch)


def draw_room(ax, room):
    coarse_type = room.get("coarse_type", "Unknown")
    raw_type_id = room.get("raw_type_id", "NA")
    color = ROOM_COLORS.get(coarse_type, "#ffffff")

    polygon = room.get("polygon", None)

    if polygon is not None and isinstance(polygon, list) and len(polygon) >= 3:
        points = [[p[0], p[1]] for p in polygon if len(p) >= 2]

        patch = Polygon(
            points,
            closed=True,
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.75,
        )
        ax.add_patch(patch)
    else:
        x0, y0, x1, y1 = room["bbox"]

        patch = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.75,
        )
        ax.add_patch(patch)

    cx, cy = room["centroid"]

    ax.scatter(cx, cy, s=60, edgecolors="black", zorder=5)

    ax.text(
        cx,
        cy - 7,
        f"{room['id']}\n{coarse_type}\nraw={raw_type_id}",
        ha="center",
        va="top",
        fontsize=5.5,
        zorder=6,
    )


def draw_edges(ax, layout, graph):
    rooms_by_id = {room["id"]: room for room in layout["rooms"]}

    for edge in graph["edges"]:
        source = edge["source"]
        target = edge["target"]

        if source not in rooms_by_id or target not in rooms_by_id:
            continue

        room_a = rooms_by_id[source]
        room_b = rooms_by_id[target]

        x1, y1 = room_a["centroid"]
        x2, y2 = room_b["centroid"]

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=1.8,
            alpha=0.85,
            zorder=4,
        )

        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0

        ax.text(
            mx,
            my,
            edge.get("relation", ""),
            ha="center",
            va="center",
            fontsize=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            zorder=7,
        )


def draw_graph_preview(sample_id, compare=True):
    layout_path = LAYOUT_DIR / f"{sample_id}.json"
    graph_path = GRAPH_DIR / f"{sample_id}.json"
    snapshot_path = SNAPSHOT_DIR / f"{sample_id}.png"

    if not layout_path.exists():
        raise FileNotFoundError(f"Missing layout JSON: {layout_path}")

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph JSON: {graph_path}")

    layout = load_json(layout_path)
    graph = load_json(graph_path)

    if compare:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        axes[0].set_title(f"Original snapshot: {sample_id}")

        if snapshot_path.exists():
            img = Image.open(snapshot_path).convert("RGB")
            axes[0].imshow(img)
        else:
            axes[0].text(
                0.5,
                0.5,
                f"Snapshot not found:\n{snapshot_path}",
                ha="center",
                va="center",
            )

        axes[0].axis("off")

        ax = axes[1]
        ax.set_title(f"Graph render: {sample_id}")
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(f"Graph render: {sample_id}")

    for room in layout["rooms"]:
        draw_room(ax, room)

    draw_edges(ax, layout, graph)
    draw_boundary(ax, layout)

    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    out_path = OUT / f"{sample_id}_graph_preview.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None, help="Specific graph id to visualize.")
    parser.add_argument("--n", type=int, default=20, help="Number of random graph previews.")
    parser.add_argument("--no_compare", action="store_true", help="Do not show original snapshot.")
    args = parser.parse_args()

    compare = not args.no_compare

    if args.id is not None:
        draw_graph_preview(args.id, compare=compare)
        return

    graph_paths = list(GRAPH_DIR.glob("*.json"))

    if not graph_paths:
        raise RuntimeError("No graph JSON files found. Run script 04 first.")

    sample = random.sample(graph_paths, min(args.n, len(graph_paths)))

    for path in sample:
        draw_graph_preview(path.stem, compare=compare)


if __name__ == "__main__":
    main()