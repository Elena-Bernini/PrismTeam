from pathlib import Path
import argparse
import json
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

LAYOUT_DIR = Path("data/processed/rplan/layouts_json")
SNAPSHOT_DIR = Path("data/raw/rplan/extracted/Interface/static/Data/snapshot_train")
OUT = Path("outputs/dataset_audit/layout_previews")
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


def draw_boundary(ax, layout):
    boundary = layout.get("boundary", None)

    if not boundary:
        return

    points = [[p[0], p[1]] for p in boundary if len(p) >= 2]

    if len(points) >= 3:
        poly = Polygon(
            points,
            closed=True,
            facecolor="none",
            edgecolor="black",
            linewidth=2.5,
        )
        ax.add_patch(poly)


def draw_room(ax, room):
    display_type = room.get("coarse_type", "Unknown")
    raw_type_id = room.get("raw_type_id", "NA")
    color = ROOM_COLORS.get(display_type, "#ffffff")

    polygon = room.get("polygon", None)

    if polygon is not None and isinstance(polygon, list) and len(polygon) >= 3:
        points = [[p[0], p[1]] for p in polygon if len(p) >= 2]

        patch = Polygon(
            points,
            closed=True,
            facecolor=color,
            edgecolor="black",
            linewidth=1.3,
            alpha=0.85,
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
            linewidth=1.3,
            alpha=0.85,
        )
        ax.add_patch(patch)

    cx, cy = room["centroid"]

    ax.text(
        cx,
        cy,
        f"{room['id']}\n{display_type}\nraw={raw_type_id}",
        ha="center",
        va="center",
        fontsize=5.5,
    )


def draw_layout_only(layout_path, output_path):
    with open(layout_path, "r", encoding="utf-8") as f:
        layout = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 6))

    for room in layout["rooms"]:
        draw_room(ax, room)

    draw_boundary(ax, layout)

    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_aspect("equal")
    ax.set_title(f"JSON render: {layout['id']}")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_comparison(layout_path, output_path):
    with open(layout_path, "r", encoding="utf-8") as f:
        layout = json.load(f)

    sample_id = layout["id"]
    snapshot_path = SNAPSHOT_DIR / f"{sample_id}.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

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

    axes[1].set_title(f"JSON polygon render: {sample_id}")

    for room in layout["rooms"]:
        draw_room(axes[1], room)

    draw_boundary(axes[1], layout)

    axes[1].set_xlim(0, 256)
    axes[1].set_ylim(256, 0)
    axes[1].set_aspect("equal")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None, help="Specific layout id to visualize.")
    parser.add_argument("--n", type=int, default=25, help="Number of random previews.")
    parser.add_argument("--compare", action="store_true", help="Create side-by-side comparison.")
    args = parser.parse_args()

    if args.id is not None:
        layout_path = LAYOUT_DIR / f"{args.id}.json"

        if not layout_path.exists():
            raise FileNotFoundError(f"Layout not found: {layout_path}")

        if args.compare:
            output_path = OUT / f"{args.id}_comparison.png"
            draw_comparison(layout_path, output_path)
        else:
            output_path = OUT / f"{args.id}_polygon.png"
            draw_layout_only(layout_path, output_path)

        print(f"Saved: {output_path}")
        return

    files = list(LAYOUT_DIR.glob("*.json"))

    if not files:
        raise RuntimeError("No layout JSON files found. Run conversion first.")

    sample = random.sample(files, min(args.n, len(files)))

    for path in sample:
        if args.compare:
            output_path = OUT / f"{path.stem}_comparison.png"
            draw_comparison(path, output_path)
        else:
            output_path = OUT / f"{path.stem}_polygon.png"
            draw_layout_only(path, output_path)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()