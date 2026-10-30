from pathlib import Path
import argparse
import pickle
import json
import numpy as np
from tqdm import tqdm

ROOT = Path("data/raw/rplan/extracted")
PKL_PATH = ROOT / "Interface/static/Data/data_train_converted.pkl"
TYPE_MAPPING_PATH = Path("configs/rplan_type_mapping.json")

OUT_LAYOUTS = Path("data/processed/rplan/layouts_json")
OUT_METADATA = Path("data/processed/rplan/metadata")

OUT_LAYOUTS.mkdir(parents=True, exist_ok=True)
OUT_METADATA.mkdir(parents=True, exist_ok=True)

RELATION_ID_TO_NAME = {
    0: "left_of",
    1: "right_of",
    2: "above",
    3: "below",
    4: "left_above",
    5: "right_above",
    6: "left_below",
    7: "right_below",
    8: "inside",
    9: "outside",
}


def load_type_mapping():
    if not TYPE_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Missing type mapping file: {TYPE_MAPPING_PATH}")

    with open(TYPE_MAPPING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def to_python(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def compute_grid_location(cx, cy, width=256, height=256, grid_size=5):
    gx = min(grid_size - 1, max(0, int(cx / width * grid_size)))
    gy = min(grid_size - 1, max(0, int(cy / height * grid_size)))
    return [gx, gy]


def get_type_info(type_id, type_mapping):
    item = type_mapping.get(str(type_id), None)

    if item is None:
        return {
            "type_name": f"Unknown_{type_id}",
            "coarse_type": "Unknown",
        }

    return item


def convert_sample(sample, type_mapping):
    sample_id = str(sample.name)

    boundary = np.asarray(sample.boundary)
    boxes = np.asarray(sample.box)
    edges = np.asarray(sample.edge)

    width = 256
    height = 256
    building_area = width * height

    rooms = []

    for room_id, box in enumerate(boxes):
        x_min, y_min, x_max, y_max, raw_type_id = [int(v) for v in box]

        room_width = max(0, x_max - x_min)
        room_height = max(0, y_max - y_min)
        area_pixels = room_width * room_height

        cx = x_min + room_width / 2.0
        cy = y_min + room_height / 2.0

        type_info = get_type_info(raw_type_id, type_mapping)

        room = {
            "id": room_id,
            "raw_type_id": raw_type_id,
            "type_name": type_info["type_name"],
            "coarse_type": type_info["coarse_type"],
            "bbox": [x_min, y_min, x_max, y_max],
            "width": room_width,
            "height": room_height,
            "area_pixels": area_pixels,
            "area_ratio": area_pixels / building_area,
            "centroid": [cx, cy],
            "grid_location_5x5": compute_grid_location(cx, cy, width, height),
        }

        if hasattr(sample, "rBoundary"):
            try:
                polygon = sample.rBoundary[room_id]
                room["polygon"] = to_python(polygon)
            except Exception:
                room["polygon"] = None
        else:
            room["polygon"] = None

        rooms.append(room)

    graph_edges = []

    for edge_id, edge in enumerate(edges):
        source, target, relation_id = [int(v) for v in edge]

        graph_edges.append({
            "id": edge_id,
            "source": source,
            "target": target,
            "relation_id": relation_id,
            "relation": RELATION_ID_TO_NAME.get(relation_id, f"relation_{relation_id}"),
        })

    layout = {
        "id": sample_id,
        "coordinate_system": {
            "width": width,
            "height": height,
            "origin": "top_left",
        },
        "source": {
            "dataset": "RPLAN_Graph2Plan_processed",
            "pkl_file": str(PKL_PATH),
        },
        "boundary": to_python(boundary),
        "rooms": rooms,
        "edges": graph_edges,
        "raw": {
            "order": to_python(sample.order) if hasattr(sample, "order") else None,
        },
    }

    return layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N samples. Useful for debugging.",
    )
    args = parser.parse_args()

    type_mapping = load_type_mapping()

    with open(PKL_PATH, "rb") as f:
        obj = pickle.load(f)

    data = obj["data"]

    if args.limit is not None:
        data = data[:args.limit]

    metadata = []

    for sample in tqdm(data, desc="Converting RPLAN samples"):
        layout = convert_sample(sample, type_mapping)
        out_path = OUT_LAYOUTS / f"{layout['id']}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2)

        coarse_counts = {}
        for room in layout["rooms"]:
            coarse = room["coarse_type"]
            coarse_counts[coarse] = coarse_counts.get(coarse, 0) + 1

        metadata.append({
            "id": layout["id"],
            "num_rooms": len(layout["rooms"]),
            "num_edges": len(layout["edges"]),
            "coarse_counts": coarse_counts,
            "layout_json": str(out_path),
        })

    metadata_path = OUT_METADATA / "layouts_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Converted {len(metadata)} layouts")
    print(f"Saved layouts to: {OUT_LAYOUTS}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()