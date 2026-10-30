from pathlib import Path
from collections import Counter
import json
import numpy as np

LAYOUT_DIR = Path("data/processed/rplan/layouts_json")
OUT = Path("outputs/dataset_audit")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    coarse_counter = Counter()
    raw_type_counter = Counter()
    relation_counter = Counter()
    room_count_per_plan = []
    edge_count_per_plan = []
    area_by_coarse_type = {}

    layout_paths = list(LAYOUT_DIR.glob("*.json"))

    if not layout_paths:
        raise RuntimeError("No layout JSON files found.")

    for path in layout_paths:
        with open(path, "r", encoding="utf-8") as f:
            layout = json.load(f)

        rooms = layout["rooms"]
        edges = layout["edges"]

        room_count_per_plan.append(len(rooms))
        edge_count_per_plan.append(len(edges))

        for room in rooms:
            coarse_type = room["coarse_type"]
            raw_type_id = room["raw_type_id"]

            coarse_counter[coarse_type] += 1
            raw_type_counter[str(raw_type_id)] += 1

            area_by_coarse_type.setdefault(coarse_type, []).append(room["area_ratio"])

        for edge in edges:
            relation_counter[edge["relation"]] += 1

    stats = {
        "num_layouts": len(layout_paths),
        "coarse_type_counts": dict(coarse_counter),
        "raw_type_id_counts": dict(raw_type_counter),
        "relation_type_counts": dict(relation_counter),
        "room_count_per_plan": {
            "min": int(np.min(room_count_per_plan)),
            "max": int(np.max(room_count_per_plan)),
            "mean": float(np.mean(room_count_per_plan)),
            "median": float(np.median(room_count_per_plan)),
        },
        "edge_count_per_plan": {
            "min": int(np.min(edge_count_per_plan)),
            "max": int(np.max(edge_count_per_plan)),
            "mean": float(np.mean(edge_count_per_plan)),
            "median": float(np.median(edge_count_per_plan)),
        },
        "area_by_coarse_type_mean": {
            coarse_type: float(np.mean(values))
            for coarse_type, values in area_by_coarse_type.items()
        },
    }

    print(json.dumps(stats, indent=2))

    out_path = OUT / "rplan_processed_statistics.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved statistics to {out_path}")


if __name__ == "__main__":
    main()