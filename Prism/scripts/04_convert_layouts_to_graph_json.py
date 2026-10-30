from pathlib import Path
import json
from tqdm import tqdm

LAYOUT_DIR = Path("data/processed/rplan/layouts_json")
OUT_GRAPH = Path("data/processed/rplan/graphs_json")
OUT_GRAPH.mkdir(parents=True, exist_ok=True)


def convert_layout_to_graph(layout):
    nodes = []

    for room in layout["rooms"]:
        nodes.append({
            "id": room["id"],
            "raw_type_id": room["raw_type_id"],
            "type_name": room["type_name"],
            "coarse_type": room["coarse_type"],
            "area_ratio": room["area_ratio"],
            "bbox": room["bbox"],
            "centroid": room["centroid"],
            "grid_location_5x5": room["grid_location_5x5"],
        })

    edges = []

    for edge in layout["edges"]:
        edges.append({
            "source": edge["source"],
            "target": edge["target"],
            "relation": edge["relation"],
            "relation_id": edge["relation_id"],
        })

    graph = {
        "id": layout["id"],
        "nodes": nodes,
        "edges": edges,
    }

    return graph


def main():
    layout_paths = sorted(LAYOUT_DIR.glob("*.json"))

    if not layout_paths:
        raise RuntimeError("No layout JSON files found.")

    for path in tqdm(layout_paths, desc="Building graph JSON"):
        with open(path, "r", encoding="utf-8") as f:
            layout = json.load(f)

        graph = convert_layout_to_graph(layout)

        with open(OUT_GRAPH / f"{graph['id']}.json", "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)

    print(f"Saved graph JSON files to {OUT_GRAPH}")


if __name__ == "__main__":
    main()