from pathlib import Path
from collections import Counter
import json
import pickle
import numpy as np

ROOT = Path("data/raw/rplan/extracted")
PKL_PATH = ROOT / "Interface/static/Data/data_train_converted.pkl"
OUT = Path("outputs/dataset_audit")
OUT.mkdir(parents=True, exist_ok=True)


def make_json_safe(value, max_items=5):
    """
    Convert common Python / NumPy / SciPy-MATLAB objects into something
    that can be safely written into JSON.

    This is only for inspection, not for the final dataset conversion.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        summary = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

        # For numeric arrays, save a small preview.
        if value.dtype != object:
            flat = value.flatten()
            summary["preview"] = [make_json_safe(x) for x in flat[:max_items]]
        else:
            # Object arrays may contain MATLAB structs, so do not expand deeply.
            flat = value.flatten()
            summary["preview"] = [str(type(x)) for x in flat[:max_items]]

        return summary

    if isinstance(value, (list, tuple)):
        return [make_json_safe(x) for x in value[:max_items]]

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    # MATLAB structs loaded by scipy often expose __dict__.
    if hasattr(value, "__dict__"):
        return {
            "type": str(type(value)),
            "attributes": list(value.__dict__.keys()),
        }

    return str(value)


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {ROOT}")

    files = [p for p in ROOT.rglob("*") if p.is_file()]
    extensions = Counter(p.suffix.lower() for p in files)

    report = {
        "dataset_root": str(ROOT),
        "num_files": len(files),
        "extensions": dict(extensions),
        "first_100_files": [str(p) for p in files[:100]],
        "pkl_exists": PKL_PATH.exists(),
        "pkl_path": str(PKL_PATH),
    }

    print(json.dumps(report, indent=2))

    with open(OUT / "rplan_file_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not PKL_PATH.exists():
        raise FileNotFoundError(
            "data_train_converted.pkl not found. Check your extracted folder."
        )

    with open(PKL_PATH, "rb") as f:
        obj = pickle.load(f)

    print("\nPKL keys:", obj.keys())

    data = obj["data"]
    print("Number of samples:", len(data))

    sample = data[0]
    attrs = [a for a in dir(sample) if not a.startswith("_")]
    print("Sample attributes:", attrs)

    sample_report = {}

    for attr in attrs:
        value = getattr(sample, attr)

        sample_report[attr] = {
            "python_type": str(type(value)),
            "summary": make_json_safe(value),
        }

    with open(OUT / "train_converted_sample_report.json", "w", encoding="utf-8") as f:
        json.dump(sample_report, f, indent=2)

    print("\nSaved:")
    print(OUT / "rplan_file_audit.json")
    print(OUT / "train_converted_sample_report.json")


if __name__ == "__main__":
    main()