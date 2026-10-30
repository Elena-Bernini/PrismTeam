"""Small, dependency-free I/O helpers (JSON, JSONL.GZ, CSV, NumPy, progress bars)."""

from __future__ import annotations

import csv
import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_dir(path: Path) -> Path:
    """Delete ``path`` if it exists and recreate it empty."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays and tuples into JSON-friendly types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any, indent: Optional[int] = 2) -> None:
    """Write ``obj`` as UTF-8 JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=indent)


def write_jsonl_gz(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Write records to a gzip-compressed JSON-lines file and return how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), separators=(",", ":")))
            f.write("\n")
            count += 1
    return count


def read_jsonl_gz(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over the records of a gzip-compressed JSON-lines file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write records to an uncompressed JSON-lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record)))
            f.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read an uncompressed JSON-lines file into a list."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV using the union of all keys as header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_jsonable(v) for k, v in row.items()})


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of string dictionaries."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_vectors(path: Path, vectors: np.ndarray) -> None:
    """Save a float32 matrix as ``.npy``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(vectors, dtype=np.float32))


def progress(iterable: Iterable[Any], desc: str = "", total: Optional[int] = None) -> Iterable[Any]:
    """Wrap ``iterable`` in a tqdm progress bar when tqdm is installed, else return it unchanged."""
    try:
        from tqdm import tqdm
    except ImportError:
        if desc:
            print(f"  {desc} ...")
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
