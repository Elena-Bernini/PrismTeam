"""Read access to processed layouts, graphs and images.

``LayoutStore`` hides whether the data lives in per-sample JSON/PNG files (the
canonical, human-inspectable format) or in the packed ``layouts.jsonl.gz`` /
``graphs.jsonl.gz`` / ``images.zip`` files created by phase 05 (``--pack``) for
fast I/O on the cluster file system (few large files instead of ~240k small ones).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from PIL import Image

from src.prism.common.io import read_json, read_jsonl_gz
from src.prism.common.workspace import Workspace


class LayoutStore:
    """Cached access to the layout / graph JSON of one workspace."""

    def __init__(self, ws: Workspace, prefer_packed: Optional[bool] = None, cache: bool = True) -> None:
        """Open the store; packed files are used when present and ``prefer_packed`` is true."""
        self.ws = ws
        self.cache = cache
        self._layouts: Dict[str, Dict[str, Any]] = {}
        self._graphs: Dict[str, Dict[str, Any]] = {}
        self._images_zip: Optional[zipfile.ZipFile] = None
        if prefer_packed is None:
            prefer_packed = bool(ws.cfg["runtime"].get("prefer_packed", False))
        self.packed = prefer_packed and (ws.packed_dir / "layouts.jsonl.gz").exists()
        if self.packed:
            self._load_packed()

    # ---------------------------------------------------------------- packed
    def _load_packed(self) -> None:
        """Load the packed layouts/graphs into memory (images are memory-mapped)."""
        for rec in read_jsonl_gz(self.ws.packed_dir / "layouts.jsonl.gz"):
            self._layouts[str(rec["id"])] = rec
        graphs_file = self.ws.packed_dir / "graphs.jsonl.gz"
        if graphs_file.exists():
            for rec in read_jsonl_gz(graphs_file):
                self._graphs[str(rec["id"])] = rec
        zip_file = self.ws.packed_dir / "images.zip"
        if zip_file.exists():
            self._images_zip = zipfile.ZipFile(zip_file, "r")

    # ------------------------------------------------------------------ ids
    def ids(self) -> List[str]:
        """Sorted list of every layout id available in this workspace."""
        if self.packed:
            return sorted(self._layouts.keys(), key=_id_sort_key)
        return sorted((p.stem for p in self.ws.layouts_dir.glob("*.json")), key=_id_sort_key)

    def graph_ids(self) -> List[str]:
        """Sorted list of ids that have a graph JSON."""
        if self.packed and self._graphs:
            return sorted(self._graphs.keys(), key=_id_sort_key)
        return sorted((p.stem for p in self.ws.graphs_dir.glob("*.json")), key=_id_sort_key)

    # ---------------------------------------------------------------- access
    def layout(self, sid: str) -> Dict[str, Any]:
        """Return the layout dictionary of ``sid``."""
        sid = str(sid)
        if sid in self._layouts:
            return self._layouts[sid]
        data = read_json(self.ws.layouts_dir / f"{sid}.json")
        if self.cache:
            self._layouts[sid] = data
        return data

    def graph(self, sid: str) -> Dict[str, Any]:
        """Return the graph dictionary of ``sid``."""
        sid = str(sid)
        if sid in self._graphs:
            return self._graphs[sid]
        data = read_json(self.ws.graphs_dir / f"{sid}.json")
        if self.cache:
            self._graphs[sid] = data
        return data

    def image(self, sid: str) -> Image.Image:
        """Return the phase-04 render of ``sid`` as a PIL RGB image."""
        sid = str(sid)
        if self._images_zip is not None:
            try:
                return Image.open(io.BytesIO(self._images_zip.read(f"{sid}.png"))).convert("RGB")
            except KeyError:
                pass
        return Image.open(self.ws.images_dir / f"{sid}.png").convert("RGB")

    def iter_layouts(self, ids: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over layouts (all, or the given ids)."""
        for sid in ids or self.ids():
            yield self.layout(sid)


def _id_sort_key(sid: str) -> Any:
    """Sort numeric ids numerically and other ids alphabetically."""
    return (0, int(sid), "") if str(sid).isdigit() else (1, 0, str(sid))


def load_splits(ws: Workspace) -> Dict[str, List[str]]:
    """Load ``splits.json`` (``{"train": [...], "val": [...], "test": [...]}``)."""
    if not ws.splits_file.exists():
        raise FileNotFoundError(f"Splits not found: {ws.splits_file}. Run phase 05 first.")
    return read_json(ws.splits_file)


def resolve_image_path(ws: Workspace, sid: str, source: str) -> Path:
    """Path of the image used by the vision pipeline for ``source`` = render|snapshot|img."""
    if source == "render":
        return ws.images_dir / f"{sid}.png"
    if source == "snapshot":
        return ws.raw_path("rplan.snapshot_dir") / f"{sid}.png"
    if source == "img":
        return ws.raw_path("rplan.img_dir") / f"{sid}.png"
    raise ValueError(f"Unknown image source '{source}'")
