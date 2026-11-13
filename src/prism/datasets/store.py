"""Read access to processed layouts, graphs and images.

``LayoutStore`` hides where the data lives:

- per-sample JSON/PNG files of one dataset (the canonical, human-inspectable
  format, written by phases 02-04);
- the packed ``layouts.jsonl.gz`` / ``graphs.jsonl.gz`` / ``images.zip`` files
  created by phase 05 (``--pack``) for fast I/O on the cluster file system;
- for ``--dataset combined``: the processed folders of every member dataset
  (RPLAN + ResPlan). Layout ids carry the dataset prefix (``rplan_75342``,
  ``resplan_123``), so each id is routed to the right folder.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from PIL import Image

from prism.common.io import read_json, read_jsonl_gz
from prism.common.workspace import Workspace
from prism.datasets.schema import dataset_of, source_id_of


class LayoutStore:
    """Cached access to the layout / graph JSON and renders of a workspace."""

    def __init__(self, ws: Workspace, prefer_packed: Optional[bool] = None, cache: bool = True) -> None:
        """Open the store; packed files are used when present and ``prefer_packed`` is true."""
        self.ws = ws
        self.cache = cache
        self._layouts: Dict[str, Dict[str, Any]] = {}
        self._graphs: Dict[str, Dict[str, Any]] = {}
        self._images_zip: Optional[zipfile.ZipFile] = None
        self.members: Dict[str, Workspace] = {d: ws.member(d) for d in ws.member_datasets}
        if prefer_packed is None:
            prefer_packed = bool(ws.cfg["runtime"].get("prefer_packed", False))
        self.packed = prefer_packed and (ws.packed_dir / "layouts.jsonl.gz").exists()
        if self.packed:
            self._load_packed()

    # ---------------------------------------------------------------- packed
    def _load_packed(self) -> None:
        """Load the packed layouts/graphs into memory (images stay in the zip)."""
        for rec in read_jsonl_gz(self.ws.packed_dir / "layouts.jsonl.gz"):
            self._layouts[str(rec["id"])] = rec
        graphs_file = self.ws.packed_dir / "graphs.jsonl.gz"
        if graphs_file.exists():
            for rec in read_jsonl_gz(graphs_file):
                self._graphs[str(rec["id"])] = rec
        zip_file = self.ws.packed_dir / "images.zip"
        if zip_file.exists():
            self._images_zip = zipfile.ZipFile(zip_file, "r")

    def _member_ws(self, sid: str) -> Workspace:
        """Workspace that holds the files of ``sid``."""
        if len(self.members) == 1:
            return next(iter(self.members.values()))
        return self.members[dataset_of(sid)]

    # ------------------------------------------------------------------ ids
    def ids(self) -> List[str]:
        """Sorted list of every layout id available (all member datasets)."""
        if self.packed:
            return sorted(self._layouts.keys(), key=_id_sort_key)
        out: List[str] = []
        for mws in self.members.values():
            out.extend(p.stem for p in mws.layouts_dir.glob("*.json"))
        return sorted(out, key=_id_sort_key)

    def graph_ids(self) -> List[str]:
        """Sorted list of ids that have a graph JSON."""
        if self.packed and self._graphs:
            return sorted(self._graphs.keys(), key=_id_sort_key)
        out: List[str] = []
        for mws in self.members.values():
            out.extend(p.stem for p in mws.graphs_dir.glob("*.json"))
        return sorted(out, key=_id_sort_key)

    # ---------------------------------------------------------------- access
    def layout(self, sid: str) -> Dict[str, Any]:
        """Return the layout dictionary of ``sid``."""
        sid = str(sid)
        if sid in self._layouts:
            return self._layouts[sid]
        data = read_json(self._member_ws(sid).layouts_dir / f"{sid}.json")
        if self.cache:
            self._layouts[sid] = data
        return data

    def graph(self, sid: str) -> Dict[str, Any]:
        """Return the graph dictionary of ``sid``."""
        sid = str(sid)
        if sid in self._graphs:
            return self._graphs[sid]
        data = read_json(self._member_ws(sid).graphs_dir / f"{sid}.json")
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
        return Image.open(self._member_ws(sid).images_dir / f"{sid}.png").convert("RGB")

    def image_path(self, sid: str, source: str) -> Path:
        """File of the vision-pipeline input of ``sid`` for ``source`` = render | snapshot | img."""
        return resolve_image_path(self._member_ws(sid), sid, source)

    def iter_layouts(self, ids: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over layouts (all, or the given ids)."""
        for sid in ids or self.ids():
            yield self.layout(sid)


def _id_sort_key(sid: str) -> Any:
    """Sort by dataset prefix, then numerically when the source id is a number."""
    try:
        ds, src = dataset_of(sid), source_id_of(sid)
    except ValueError:
        ds, src = "", str(sid)
    return (ds, 0, int(src), "") if src.isdigit() else (ds, 1, 0, src)


def load_splits(ws: Workspace) -> Dict[str, List[str]]:
    """Load ``splits.json`` (``{"train": [...], "val": [...], "test": [...]}``)."""
    if not ws.splits_file.exists():
        hint = (" For --dataset combined, run phase 05 for every member dataset first, then with "
                "--dataset combined.") if ws.is_combined else ""
        raise FileNotFoundError(f"Splits not found: {ws.splits_file}. Run phase 05 first.{hint}")
    return read_json(ws.splits_file)


def resolve_image_path(ws: Workspace, sid: str, source: str) -> Path:
    """Path of the image used by the vision pipeline for ``source`` = render|snapshot|img.

    Renders exist for every dataset; ``snapshot`` and ``img`` are RPLAN-only
    (Graph2Plan files named by the original RPLAN id).
    """
    if source == "render":
        return ws.images_dir / f"{sid}.png"
    if ws.dataset != "rplan":
        raise ValueError(f"Image source '{source}' exists only for RPLAN; use --image-source render.")
    if source == "snapshot":
        return ws.raw_path("rplan.snapshot_dir") / f"{source_id_of(sid)}.png"
    if source == "img":
        return ws.raw_path("rplan.img_dir") / f"{source_id_of(sid)}.png"
    raise ValueError(f"Unknown image source '{source}'")
