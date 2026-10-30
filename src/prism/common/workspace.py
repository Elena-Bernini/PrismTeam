"""Filesystem layout of a run.

Every artifact is scoped by dataset and mode so that a preview run never
overwrites a full run::

    data/processed/<dataset>/<mode>/layouts_json/ ...
    artifacts/<dataset>/<mode>/indexes/ ...
    outputs/<dataset>/<mode>/<NN_phase>/report.html
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.prism.common.config import resolve_path


@dataclass
class Workspace:
    """Resolved directories for one ``(dataset, mode)`` pair."""

    cfg: Dict[str, Any]

    @property
    def dataset(self) -> str:
        """Dataset name, e.g. ``rplan``."""
        return str(self.cfg["dataset"])

    @property
    def mode(self) -> str:
        """Run mode, ``preview`` or ``full``."""
        return str(self.cfg["mode"])

    # ------------------------------------------------------------------ roots
    @property
    def data_root(self) -> Path:
        """Root folder that contains ``raw/`` and ``processed/``."""
        return resolve_path(self.cfg["paths"]["data_root"])

    @property
    def raw_root(self) -> Path:
        """Folder with the untouched dataset files."""
        return self.data_root / "raw"

    @property
    def processed(self) -> Path:
        """Processed data for this dataset and mode."""
        override = self.cfg["paths"].get("processed_override")
        if override:
            return resolve_path(override)
        return self.data_root / "processed" / self.dataset / self.mode

    @property
    def artifacts(self) -> Path:
        """Models, embeddings and indexes for this dataset and mode."""
        return resolve_path(self.cfg["paths"]["artifacts_root"]) / self.dataset / self.mode

    @property
    def outputs(self) -> Path:
        """Reports and figures for this dataset and mode."""
        return resolve_path(self.cfg["paths"]["outputs_root"]) / self.dataset / self.mode

    # ------------------------------------------------------------- processed
    @property
    def layouts_dir(self) -> Path:
        """One layout JSON per floor plan."""
        return self.processed / "layouts_json"

    @property
    def graphs_dir(self) -> Path:
        """One graph JSON per floor plan."""
        return self.processed / "graphs_json"

    @property
    def images_dir(self) -> Path:
        """Semantic renders produced by phase 04."""
        return self.processed / "images"

    @property
    def metadata_dir(self) -> Path:
        """Metadata, statistics, id lists."""
        return self.processed / "metadata"

    @property
    def splits_file(self) -> Path:
        """Train/val/test split produced by phase 05."""
        return self.processed / "splits" / "splits.json"

    @property
    def packed_dir(self) -> Path:
        """Packed copies (jsonl.gz + npy) for fast cluster I/O."""
        return self.processed / "packed"

    # -------------------------------------------------------------- artifacts
    @property
    def gnn_dir(self) -> Path:
        """Trained GNN checkpoint and training logs."""
        return self.artifacts / "gnn"

    def index_dir(self, name: str) -> Path:
        """Folder of one vector index, e.g. ``graph_descriptor`` or ``vision_dinov2_base_render``."""
        return self.artifacts / "indexes" / name

    @property
    def indexes_root(self) -> Path:
        """Parent folder of all vector indexes."""
        return self.artifacts / "indexes"

    # ---------------------------------------------------------------- outputs
    def phase_dir(self, phase: str) -> Path:
        """Output folder for one phase, e.g. ``03_build_graphs``."""
        path = self.outputs / phase
        path.mkdir(parents=True, exist_ok=True)
        return path

    def raw_path(self, key: str) -> Path:
        """Resolve a dataset-relative path from the config (e.g. ``rplan.pkl``)."""
        section, name = key.split(".", 1)
        return self.data_root / str(self.cfg[section][name])
