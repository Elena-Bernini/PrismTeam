"""Filesystem layout of a run.

Every artifact is scoped by dataset and mode so that a preview run never
overwrites a full run::

    data/processed/<dataset>/<mode>/layouts_json/ ...
    artifacts/<dataset>/<mode>/indexes/ ...
    outputs/<dataset>/<mode>/<NN_phase>/report.html

``<dataset>`` is ``rplan``, ``resplan`` or ``combined``. The combined workspace
has no layouts of its own: it reads the processed data of its member datasets
(``combined.datasets``) and stores only its merged splits, metadata, packed
files, models, indexes and reports.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from prism.common.config import resolve_path


@dataclass
class Workspace:
    """Resolved directories for one ``(dataset, mode)`` pair."""

    cfg: Dict[str, Any]

    @property
    def dataset(self) -> str:
        """Dataset name, e.g. ``rplan``."""
        return str(self.cfg["dataset"])

    @property
    def is_combined(self) -> bool:
        """True for the ``combined`` workspace (several datasets in one gallery)."""
        return self.dataset == "combined"

    @property
    def member_datasets(self) -> List[str]:
        """Datasets whose processed data this workspace reads."""
        if self.is_combined:
            return [str(d) for d in self.cfg["combined"]["datasets"]]
        return [self.dataset]

    def member(self, dataset: str) -> "Workspace":
        """Workspace of one member dataset (same mode, no processed override)."""
        if not self.is_combined:
            if dataset != self.dataset:
                raise ValueError(f"'{dataset}' is not part of workspace '{self.dataset}'")
            return self
        cfg = copy.copy(self.cfg)
        cfg["dataset"] = dataset
        cfg["paths"] = {k: v for k, v in self.cfg["paths"].items() if k != "processed_override"}
        return Workspace(cfg)

    def members(self) -> List["Workspace"]:
        """One workspace per member dataset (just ``[self]`` for a single dataset)."""
        return [self.member(d) for d in self.member_datasets] if self.is_combined else [self]

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
        """Resolve a data-root-relative path from the config (e.g. ``rplan.pkl``, ``resplan.split_json``)."""
        section, name = key.split(".", 1)
        return self.data_root / str(self.cfg[section][name])
