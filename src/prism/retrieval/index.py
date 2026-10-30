"""Vector index with a FAISS backend and an exact NumPy fallback.

On disk an index folder contains::

    vectors.npy     raw (un-normalised) embeddings, one row per layout
    ids.json        layout id of each row
    meta.json       encoder name, metric, dimension, backend, extra info
    scaler.json     optional standardisation (descriptor features)
    index.faiss     FAISS index (only when FAISS is installed)

``vectors.npy`` is always written, so an index built on the cluster with FAISS
can be queried on a laptop without FAISS (NumPy exact search) and vice versa.
For ~80k layouts exact search with NumPy is still well under a second per query.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.prism.common.io import read_json, write_json


def faiss_available() -> bool:
    """True when ``import faiss`` works."""
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def fit_scaler(x: np.ndarray, eps: float = 1e-6) -> Dict[str, List[float]]:
    """Per-feature mean/std for standardisation (used for hand-crafted descriptors)."""
    return {"mean": x.mean(axis=0).tolist(), "std": np.maximum(x.std(axis=0), eps).tolist()}


def apply_scaler(x: np.ndarray, scaler: Optional[Dict[str, List[float]]]) -> np.ndarray:
    """Standardise ``x`` with a fitted scaler (identity when ``scaler`` is None)."""
    if not scaler:
        return np.asarray(x, dtype=np.float32)
    return ((np.asarray(x, dtype=np.float32) - np.asarray(scaler["mean"], dtype=np.float32))
            / np.asarray(scaler["std"], dtype=np.float32)).astype(np.float32)


class VectorIndex:
    """Nearest-neighbour index over layout embeddings."""

    def __init__(self, ids: List[str], vectors: np.ndarray, metric: str = "cosine", backend: str = "auto",
                 kind: str = "flat", scaler: Optional[Dict[str, List[float]]] = None,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        """Build the index in memory from raw vectors."""
        self.ids = [str(i) for i in ids]
        self.raw = np.asarray(vectors, dtype=np.float32)
        self.metric = metric
        self.kind = kind
        self.scaler = scaler
        self.meta = dict(meta or {})
        self.id_to_row = {sid: k for k, sid in enumerate(self.ids)}
        self.prepared = self.prepare(self.raw)
        if backend == "auto":
            backend = "faiss" if faiss_available() else "numpy"
        self.backend = backend
        self._faiss = self._build_faiss() if backend == "faiss" else None

    # ------------------------------------------------------------- vectors
    def prepare(self, x: np.ndarray) -> np.ndarray:
        """Apply scaler and (for cosine) L2 normalisation to raw vectors."""
        x = apply_scaler(np.atleast_2d(x), self.scaler)
        return l2_normalize(x) if self.metric == "cosine" else x.astype(np.float32)

    def _build_faiss(self) -> Any:
        """Create the FAISS index (flat or HNSW, inner product or L2)."""
        import faiss

        d = self.prepared.shape[1]
        metric = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
        if self.kind == "hnsw":
            index = faiss.IndexHNSWFlat(d, 32, metric)
            index.hnsw.efConstruction, index.hnsw.efSearch = 80, 64
        else:
            index = faiss.IndexFlatIP(d) if self.metric == "cosine" else faiss.IndexFlatL2(d)
        index.add(np.ascontiguousarray(self.prepared))
        return index

    # -------------------------------------------------------------- search
    def search_prepared(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search already-prepared query vectors; returns ``(scores, rows)``.

        Scores are similarities (higher is better): cosine similarity, or the
        negative squared L2 distance.
        """
        k = min(k, len(self.ids))
        q = np.ascontiguousarray(q, dtype=np.float32)
        if self._faiss is not None:
            scores, rows = self._faiss.search(q, k)
            if self.metric == "l2":
                scores = -scores
            return scores, rows
        if self.metric == "cosine":
            sims = q @ self.prepared.T
        else:
            sims = -(np.sum(q ** 2, axis=1, keepdims=True) - 2 * q @ self.prepared.T
                     + np.sum(self.prepared ** 2, axis=1)[None, :])
        rows = np.argpartition(-sims, kth=min(k, sims.shape[1]) - 1, axis=1)[:, :k]
        part = np.take_along_axis(sims, rows, axis=1)
        order = np.argsort(-part, axis=1)
        return np.take_along_axis(part, order, axis=1), np.take_along_axis(rows, order, axis=1)

    def search(self, raw_queries: np.ndarray, k: int, exclude: Optional[List[Optional[str]]] = None) -> List[List[Dict[str, Any]]]:
        """Search raw query vectors; optionally exclude one id per query (the query itself)."""
        extra = 1 if exclude else 0
        scores, rows = self.search_prepared(self.prepare(raw_queries), k + extra)
        results: List[List[Dict[str, Any]]] = []
        for qi in range(scores.shape[0]):
            skip = exclude[qi] if exclude else None
            items = []
            for s, r in zip(scores[qi], rows[qi]):
                if r < 0:
                    continue
                sid = self.ids[int(r)]
                if skip is not None and sid == skip:
                    continue
                items.append({"id": sid, "score": float(s)})
                if len(items) >= k:
                    break
            results.append(items)
        return results

    def search_by_id(self, sid: str, k: int) -> List[Dict[str, Any]]:
        """Use the stored vector of ``sid`` as query and exclude ``sid`` from the results."""
        row = self.id_to_row[str(sid)]
        return self.search(self.raw[row:row + 1], k, exclude=[str(sid)])[0]

    # ------------------------------------------------------------ persist
    def save(self, folder: Path) -> None:
        """Write vectors, ids, metadata, scaler and (if FAISS) the FAISS index."""
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.raw)
        write_json(folder / "ids.json", self.ids, indent=None)
        meta = dict(self.meta)
        meta.update({"metric": self.metric, "kind": self.kind, "dim": int(self.raw.shape[1]),
                     "count": len(self.ids), "built_with_backend": self.backend})
        write_json(folder / "meta.json", meta)
        if self.scaler:
            write_json(folder / "scaler.json", self.scaler, indent=None)
        if self._faiss is not None:
            import faiss
            faiss.write_index(self._faiss, str(folder / "index.faiss"))

    @classmethod
    def load(cls, folder: Path, backend: str = "auto") -> "VectorIndex":
        """Load an index folder (re-uses ``index.faiss`` when FAISS is available)."""
        if not (folder / "vectors.npy").exists():
            raise FileNotFoundError(f"Index not found: {folder}")
        meta = read_json(folder / "meta.json")
        scaler = read_json(folder / "scaler.json") if (folder / "scaler.json").exists() else None
        ids = read_json(folder / "ids.json")
        vectors = np.load(folder / "vectors.npy")
        if backend == "auto":
            backend = "faiss" if faiss_available() else "numpy"
        obj = cls.__new__(cls)
        obj.ids = [str(i) for i in ids]
        obj.raw = vectors.astype(np.float32)
        obj.metric = meta["metric"]
        obj.kind = meta.get("kind", "flat")
        obj.scaler = scaler
        obj.meta = meta
        obj.id_to_row = {sid: k for k, sid in enumerate(obj.ids)}
        obj.prepared = obj.prepare(obj.raw)
        obj.backend = backend
        obj._faiss = None
        if backend == "faiss":
            import faiss
            fpath = folder / "index.faiss"
            obj._faiss = faiss.read_index(str(fpath)) if fpath.exists() else obj._build_faiss()
        return obj


def list_indexes(indexes_root: Path) -> List[str]:
    """Names of every index folder under ``indexes_root``."""
    if not indexes_root.exists():
        return []
    return sorted(p.name for p in indexes_root.iterdir() if (p / "vectors.npy").exists())
