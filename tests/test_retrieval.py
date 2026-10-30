"""Unit tests for the vector index, rank fusion and metrics."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from prism.evaluation.metrics import generalized_jaccard, ranking_metrics  # noqa: E402
from prism.partial.compatibility import counter_coverage  # noqa: E402
from prism.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from prism.retrieval.index import VectorIndex  # noqa: E402


def test_numpy_index_cosine_and_save(tmp_path):
    """Nearest neighbour is correct, self is excluded, save/load round-trips."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 8)).astype(np.float32)
    x[1] = x[0] * 2.0 + 0.01  # same direction as row 0
    ix = VectorIndex([str(i) for i in range(50)], x, backend="numpy")
    res = ix.search_by_id("0", 3)
    assert res[0]["id"] == "1" and all(r["id"] != "0" for r in res)
    ix.save(tmp_path / "ix")
    ix2 = VectorIndex.load(tmp_path / "ix", backend="numpy")
    assert ix2.search_by_id("0", 1)[0]["id"] == "1"


def test_rrf_prefers_consensus():
    """An item ranked well by both lists wins the fusion."""
    a = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    b = [{"id": "y"}, {"id": "w"}, {"id": "x"}]
    fused = reciprocal_rank_fusion([a, b], k=60, top_k=2)
    assert fused[0]["id"] in {"x", "y"} and {f["id"] for f in fused} == {"x", "y"}


def test_metrics():
    """Generalised Jaccard, coverage and Recall@K on toy inputs."""
    assert generalized_jaccard(Counter(a=2, b=1), Counter(a=1, b=1)) == 2 / 3
    assert counter_coverage(Counter(a=2), Counter(a=1, b=5)) == 0.5
    m = ranking_metrics(["1", "2", "3"], {"2", "9"}, 3)
    assert m["recall_at_k"] == 0.5 and m["mrr"] == 0.5
