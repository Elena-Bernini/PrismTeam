"""Qualitative retrieval figures (query + top-k renders) for the HTML reports."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.prism.common.viz import image_grid


def retrieval_figure(store: Any, query_id: str, results: Sequence[Dict[str, Any]], title: str,
                     query_image: Optional[Any] = None, k: int = 7, extra: Optional[Dict[str, str]] = None) -> Any:
    """Grid with the query render followed by the top-k retrieved renders and their scores.

    ``extra`` maps result ids to an additional caption line (e.g. architectural score).
    """
    images = [query_image if query_image is not None else store.image(query_id)]
    titles = [f"QUERY {query_id}"]
    for rank, r in enumerate(list(results)[:k], start=1):
        images.append(store.image(r["id"]))
        cap = f"#{rank} {r['id']}  {r.get('score', 0):.3f}"
        if extra and r["id"] in extra:
            cap += f"\n{extra[r['id']]}"
        if r["id"] == query_id:
            cap += "  (original)"
        titles.append(cap)
    return image_grid(images, titles, ncols=len(images), cell=1.9, suptitle=title)


def comparison_figure(store: Any, query_id: str, rankings: Dict[str, List[Dict[str, Any]]], k: int = 5) -> Any:
    """One row per pipeline: query render followed by its top-k results."""
    images: List[Any] = []
    titles: List[str] = []
    for name, res in rankings.items():
        images.append(store.image(query_id))
        titles.append(f"{name}\nquery {query_id}")
        for rank, r in enumerate(res[:k], start=1):
            images.append(store.image(r["id"]))
            titles.append(f"#{rank} {r['id']} {r.get('score', 0):.3f}")
        missing = k - min(k, len(res))
        images.extend([None] * missing)
        titles.extend([""] * missing)
    return image_grid(images, titles, ncols=k + 1, cell=1.8)
