"""Masking strategies that turn a complete layout into a partial query.

Strategies (``partial.strategy``):

- ``random``           remove ``round(remove_ratio * n)`` random rooms (Pier)
- ``remove_type``      remove every room of fine type ``remove_type`` (Pier)
- ``remove_bedroom``   remove one bedroom-like room (Pier)
- ``remove_smallest``  remove the smallest room (Pier)
- ``remove_largest``   remove the largest room (Pier)
- ``keep_connected``   keep ``keep_count`` rooms that form a connected sub-graph,
                       grown by BFS from a random room (Elena)

The boundary and the front door are always kept (they are part of the
building shell). With ``protect_entrance`` the room behind the front door is
never removed. At least one room is always visible.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from src.prism.datasets.schema import Taxonomy
from src.prism.graph.build import room_edges


def choose_visible_rooms(layout: Dict[str, Any], graph: Dict[str, Any], pcfg: Dict[str, Any],
                         taxonomy: Taxonomy, rng: random.Random) -> Tuple[Set[int], Set[int]]:
    """Return ``(visible_room_ids, removed_room_ids)`` for one layout."""
    rooms = layout["rooms"]
    all_ids = {int(r["id"]) for r in rooms}
    if len(all_ids) <= 1:
        return all_ids, set()
    strategy = pcfg["strategy"]
    protected: Set[int] = set()
    door = layout.get("front_door") or {}
    if pcfg.get("protect_entrance") and door.get("entrance_room_id") is not None:
        protected.add(int(door["entrance_room_id"]))
    candidates = sorted(all_ids - protected)

    def random_removal() -> Set[int]:
        """Remove a random fraction of the removable rooms."""
        k = max(1, int(round(len(all_ids) * float(pcfg["remove_ratio"]))))
        k = min(k, len(candidates), len(all_ids) - 1)
        return set(rng.sample(candidates, k)) if k > 0 else set()

    if strategy == "random":
        removed = random_removal()
    elif strategy == "remove_type":
        if pcfg.get("remove_type") is None:
            raise ValueError("partial.remove_type must be set for strategy remove_type")
        removed = {int(r["id"]) for r in rooms if int(r["room_type_id"]) == int(pcfg["remove_type"])} - protected
        removed = removed or random_removal()
    elif strategy == "remove_bedroom":
        beds = [int(r["id"]) for r in rooms if taxonomy.is_bedroom(int(r["room_type_id"])) and int(r["id"]) not in protected]
        removed = {rng.choice(beds)} if beds else ({rng.choice(candidates)} if candidates else set())
    elif strategy in {"remove_smallest", "remove_largest"}:
        pool = [r for r in rooms if int(r["id"]) not in protected]
        key = lambda r: float(r.get("polygon_area_ratio", r["area_ratio"]))  # noqa: E731
        pick = min(pool, key=key) if strategy == "remove_smallest" else max(pool, key=key)
        removed = {int(pick["id"])} if pool else set()
    elif strategy == "keep_connected":
        visible = connected_subset(graph, int(pcfg["keep_count"]), rng, must_include=protected)
        return visible, all_ids - visible
    else:
        raise ValueError(f"Unknown partial strategy '{strategy}'")

    if len(removed) >= len(all_ids):
        removed = set(sorted(removed)[:-1])
    return all_ids - removed, removed


def connected_subset(graph: Dict[str, Any], keep_count: int, rng: random.Random,
                     must_include: Optional[Set[int]] = None) -> Set[int]:
    """Grow a connected set of ``keep_count`` rooms by randomised BFS (Elena's strategy)."""
    room_ids = sorted({int(nd["id"]) for nd in graph["nodes"] if nd.get("kind", "room") == "room"})
    keep_count = max(1, min(keep_count, len(room_ids) - 1)) if len(room_ids) > 1 else 1
    adj: Dict[int, List[int]] = {i: [] for i in room_ids}
    for e in room_edges(graph):
        u, v = int(e["source"]), int(e["target"])
        if u in adj and v in adj:
            adj[u].append(v)
    start = next(iter(must_include)) if must_include else rng.choice(room_ids)
    selected, queue = {start}, deque([start])
    while queue and len(selected) < keep_count:
        cur = queue.popleft()
        nbrs = adj[cur][:]
        rng.shuffle(nbrs)
        for nb in nbrs:
            if nb not in selected and len(selected) < keep_count:
                selected.add(nb)
                queue.append(nb)
    # Disconnected plans: top up with random rooms.
    rest = [i for i in room_ids if i not in selected]
    rng.shuffle(rest)
    while len(selected) < keep_count and rest:
        selected.add(rest.pop())
    return selected
