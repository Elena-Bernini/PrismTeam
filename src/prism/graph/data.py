"""Mini-batching of variable-size graphs for the GNN (NumPy -> torch).

Graphs are concatenated into one big disconnected graph per batch (the standard
PyTorch-Geometric trick): node indices of graph ``g`` are offset by the number of
nodes of the previous graphs and ``batch[i]`` tells which graph node ``i``
belongs to. Implemented here so that PyTorch Geometric is not required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

Arrays = Dict[str, np.ndarray]


@dataclass
class GraphBatch:
    """A batch of graphs as torch tensors."""

    x: "object"            # (N, F) float
    edge_index: "object"   # (2, E) long
    edge_attr: "object"    # (E, D) float
    batch: "object"        # (N,) long, graph id of each node
    ptr: "object"          # (G+1,) long, node offsets
    num_graphs: int

    def to(self, device: str) -> "GraphBatch":
        """Move every tensor to ``device``."""
        return GraphBatch(self.x.to(device), self.edge_index.to(device), self.edge_attr.to(device),
                          self.batch.to(device), self.ptr.to(device), self.num_graphs)


def collate(graphs: Sequence[Arrays]) -> GraphBatch:
    """Concatenate a list of per-graph arrays into one :class:`GraphBatch`."""
    import torch

    xs, eis, eas, batch, ptr = [], [], [], [], [0]
    offset = 0
    for g_id, g in enumerate(graphs):
        n = int(g["x"].shape[0])
        xs.append(g["x"])
        eis.append(g["edge_index"] + offset)
        eas.append(g["edge_attr"])
        batch.append(np.full(n, g_id, dtype=np.int64))
        offset += n
        ptr.append(offset)
    edge_dim = graphs[0]["edge_attr"].shape[1] if graphs else 0
    return GraphBatch(
        x=torch.from_numpy(np.concatenate(xs, axis=0).astype(np.float32)),
        edge_index=torch.from_numpy(np.concatenate(eis, axis=1).astype(np.int64)) if eis else torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.from_numpy(np.concatenate(eas, axis=0).astype(np.float32)) if eas else torch.zeros((0, edge_dim)),
        batch=torch.from_numpy(np.concatenate(batch).astype(np.int64)),
        ptr=torch.tensor(ptr, dtype=torch.long),
        num_graphs=len(graphs),
    )


def iterate_batches(graphs: Sequence[Arrays], batch_size: int, shuffle: bool = False,
                    rng: Optional[np.random.Generator] = None) -> Iterator[List[int]]:
    """Yield lists of graph indices forming each batch."""
    order = np.arange(len(graphs))
    if shuffle:
        (rng or np.random.default_rng()).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size].tolist()


def augment(g: Arrays, rng: np.random.Generator, node_drop: float, edge_drop: float, feature_noise: float,
            geom_slice: slice) -> Arrays:
    """Random view of a graph for contrastive learning.

    - node dropping (the FrontDoor node and at least one room are always kept):
      this simulates the partial queries of the partial-retrieval phase;
    - edge dropping;
    - Gaussian noise on the geometric node features only.
    """
    x, ei, ea = g["x"], g["edge_index"], g["edge_attr"]
    n = x.shape[0]
    keep = rng.random(n) >= node_drop
    door_col = geom_slice.start - 1  # the last one-hot slot is FrontDoor
    keep |= x[:, door_col] > 0.5
    if not keep.any() or (keep & (x[:, door_col] < 0.5)).sum() == 0:
        rooms = np.nonzero(x[:, door_col] < 0.5)[0]
        if len(rooms):
            keep[rng.choice(rooms)] = True
        else:
            keep[:] = True
    new_index = -np.ones(n, dtype=np.int64)
    new_index[keep] = np.arange(int(keep.sum()))
    x2 = x[keep].copy()
    if ei.shape[1]:
        mask = keep[ei[0]] & keep[ei[1]] & (rng.random(ei.shape[1]) >= edge_drop)
        ei2 = new_index[ei[:, mask]]
        ea2 = ea[mask]
    else:
        ei2, ea2 = ei, ea
    if feature_noise > 0:
        x2[:, geom_slice] += rng.normal(0.0, feature_noise, size=x2[:, geom_slice].shape).astype(np.float32)
    return {"x": x2, "edge_index": ei2, "edge_attr": ea2}
