"""Self-supervised training of the GNN encoder.

Objectives (``graph.objective``):

- ``gae``          link prediction (Elena's graph auto-encoder): node embeddings
                   of connected rooms should have a high dot product, random
                   room pairs of the same floor plan a low one.
- ``contrastive``  GraphCL-style NT-Xent: two random views of the same plan
                   (node drop, edge drop, feature noise) should map close
                   together and far from other plans. Node dropping mimics the
                   partial queries of phase 10.
- ``both``         sum of the two losses.

Training uses the train split; the validation split drives early stopping.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.prism.graph.data import Arrays, augment, collate, iterate_batches
from src.prism.graph.models import GraphEncoder


def link_prediction_loss(h: torch.Tensor, batch: Any, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Binary cross-entropy on positive edges vs. negatives sampled inside the same graph.

    For every positive edge ``(u, v)`` a negative ``(u, w)`` is drawn with ``w``
    uniform among the nodes of the same floor plan (occasional collisions with a
    real edge are tolerated, as in standard negative sampling).
    """
    ei = batch.edge_index
    if ei.size(1) == 0:
        return h.sum() * 0.0
    pos = (h[ei[0]] * h[ei[1]]).sum(-1)
    graph_of_u = batch.batch[ei[0]]
    start = batch.ptr[graph_of_u]
    count = (batch.ptr[graph_of_u + 1] - start).clamp(min=1)
    if generator is not None:
        # CPU generator (portable across CUDA / MPS), then moved to the model device.
        rand = torch.rand(ei.size(1), generator=generator).to(h.device)
    else:
        rand = torch.rand(ei.size(1), device=h.device)
    w = start + torch.minimum((rand * count.to(rand.dtype)).long(), count - 1)
    neg = (h[ei[0]] * h[w]).sum(-1)
    return (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalised temperature-scaled cross-entropy (SimCLR / GraphCL)."""
    b = z1.size(0)
    if b < 2:
        return z1.sum() * 0.0
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(2 * b, dtype=torch.bool, device=z.device), float("-inf"))
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)


def batch_loss(model: GraphEncoder, graphs: Sequence[Arrays], idx: List[int], cfg: Dict[str, Any],
               device: str, rng: np.random.Generator, geom_slice: slice,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Compute the configured objective on one batch of graph indices.

    ``generator`` (CPU) makes negative sampling reproducible during validation
    without touching the global torch RNG used by training.
    """
    gcfg = cfg["graph"]
    objective = gcfg["objective"]
    loss = torch.zeros((), device=device)
    if objective in {"gae", "both"}:
        batch = collate([graphs[i] for i in idx]).to(device)
        h = model.node_embeddings(batch)
        loss = loss + link_prediction_loss(h, batch, generator)
    if objective in {"contrastive", "both"}:
        c = gcfg["contrastive"]
        views = []
        for _ in range(2):
            aug = [augment(graphs[i], rng, c["node_drop"], c["edge_drop"], c["feature_noise"], geom_slice) for i in idx]
            b = collate(aug).to(device)
            h = model.node_embeddings(b)
            views.append(model.proj_head(model.readout(model.pool(h, b))))
        loss = loss + nt_xent(views[0], views[1], float(c["temperature"]))
    return loss


def evaluate_loss(model: GraphEncoder, graphs: Sequence[Arrays], cfg: Dict[str, Any], device: str,
                  geom_slice: slice, seed: int = 0) -> float:
    """Mean validation loss with a fixed RNG (so epochs are comparable)."""
    if not graphs:
        return float("nan")
    model.eval()
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    total, count = 0.0, 0
    with torch.no_grad():
        for idx in iterate_batches(graphs, int(cfg["graph"]["batch_size"])):
            total += float(batch_loss(model, graphs, idx, cfg, device, rng, geom_slice, generator)) * len(idx)
            count += len(idx)
    return total / max(1, count)


def train_gnn(model: GraphEncoder, train_graphs: Sequence[Arrays], val_graphs: Sequence[Arrays],
              cfg: Dict[str, Any], device: str, epochs: int, geom_slice: slice,
              log: Callable[[str], None] = print) -> Tuple[GraphEncoder, List[Dict[str, float]]]:
    """Train ``model`` and return the best (lowest validation loss) weights and the history."""
    gcfg = cfg["graph"]
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(gcfg["lr"]), weight_decay=float(gcfg["weight_decay"]))
    rng = np.random.default_rng(int(cfg["runtime"]["seed"]))
    best_state, best_val, bad_epochs = copy.deepcopy(model.state_dict()), math.inf, 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, count = time.perf_counter(), 0.0, 0
        for idx in iterate_batches(train_graphs, int(gcfg["batch_size"]), shuffle=True, rng=rng):
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(model, train_graphs, idx, cfg, device, rng, geom_slice)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        train_loss = total / max(1, count)
        val_loss = evaluate_loss(model, val_graphs, cfg, device, geom_slice)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "seconds": time.perf_counter() - t0})
        log(f"  epoch {epoch:3d}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"({history[-1]['seconds']:.1f}s)")
        monitored = val_loss if not math.isnan(val_loss) else train_loss
        if monitored < best_val - 1e-4:
            best_val, bad_epochs = monitored, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= int(gcfg["patience"]):
                log(f"  early stopping at epoch {epoch} (best val {best_val:.4f})")
                break
    model.load_state_dict(best_state)
    return model, history
