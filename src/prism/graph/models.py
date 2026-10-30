"""GNN encoders in plain PyTorch (GCN, GraphSAGE, GAT), with edge features.

Only ``index_add_`` / ``scatter_reduce_`` are used for message passing, which are
available on CPU, CUDA and Apple MPS. No PyTorch Geometric extension is needed.

Architecture::

    node features -> Linear -> [conv -> LayerNorm -> ReLU -> Dropout, + residual] x L
                  -> node embeddings h (hidden_dim)
                  -> readout = concat(mean_pool(h), max_pool(h))      (2 * hidden_dim)
                  -> graph embedding:
                       objective "gae":         the readout itself
                       "contrastive"/"both":    Linear(readout) -> embed_dim
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Scatter helpers
# ----------------------------------------------------------------------------
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum rows of ``src`` into ``dim_size`` buckets given by ``index``."""
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    if src.numel():
        out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Average rows of ``src`` per bucket (empty buckets stay 0)."""
    total = scatter_sum(src, index, dim_size)
    count = scatter_sum(torch.ones_like(index, dtype=src.dtype).unsqueeze(-1), index, dim_size).clamp(min=1.0)
    return total / count.view((dim_size,) + (1,) * (src.dim() - 1))


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Element-wise max of rows of ``src`` per bucket (empty buckets are 0)."""
    out = src.new_full((dim_size,) + tuple(src.shape[1:]), float("-inf"))
    if src.numel():
        idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        out = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Softmax of ``scores`` (E, H) over the edges that share the same target node."""
    smax = scatter_max(scores, index, dim_size)
    exp = torch.exp(scores - smax[index])
    denom = scatter_sum(exp, index, dim_size)[index].clamp(min=1e-16)
    return exp / denom


# ----------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------
class GCNLayer(nn.Module):
    """Kipf & Welling GCN with symmetric normalisation, self loops and additive edge messages."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0) -> None:
        """Create the node and (optional) edge projections."""
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.edge_lin = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        """Aggregate normalised neighbour messages: ``sum_j norm_ij (W x_j + W_e e_ij)``."""
        n = x.size(0)
        loops = torch.arange(n, device=x.device)
        src = torch.cat([edge_index[0], loops])
        dst = torch.cat([edge_index[1], loops])
        deg = scatter_sum(torch.ones_like(dst, dtype=x.dtype).unsqueeze(-1), dst, n).squeeze(-1)
        norm = deg[src].clamp(min=1).rsqrt() * deg[dst].clamp(min=1).rsqrt()
        h = self.lin(x)
        msg = h[src]
        if self.edge_lin is not None and edge_attr is not None and edge_attr.numel():
            e = self.edge_lin(edge_attr)
            msg = msg + torch.cat([e, e.new_zeros((n, e.size(1)))], dim=0)
        return scatter_sum(msg * norm.unsqueeze(-1), dst, n)


class SAGELayer(nn.Module):
    """GraphSAGE (mean aggregator) with edge features added to neighbour messages."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0) -> None:
        """Create the self, neighbour and (optional) edge projections."""
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_lin = nn.Linear(edge_dim, in_dim, bias=False) if edge_dim else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        """``W_s x_i + W_n mean_j (x_j + W_e e_ij)``."""
        src, dst = edge_index[0], edge_index[1]
        msg = x[src]
        if self.edge_lin is not None and edge_attr is not None and edge_attr.numel():
            msg = msg + self.edge_lin(edge_attr)
        agg = scatter_mean(msg, dst, x.size(0))
        return self.lin_self(x) + self.lin_neigh(agg)


class GATLayer(nn.Module):
    """Multi-head graph attention (Velickovic et al.) with edge features in the attention score."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0, heads: int = 4) -> None:
        """Create per-head projections; ``out_dim`` must be divisible by ``heads``."""
        super().__init__()
        if out_dim % heads:
            raise ValueError(f"GAT out_dim ({out_dim}) must be divisible by heads ({heads})")
        self.heads, self.ch = heads, out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.ch))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.ch))
        self.edge_lin = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim else None
        self.att_edge = nn.Parameter(torch.empty(1, heads, self.ch)) if edge_dim else None
        self.bias = nn.Parameter(torch.zeros(out_dim))
        for p in (self.att_src, self.att_dst, self.att_edge):
            if p is not None:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        """Attention-weighted sum of neighbour messages (self loops included)."""
        n = x.size(0)
        loops = torch.arange(n, device=x.device)
        src = torch.cat([edge_index[0], loops])
        dst = torch.cat([edge_index[1], loops])
        h = self.lin(x).view(n, self.heads, self.ch)
        score = (h[src] * self.att_src).sum(-1) + (h[dst] * self.att_dst).sum(-1)
        msg = h[src]
        if self.edge_lin is not None and edge_attr is not None and edge_attr.numel():
            e = self.edge_lin(edge_attr).view(-1, self.heads, self.ch)
            e = torch.cat([e, e.new_zeros((n, self.heads, self.ch))], dim=0)
            score = score + (e * self.att_edge).sum(-1)
            msg = msg + e
        alpha = segment_softmax(F.leaky_relu(score, 0.2), dst, n)
        out = scatter_sum(msg * alpha.unsqueeze(-1), dst, n)
        return out.reshape(n, self.heads * self.ch) + self.bias


LAYERS = {"gcn": GCNLayer, "sage": SAGELayer, "gat": GATLayer}


# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------
class GraphEncoder(nn.Module):
    """Stack of message-passing layers plus mean/max readout."""

    def __init__(self, node_dim: int, edge_dim: int, model: str = "sage", hidden_dim: int = 64,
                 embed_dim: int = 128, num_layers: int = 3, heads: int = 4, dropout: float = 0.1,
                 use_edge_attr: bool = True, objective: str = "gae") -> None:
        """Build the encoder; arguments mirror the ``graph`` section of the config."""
        super().__init__()
        if model not in LAYERS:
            raise ValueError(f"Unknown GNN model '{model}'. Choose from {sorted(LAYERS)}")
        self.objective = objective
        self.use_edge_attr = use_edge_attr
        ed = edge_dim if use_edge_attr else 0
        self.input = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if model == "gat":
                self.convs.append(GATLayer(hidden_dim, hidden_dim, ed, heads))
            else:
                self.convs.append(LAYERS[model](hidden_dim, hidden_dim, ed))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(2 * hidden_dim, embed_dim)
        self.proj_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

    def node_embeddings(self, batch: Any) -> torch.Tensor:
        """Run message passing and return per-node embeddings ``(N, hidden_dim)``."""
        ea = batch.edge_attr if self.use_edge_attr else None
        h = F.relu(self.input(batch.x))
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.dropout(F.relu(norm(conv(h, batch.edge_index, ea))))
        return h

    def pool(self, h: torch.Tensor, batch: Any) -> torch.Tensor:
        """Concatenate mean- and max-pooled node embeddings per graph."""
        return torch.cat([scatter_mean(h, batch.batch, batch.num_graphs),
                          scatter_max(h, batch.batch, batch.num_graphs)], dim=-1)

    def graph_embedding(self, h: torch.Tensor, batch: Any) -> torch.Tensor:
        """Graph-level embedding used for retrieval (see module docstring)."""
        pooled = self.pool(h, batch)
        return pooled if self.objective == "gae" else self.readout(pooled)

    def forward(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(node_embeddings, graph_embeddings)``."""
        h = self.node_embeddings(batch)
        return h, self.graph_embedding(h, batch)


def model_from_config(node_dim: int, edge_dim: int, gcfg: Dict[str, Any]) -> GraphEncoder:
    """Instantiate :class:`GraphEncoder` from the ``graph`` config section."""
    return GraphEncoder(node_dim=node_dim, edge_dim=edge_dim, model=gcfg["model"],
                        hidden_dim=int(gcfg["hidden_dim"]), embed_dim=int(gcfg["embed_dim"]),
                        num_layers=int(gcfg["num_layers"]), heads=int(gcfg["heads"]),
                        dropout=float(gcfg["dropout"]), use_edge_attr=bool(gcfg["use_edge_attr"]),
                        objective=gcfg["objective"])
