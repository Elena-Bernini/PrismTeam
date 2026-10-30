"""Load a trained GNN checkpoint and embed (full or partial) graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from src.prism.datasets.schema import Taxonomy
from src.prism.graph.features import feature_dims, graph_to_arrays


def checkpoint_path(gnn_dir: Path) -> Path:
    """Location of the best checkpoint written by phase 06."""
    return gnn_dir / "model.pt"


def save_checkpoint(path: Path, model: Any, model_config: Dict[str, Any], history: List[Dict[str, float]]) -> None:
    """Save weights, the configuration needed to rebuild the model and the training history."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_config": model_config, "history": history}, path)


class GnnEncoder:
    """Wrapper that turns graph dictionaries into L2-ready embedding vectors."""

    def __init__(self, ckpt: Path, taxonomy: Taxonomy, device: str = "cpu", batch_size: int = 256) -> None:
        """Rebuild the encoder from ``ckpt`` and move it to ``device``."""
        import torch

        from src.prism.graph.models import GraphEncoder

        if not ckpt.exists():
            raise FileNotFoundError(f"GNN checkpoint not found: {ckpt}. Run phase 06 first.")
        state = torch.load(ckpt, map_location="cpu")
        mc = state["model_config"]
        self.cfg = mc
        self.taxonomy = taxonomy
        self.device = device
        self.batch_size = batch_size
        node_dim, edge_dim = feature_dims(taxonomy, mc["type_vocab"])
        if node_dim != mc["node_dim"] or edge_dim != mc["edge_dim"]:
            raise ValueError("Feature dimensions changed since training "
                             f"({node_dim},{edge_dim}) vs ({mc['node_dim']},{mc['edge_dim']}). Retrain phase 06.")
        self.model = GraphEncoder(node_dim=node_dim, edge_dim=edge_dim, model=mc["model"],
                                  hidden_dim=mc["hidden_dim"], embed_dim=mc["embed_dim"],
                                  num_layers=mc["num_layers"], heads=mc["heads"], dropout=mc["dropout"],
                                  use_edge_attr=mc["use_edge_attr"], objective=mc["objective"])
        self.model.load_state_dict(state["state_dict"])
        self.model.to(device).eval()

    def arrays(self, graph: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Feature arrays of one graph, consistent with training."""
        return graph_to_arrays(graph, self.taxonomy, self.cfg["type_vocab"], self.cfg["include_door"])

    def encode(self, graphs: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Embed a list of graph dictionaries -> ``(len(graphs), dim)`` float32 matrix."""
        import torch

        from src.prism.graph.data import collate

        arrs = [self.arrays(g) for g in graphs]
        out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(arrs), self.batch_size):
                batch = collate(arrs[start:start + self.batch_size]).to(self.device)
                _, emb = self.model(batch)
                out.append(emb.detach().cpu().float().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)
