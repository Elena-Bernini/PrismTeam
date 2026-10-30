"""Phase 06 - train the GNN graph encoder (self-supervised).

Model: GCN / GraphSAGE / GAT (``graph.model``) with node features (room type,
geometry, front-door features) and edge features (relation, connection,
offsets, shared wall). Objective: ``gae`` link prediction (default),
``contrastive`` (GraphCL with node dropping) or ``both``.

Runs on CUDA (Pier, cluster), Apple MPS (Elena) or CPU. Trains on the train
split, early-stops on the validation split.

    python scripts/06_train_gnn.py --mode preview
    python scripts/06_train_gnn.py --mode full --set graph.model=gat --set graph.objective=both
"""

import sys

import numpy as np

import scripts._bootstrap as _bootstrap  # noqa: F401

from prism.common.cli import Timer, build_parser, setup
from prism.common.device import describe_device, resolve_device, torch_available
from prism.common.io import progress, write_json
from prism.common.report import HtmlReport, finalize_report
from prism.common.viz import plt
from prism.datasets.schema import Taxonomy
from prism.datasets.store import LayoutStore, load_splits
from prism.graph.features import feature_dims, graph_to_arrays, node_type_vocab

PHASE = "06_train_gnn"


def pca_2d(x):
    """Project rows of ``x`` on their first two principal components (NumPy SVD)."""
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def main():
    """Train, save the checkpoint and write the report."""
    args = build_parser(__doc__).parse_args()
    cfg, ws = setup(args, PHASE)
    if not torch_available():
        print("PyTorch is not installed: phase 06 needs it (see README, 'Installation').")
        sys.exit(2)

    from prism.graph.gnn_encoder import checkpoint_path, save_checkpoint
    from prism.graph.models import model_from_config
    from prism.graph.train import train_gnn

    out = ws.phase_dir(PHASE)
    gcfg = cfg["graph"]
    taxonomy = Taxonomy(cfg, ws.dataset)
    store = LayoutStore(ws)
    splits = load_splits(ws)
    kind, include_door = gcfg["type_vocab"], bool(gcfg["include_door_node"])

    def arrays(ids):
        """Feature arrays of a list of graph ids."""
        return [graph_to_arrays(store.graph(s), taxonomy, kind, include_door) for s in progress(ids, "featurise")]

    train_g, val_g = arrays(splits["train"]), arrays(splits["val"])
    if not train_g:
        raise RuntimeError("Empty train split.")
    node_dim, edge_dim = feature_dims(taxonomy, kind)
    vocab_len = len(node_type_vocab(taxonomy, kind))
    geom_slice = slice(vocab_len, vocab_len + 5)

    device = resolve_device(cfg["runtime"]["device"])
    print(f"  device: {describe_device(device)}")
    model = model_from_config(node_dim, edge_dim, gcfg)
    n_params = sum(p.numel() for p in model.parameters())
    epochs = int(gcfg["epochs_preview"] if ws.mode == "preview" else gcfg["epochs"])
    with Timer("training") as t:
        model, history = train_gnn(model, train_g, val_g, cfg, device, epochs, geom_slice)

    model_config = {"node_dim": node_dim, "edge_dim": edge_dim, "type_vocab": kind, "include_door": include_door,
                    "model": gcfg["model"], "hidden_dim": int(gcfg["hidden_dim"]), "embed_dim": int(gcfg["embed_dim"]),
                    "num_layers": int(gcfg["num_layers"]), "heads": int(gcfg["heads"]),
                    "dropout": float(gcfg["dropout"]), "use_edge_attr": bool(gcfg["use_edge_attr"]),
                    "objective": gcfg["objective"]}
    save_checkpoint(checkpoint_path(ws.gnn_dir), model.cpu(), model_config, history)
    write_json(ws.gnn_dir / "history.json", history)
    write_json(ws.gnn_dir / "model_config.json", model_config)

    # Embedding sanity check on the validation (or train) graphs.
    from prism.graph.gnn_encoder import GnnEncoder
    enc = GnnEncoder(checkpoint_path(ws.gnn_dir), taxonomy, device)
    probe_ids = (splits["val"] or splits["train"])[:2000]
    emb = enc.encode([store.graph(s) for s in probe_ids])
    n_rooms = [len(store.layout(s)["rooms"]) for s in probe_ids]

    rep = HtmlReport("06 - GNN training", f"{ws.dataset} · {ws.mode}")
    rep.kpis({"model": gcfg["model"], "objective": gcfg["objective"], "device": describe_device(device),
              "parameters": n_params, "train graphs": len(train_g), "val graphs": len(val_g),
              "epochs run": len(history), "minutes": t.elapsed / 60, "embedding dim": int(emb.shape[1])})
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train")
    ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    rep.figure(fig, "Loss curves (best validation epoch is kept)")
    if len(emb) >= 3:
        p = pca_2d(emb)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sc = ax.scatter(p[:, 0], p[:, 1], c=n_rooms, cmap="viridis", s=10)
        fig.colorbar(sc, ax=ax, label="rooms")
        ax.set_title("PCA of graph embeddings (colour = number of rooms)", fontsize=9)
        rep.figure(fig, "If the encoder learned structure, plans with similar room counts should cluster.")
    rep.heading("Configuration")
    rep.json_block(model_config)
    rep.table(history)
    finalize_report(rep, out, ws.mode)


if __name__ == "__main__":
    main()
