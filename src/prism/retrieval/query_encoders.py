"""Encode NEW queries (e.g. partial layouts) consistently with a built index.

An index built in phase 07/08 stores which encoder produced it in
``meta.json``. :func:`make_query_encoder` rebuilds that encoder so that a query
(full or partial layout + graph) lands in the same vector space.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.prism.common.workspace import Workspace
from src.prism.datasets.schema import Taxonomy
from src.prism.graph.descriptor import graph_descriptor
from src.prism.vision.render import render_layout


class DescriptorQueryEncoder:
    """Hand-crafted graph descriptor."""

    def __init__(self, taxonomy: Taxonomy) -> None:
        """Keep the taxonomy needed by the descriptor."""
        self.taxonomy = taxonomy

    def encode(self, layouts: Sequence[Dict[str, Any]], graphs: Sequence[Dict[str, Any]],
               visible: Sequence[Optional[Sequence[int]]]) -> np.ndarray:
        """Descriptor of each (possibly partial) graph; the index applies its own scaler."""
        return np.stack([graph_descriptor(g, l, self.taxonomy)[0] for l, g in zip(layouts, graphs)])


class GnnQueryEncoder:
    """Trained GNN from phase 06."""

    def __init__(self, ws: Workspace, taxonomy: Taxonomy, device: str) -> None:
        """Load the checkpoint of this workspace."""
        from src.prism.graph.gnn_encoder import GnnEncoder, checkpoint_path

        self.encoder = GnnEncoder(checkpoint_path(ws.gnn_dir), taxonomy, device)

    def encode(self, layouts, graphs, visible) -> np.ndarray:
        """GNN embedding of each (possibly partial) graph."""
        return self.encoder.encode(list(graphs))


class VisionQueryEncoder:
    """Render the (partial) layout and embed it with the index's vision encoder."""

    def __init__(self, encoder_name: str, cfg: Dict[str, Any], taxonomy: Taxonomy, device: str,
                 image_source: str) -> None:
        """Build the vision encoder; warn if the index was not built on renders."""
        from src.prism.vision.encoders import build_encoder

        self.encoder = build_encoder(encoder_name, cfg["vision"], device)
        self.taxonomy = taxonomy
        self.rcfg = cfg["render"]
        if image_source != "render":
            print(f"  [warn] index built on '{image_source}' images: partial queries are rendered "
                  "with the semantic renderer, so query and gallery images differ in style.")

    def encode(self, layouts, graphs, visible) -> np.ndarray:
        """Render each layout (only visible rooms coloured) and embed the images."""
        images = [render_layout(l, self.taxonomy, self.rcfg, visible_room_ids=v) for l, v in zip(layouts, visible)]
        return self.encoder.encode(images)


def make_query_encoder(meta: Dict[str, Any], ws: Workspace, cfg: Dict[str, Any], taxonomy: Taxonomy,
                       device: str) -> Any:
    """Return the query encoder matching an index ``meta.json``."""
    pipeline, encoder = meta.get("pipeline"), meta.get("encoder")
    if pipeline == "graph" and encoder == "descriptor":
        return DescriptorQueryEncoder(taxonomy)
    if pipeline == "graph" and encoder == "gnn":
        return GnnQueryEncoder(ws, taxonomy, device)
    if pipeline == "vision":
        return VisionQueryEncoder(encoder, cfg, taxonomy, device, meta.get("image_source", "render"))
    raise ValueError(f"Cannot build a query encoder for index meta {meta}")
