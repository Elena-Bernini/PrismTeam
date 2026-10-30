"""Image encoders for the vision pipeline.

Registry (``vision.encoders`` in the config):

==================  ==========================================  ================
name                model                                        needs
==================  ==========================================  ================
``colorhist``       spatial RGB histogram (Pier's CPU baseline)  NumPy only
``resnet18/50``     torchvision ImageNet ResNet, global pool     torch, torchvision
``dinov2_small``    facebook/dinov2-small (CLS or mean token)    torch, transformers
``dinov2_base``     facebook/dinov2-base   (Elena's choice)      torch, transformers
``dinov2_large``    facebook/dinov2-large                        torch, transformers
``clip_b32``        openai/clip-vit-base-patch32 image tower     torch, transformers
``timm:<name>``     any timm model with ``num_classes=0``        torch, timm
==================  ==========================================  ================

All encoders expose ``encode(list_of_PIL_images) -> np.ndarray (N, D)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageOps

HF_MODELS = {
    "dinov2_small": "facebook/dinov2-small",
    "dinov2_base": "facebook/dinov2-base",
    "dinov2_large": "facebook/dinov2-large",
    "clip_b32": "openai/clip-vit-base-patch32",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ColorHistEncoder:
    """Spatial colour-histogram descriptor (CPU only, no deep learning).

    Concatenates a global RGB histogram, per-cell RGB histograms over a grid and
    a foreground-occupancy grid, so that WHERE colours are matters, not only
    how much of each colour there is.
    """

    name = "colorhist"

    def __init__(self, image_size: int = 128, bins: int = 8, grid: int = 8) -> None:
        """Store the resize size, the number of histogram bins and the grid size."""
        self.size, self.bins, self.grid = image_size, bins, grid

    def _hist(self, arr: np.ndarray) -> List[np.ndarray]:
        """Normalised per-channel histograms of an image patch."""
        out = []
        for c in range(3):
            h, _ = np.histogram(arr[:, :, c], bins=self.bins, range=(0.0, 1.0))
            out.append(h.astype(np.float32) / max(float(h.sum()), 1.0))
        return out

    def encode_one(self, img: Image.Image) -> np.ndarray:
        """Descriptor of a single image."""
        img = ImageOps.pad(img.convert("RGB"), (self.size, self.size), color=(255, 255, 255))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        parts = self._hist(arr)
        cell = self.size // self.grid
        occ = []
        for gy in range(self.grid):
            for gx in range(self.grid):
                patch = arr[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell]
                parts.extend(self._hist(patch))
                occ.append(float((patch.mean(axis=2) < 0.97).mean()))
        parts.append(np.asarray(occ, dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Descriptors of a list of images."""
        return np.stack([self.encode_one(im) for im in images]).astype(np.float32)


class TorchEncoder:
    """Common batching logic for torch-based encoders."""

    name = "torch"

    def __init__(self, device: str, batch_size: int) -> None:
        """Remember the device and the batch size."""
        self.device, self.batch_size = device, batch_size

    def preprocess(self, images: Sequence[Image.Image]) -> Any:
        """Turn PIL images into a model input batch (implemented by subclasses)."""
        raise NotImplementedError

    def forward(self, batch: Any) -> Any:
        """Run the model and return a ``(B, D)`` tensor (implemented by subclasses)."""
        raise NotImplementedError

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Embed images in mini-batches without gradients."""
        import torch

        out = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                chunk = [im.convert("RGB") for im in images[start:start + self.batch_size]]
                feats = self.forward(self.preprocess(chunk))
                out.append(feats.detach().float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


class TorchvisionEncoder(TorchEncoder):
    """ResNet-18/50 from torchvision with the classification head removed."""

    def __init__(self, name: str, device: str, batch_size: int, image_size: int, pretrained: bool) -> None:
        """Load the backbone and its preprocessing transform."""
        super().__init__(device, batch_size)
        import torch.nn as nn
        from torchvision import models, transforms

        self.name = name
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
        else:
            raise ValueError(name)
        self.model.fc = nn.Identity()
        self.model.eval().to(device)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def preprocess(self, images):
        """Resize, normalise and stack the images."""
        import torch
        return torch.stack([self.transform(im) for im in images]).to(self.device)

    def forward(self, batch):
        """Global-average-pooled ResNet features."""
        return self.model(batch)


class HFEncoder(TorchEncoder):
    """Hugging Face vision transformers (DINOv2, CLIP image tower)."""

    def __init__(self, name: str, device: str, batch_size: int, pooling: str = "cls") -> None:
        """Download/load the model and its image processor (cached in ``HF_HOME``)."""
        super().__init__(device, batch_size)
        from transformers import AutoImageProcessor

        self.name = name
        self.pooling = pooling
        model_id = HF_MODELS[name]
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        if name.startswith("clip"):
            from transformers import CLIPVisionModelWithProjection
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_id)
            self.is_clip = True
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_id)
            self.is_clip = False
        self.model.eval().to(device)

    def preprocess(self, images):
        """Apply the model's own processor (resize, crop, normalise)."""
        return self.processor(images=list(images), return_tensors="pt").to(self.device)

    def forward(self, batch):
        """CLS token (or mean of patch tokens) for DINOv2; projected embedding for CLIP."""
        out = self.model(**batch)
        if self.is_clip:
            return out.image_embeds
        tokens = out.last_hidden_state
        return tokens[:, 0] if self.pooling == "cls" else tokens[:, 1:].mean(dim=1)


class TimmEncoder(TorchEncoder):
    """Any timm model, created with ``num_classes=0`` (pooled features)."""

    def __init__(self, name: str, device: str, batch_size: int, pretrained: bool) -> None:
        """Create the timm model and its evaluation transform."""
        super().__init__(device, batch_size)
        import timm
        from timm.data import create_transform, resolve_model_data_config

        self.name = name
        self.model = timm.create_model(name.split(":", 1)[1], pretrained=pretrained, num_classes=0)
        self.model.eval().to(device)
        self.transform = create_transform(**resolve_model_data_config(self.model), is_training=False)

    def preprocess(self, images):
        """Apply the timm transform and stack."""
        import torch
        return torch.stack([self.transform(im) for im in images]).to(self.device)

    def forward(self, batch):
        """Pooled timm features."""
        return self.model(batch)


def needs_torch(name: str) -> bool:
    """True when the encoder requires PyTorch."""
    return name != "colorhist"


def build_encoder(name: str, vcfg: Dict[str, Any], device: str) -> Any:
    """Instantiate an encoder by registry name."""
    pretrained = vcfg.get("weights", "pretrained") == "pretrained"
    bs, size = int(vcfg["batch_size"]), int(vcfg["image_size"])
    if name == "colorhist":
        return ColorHistEncoder()
    if name in {"resnet18", "resnet50"}:
        return TorchvisionEncoder(name, device, bs, size, pretrained)
    if name in HF_MODELS:
        return HFEncoder(name, device, bs, vcfg.get("pooling", "cls"))
    if name.startswith("timm:"):
        return TimmEncoder(name, device, bs, pretrained)
    raise ValueError(f"Unknown vision encoder '{name}'")


def index_name(encoder: str, source: str) -> str:
    """Index folder name for a vision encoder and image source, e.g. ``vision_dinov2_base_render``."""
    return f"vision_{encoder.replace(':', '-')}_{source}"
