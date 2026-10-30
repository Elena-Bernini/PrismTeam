"""Download the pretrained vision models into HF_HOME / TORCH_HOME (run on the LOGIN node).

Compute nodes may not have internet access, so weights must be in the cache
before submitting phase 08/10 jobs.

    python cluster/predownload_models.py                       # default set
    python cluster/predownload_models.py dinov2_base clip_b32  # only these
"""

import sys

DEFAULT = ["dinov2_small", "dinov2_base", "resnet50", "resnet18", "clip_b32"]
HF = {"dinov2_small": "facebook/dinov2-small", "dinov2_base": "facebook/dinov2-base",
      "dinov2_large": "facebook/dinov2-large", "clip_b32": "openai/clip-vit-base-patch32"}


def download(name: str) -> None:
    """Fetch one model (and its preprocessor) into the local cache."""
    if name in HF:
        from transformers import AutoImageProcessor, AutoModel
        AutoImageProcessor.from_pretrained(HF[name])
        AutoModel.from_pretrained(HF[name])
    elif name in {"resnet18", "resnet50"}:
        from torchvision import models
        getattr(models, name)(weights="DEFAULT")
    elif name.startswith("timm:"):
        import timm
        timm.create_model(name.split(":", 1)[1], pretrained=True)
    else:
        raise ValueError(name)
    print(f"  cached {name}")


def main() -> None:
    """Download every requested model, continuing on errors."""
    names = sys.argv[1:] or DEFAULT
    for n in names:
        try:
            download(n)
        except Exception as exc:  # report and keep going
            print(f"  [error] {n}: {exc}")


if __name__ == "__main__":
    main()
