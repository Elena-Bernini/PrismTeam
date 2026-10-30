# 02 Retrieval pipelines

Builds and queries graph and vision FAISS pipelines. Vision backbones include `colorhist`, `resnet18`, `resnet50`, `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, and `timm:<model>`.

Reads one of the `config/02_retrieval_*.yaml` files. Safe smoke-test override: `--max-items 1000 --random-queries 3 --batch-size 8`.

This file replaces the obsolete documentation name `post_preprocessing_retrieval.py`.
