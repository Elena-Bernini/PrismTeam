# Vision-Based Retrieval

## Purpose

The vision retrieval branch is the first baseline. It retrieves floor plans based on raster image similarity.

## Input

```text
RPLAN floor plan raster image
```

The input may be:

- a complete floor plan;
- a masked floor plan;
- a partial sketch-like floor plan generated from annotations.

## Encoders

Initial encoders to test:

```text
DINOv2
CLIP visual encoder
ViT
ResNet50
```

The first implementation should use pretrained encoders without fine-tuning. This creates a fast baseline and validates the retrieval infrastructure.

## Pipeline

```text
floorplan image
    -> resize and normalize
    -> visual encoder
    -> embedding vector
    -> L2 normalization
    -> FAISS index
    -> top-k retrieval
```

## Embedding storage

Store embeddings in:

```text
data/embeddings/vision/<encoder_name>_embeddings.npy
```

Store metadata in:

```text
data/embeddings/vision/<encoder_name>_metadata.json
```

Metadata should map embedding row indices to sample IDs.

## FAISS index

Store index in:

```text
data/indexes/vision_<encoder_name>.faiss
```

The initial index can use cosine similarity through normalized vectors and inner product search.

## Retrieval script

Expected command:

```bash
python scripts/retrieve_vision.py \
  --query_id 000123 \
  --encoder dinov2 \
  --top_k 10 \
  --output outputs/vision_top10/000123
```

Expected outputs:

```text
results.json
scores.csv
top10_grid.png
query.png
```

## Vision retrieval evaluation

Evaluate on:

1. complete query images;
2. raster-masked query images;
3. room-removed query images.

Metrics:

```text
Recall@1
Recall@5
Recall@10
MRR
nDCG
room type similarity of retrieved plans
adjacency compatibility of retrieved plans
```

## Optional fine-tuning

After the baseline works, the vision encoder can be fine-tuned with contrastive learning.

Positive pairs:

```text
complete floor plan image
masked version of the same floor plan
```

Negative pairs:

```text
floor plans from different samples
```

Loss options:

```text
InfoNCE
Triplet loss
Supervised contrastive loss
```
