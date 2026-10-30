#!/bin/bash
#SBATCH --job-name=prism_07_08
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Graph and vision embeddings + indexes.
#   sbatch cluster/slurm/07_08_index.sbatch
#   sbatch --export=ALL,VISION="dinov2_base resnet50 clip_b32" cluster/slurm/07_08_index.sbatch
source "$(dirname "$0")/common.sh" 2>/dev/null || source cluster/slurm/common.sh
stage_packed
VISION="${VISION:-dinov2_base colorhist}"

python scripts/07_graph_embed_index.py ${PRISM_ARGS} --device cuda --encoders descriptor gnn
python scripts/08_vision_embed_index.py ${PRISM_ARGS} --device cuda --encoders ${VISION}
