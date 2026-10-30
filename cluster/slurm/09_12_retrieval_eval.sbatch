#!/bin/bash
#SBATCH --job-name=prism_09_12
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Retrieval (complete + partial queries), evaluation and comparison.
# The GPU is used to encode partial queries with the GNN / vision encoders.
#   sbatch cluster/slurm/09_12_retrieval_eval.sbatch
#   sbatch --export=ALL,EXTRA="--set partial.strategy=keep_connected" cluster/slurm/09_12_retrieval_eval.sbatch
source "$(dirname "$0")/common.sh" 2>/dev/null || source cluster/slurm/common.sh
stage_packed

python scripts/run_all.py --from 9 --to 12 ${PRISM_ARGS} --device cuda
