#!/bin/bash
#SBATCH --job-name=prism_06_gnn
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# GNN training on one GPU.
#   sbatch cluster/slurm/06_train_gnn.sbatch
#   sbatch --export=ALL,EXTRA="--set graph.model=gat --set graph.objective=both" cluster/slurm/06_train_gnn.sbatch
#   sbatch --qos=all_qos_dbg --time=00:30:00 --export=ALL,MODE=preview cluster/slurm/06_train_gnn.sbatch
source "$(dirname "$0")/common.sh" 2>/dev/null || source cluster/slurm/common.sh
stage_packed

python scripts/06_train_gnn.py ${PRISM_ARGS} --device cuda
