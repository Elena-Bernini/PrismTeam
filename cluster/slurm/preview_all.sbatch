#!/bin/bash
#SBATCH --job-name=prism_preview
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --qos=all_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Whole pipeline (01-12) in preview mode on a small subset, inside the 1-hour debug QoS.
#   sbatch cluster/slurm/preview_all.sbatch
#   sbatch --export=ALL,EXTRA="--n 500" cluster/slurm/preview_all.sbatch
export MODE=preview
source "$(dirname "$0")/common.sh" 2>/dev/null || source cluster/slurm/common.sh

python scripts/run_all.py --from 1 --to 12 ${PRISM_ARGS} --device cuda \
    --set "vision.encoders=[colorhist,dinov2_base]"
