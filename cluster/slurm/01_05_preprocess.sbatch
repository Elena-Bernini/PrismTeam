#!/bin/bash
#SBATCH --job-name=prism_01_05
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# CPU-only: inspection, JSON conversion, graphs, renders, statistics/splits (+ packing).
#   sbatch cluster/slurm/01_05_preprocess.sbatch                                   # full
#   sbatch --qos=all_qos_dbg --time=00:30:00 --export=ALL,MODE=preview cluster/slurm/01_05_preprocess.sbatch
source "$(dirname "$0")/common.sh" 2>/dev/null || source cluster/slurm/common.sh

python scripts/run_all.py --from 1 --to 5 --pack ${PRISM_ARGS}
