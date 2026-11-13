#!/bin/bash
#SBATCH --job-name=prism_13
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Constraint search (phase 13): k most plausible plans for a request. CPU only, no GPU.
#   sbatch cluster/slurm/13_constraint_search.sbatch
#   sbatch --export=ALL,SPEC=configs/specs/example_family.yaml cluster/slurm/13_constraint_search.sbatch
source "$(dirname "$0")/common.sh" 2>/dev/null || source cluster/slurm/common.sh

python scripts/13_constraint_search.py ${PRISM_ARGS} --spec ${SPEC:-configs/specs/example_2bed.yaml}
