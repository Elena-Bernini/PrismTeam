#!/bin/bash
#SBATCH --job-name=prism_part_bed
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/cvcs2026/PrismTeam/logs/prism_part_bed_%j.out
#SBATCH --error=/work/cvcs2026/PrismTeam/logs/prism_part_bed_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail
module load python/3.11.11-gcc-11.4.0
source /homes/psignorino/envs/prism_env/bin/activate
cd /work/cvcs2026/PrismTeam/prism-cluster
python src/run_stage.py --stage 03_partial_layout_retrieval --config config/03_partial_remove_bedroom.yaml "$@"
