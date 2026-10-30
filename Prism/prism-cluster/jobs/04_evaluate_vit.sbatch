#!/bin/bash
#SBATCH --job-name=prism_eval_vit
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/cvcs2026/PrismTeam/logs/prism_eval_vit_%j.out
#SBATCH --error=/work/cvcs2026/PrismTeam/logs/prism_eval_vit_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail
module load python/3.11.11-gcc-11.4.0
source /homes/psignorino/envs/prism_env/bin/activate
cd /work/cvcs2026/PrismTeam/prism-cluster
python src/run_stage.py --stage 04_evaluate_retrieval --config config/04_evaluate_vit.yaml "$@"
