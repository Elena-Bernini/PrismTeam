#!/bin/bash
#SBATCH --job-name=prism_retrieval_dbg
#SBATCH --partition=all_usr_prod
#SBATCH --qos=all_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=18G
#SBATCH --time=01:00:00
#SBATCH --output=/work/cvcs2026/PrismTeam/logs/prism_retrieval_%j.out
#SBATCH --error=/work/cvcs2026/PrismTeam/logs/prism_retrieval_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail
module load py-torch/2.8.0-gcc-11.4.0-cuda-12.6.3
source /homes/psignorino/envs/prism_env/bin/activate
cd /work/cvcs2026/PrismTeam/prism-cluster
python src/run_stage.py --stage 02_retrieval_pipelines --config config/02_retrieval_resnet50.yaml "$@"
