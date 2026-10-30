#!/bin/bash
# =============================================================================
# Sourced at the beginning of every Prism SLURM job.
#   - loads the Python module and activates the venv
#   - sets MODE (preview | full, default full) and PRISM_ARGS
#   - stage_packed: copies the packed dataset to node-local /tmp (fast I/O)
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

module load "${PRISM_PYTHON_MODULE}"
source "${PRISM_VENV}/bin/activate"
cd "${PRISM_CODE_DIR}"

export MODE="${MODE:-full}"
export DATASET="${DATASET:-rplan}"
export PYTHONUNBUFFERED=1
# Extra options forwarded to every phase, e.g. sbatch --export=ALL,EXTRA="--set graph.model=gat"
export EXTRA="${EXTRA:-}"
PRISM_ARGS="--env cluster --mode ${MODE} --dataset ${DATASET} ${EXTRA}"

echo "job ${SLURM_JOB_ID:-local} on $(hostname) | mode=${MODE} | gpus=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Copy packed files (phase 05 --pack) + splits/metadata to /tmp and point the
# code at them. Only for read-only phases (06+). /tmp is wiped when the job ends.
stage_packed() {
  local src="${PRISM_WORK_DIR}/data/processed/${DATASET}/${MODE}"
  if [ -f "${src}/packed/layouts.jsonl.gz" ]; then
    local dst="/tmp/prism_${SLURM_JOB_ID:-$$}"
    mkdir -p "${dst}"
    cp -r "${src}/packed" "${src}/splits" "${src}/metadata" "${dst}/"
    PRISM_ARGS="${PRISM_ARGS} --set paths.processed_override=${dst}"
    echo "staged packed data to ${dst} ($(du -sh "${dst}" | cut -f1))"
  else
    echo "no packed data found in ${src}/packed (run phase 05 with --pack); reading from /work"
  fi
}
