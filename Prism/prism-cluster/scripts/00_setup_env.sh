#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${1:-/work/cvcs2026/PrismTeam/prism-cluster}"
USER_NAME="${2:-psignorino}"
VENV_PATH="/homes/${USER_NAME}/envs/prism_env"

mkdir -p \
  "${PROJECT_ROOT}/config" \
  "${PROJECT_ROOT}/scripts" \
  "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/docs/pipeline" \
  "/work/cvcs2026/PrismTeam/data/raw/rplan/extracted" \
  "/work/cvcs2026/PrismTeam/data/processed/rplan" \
  "/work/cvcs2026/PrismTeam/outputs/dataset_audit" \
  "/work/cvcs2026/PrismTeam/outputs/retrieval" \
  "/work/cvcs2026/PrismTeam/outputs/partial_layout_retrieval" \
  "/work/cvcs2026/PrismTeam/outputs/evaluation_resnet50" \
  "/work/cvcs2026/PrismTeam/outputs/evaluation_dinov2" \
  "/work/cvcs2026/PrismTeam/outputs/evaluation_vit" \
  "/work/cvcs2026/PrismTeam/outputs/evaluation_comparison" \
  "/work/cvcs2026/PrismTeam/logs" \
  "/homes/${USER_NAME}/envs"

PYVER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
if [[ "${PYVER}" != 3.11* ]]; then
  echo "WARNING: recommended Python is 3.11.x. Detected: ${PYVER}" >&2
fi

cat <<EOF
Folder skeleton ready.
Recommended venv setup on AImageLab-HPC:
  module load python/3.11.11-gcc-11.4.0
  python -m venv ${VENV_PATH}
  source ${VENV_PATH}/bin/activate
  pip install -r ${PROJECT_ROOT}/requirements.txt

For GPU retrieval jobs, the job scripts load py-torch/2.8.0-gcc-11.4.0-cuda-12.6.3 before activating the venv.
EOF
