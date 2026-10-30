#!/bin/bash
# =============================================================================
# One-time environment setup on the AImageLab-HPC login node.
#
#   bash cluster/setup_env.sh
#
# Creates the venv in /homes/$USER/envs/prism with the CUDA build of PyTorch,
# installs the project in editable mode and pre-downloads the vision models
# (compute nodes may not have internet access).
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/config.sh"

module load "${PRISM_PYTHON_MODULE}"
mkdir -p "$(dirname "${PRISM_VENV}")" "${HF_HOME}" "${TORCH_HOME}"

if [ ! -d "${PRISM_VENV}" ]; then
  python -m venv "${PRISM_VENV}"
fi
source "${PRISM_VENV}/bin/activate"
python -m pip install --upgrade pip

python -m pip install torch torchvision --index-url "${PRISM_TORCH_INDEX}"
python -m pip install -r "${PRISM_CODE_DIR}/requirements/base.txt" -r "${PRISM_CODE_DIR}/requirements/vision.txt"
python -m pip install -e "${PRISM_CODE_DIR}"

python "${PRISM_CODE_DIR}/cluster/predownload_models.py"
python "${PRISM_CODE_DIR}/scripts/00_check_env.py" --env cluster

echo "Environment ready: ${PRISM_VENV}"
