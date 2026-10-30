#!/bin/bash
# =============================================================================
# Prism - cluster settings (AImageLab-HPC, UNIMORE). EDIT THIS FILE ONCE.
# Sourced by setup_env.sh and by every job in cluster/slurm/.
# =============================================================================

# Where the code lives on the cluster (clone/unzip the project here).
export PRISM_CODE_DIR="/work/cvcs2026/PrismTeam/prism"

# Shared data/artifacts/outputs (must match configs/env/cluster.yaml).
export PRISM_WORK_DIR="/work/cvcs2026/PrismTeam"

# Python virtual environment: the docs recommend /homes (NFS) for venvs, not /work.
export PRISM_VENV="/homes/${USER}/envs/prism"

# Python module used to create AND run the venv (must be the same).
export PRISM_PYTHON_MODULE="python/3.11.11-gcc-11.4.0"

# CUDA build of PyTorch to install (cu124 works on L40S / A40 / RTX nodes).
export PRISM_TORCH_INDEX="https://download.pytorch.org/whl/cu124"

# Model caches on /work (the /homes quota is 100 GB).
export HF_HOME="${PRISM_WORK_DIR}/cache/huggingface"
export TORCH_HOME="${PRISM_WORK_DIR}/cache/torch"

export PRISM_ENV="cluster"
