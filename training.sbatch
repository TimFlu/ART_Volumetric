#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm/train_%j.out
#SBATCH --error=logs/slurm/train_%j.err

set -euo pipefail

# Optional: load conda module only if your cluster uses modules
# module load Anaconda3

ENV_NAME="${ENV_NAME:-ART_Volumetric}"
ENV_FILE="${ENV_FILE:-environment.yml}"
PY_ENTRY="${PY_ENTRY:-scripts/training.py}"

# Make sure conda is available (works if module provides it, or if conda is on PATH)
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Load your Anaconda/Miniconda module first." >&2
  exit 1
fi

# Create env if missing (idempotent-ish)
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env create -f "$ENV_FILE"
fi

# Reproducible run (no activation required)
conda run -n "$ENV_NAME" python -m scripts.training
