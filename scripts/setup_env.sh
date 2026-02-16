#!/usr/bin/env bash
# DreamZero environment setup for H100 with CUDA 12.9 and PyTorch 2.8.
# Run from repo root: bash scripts/setup_env.sh
# Ensure conda is on PATH (e.g. run: conda init bash && source ~/.bashrc)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda &>/dev/null; then
  echo "Conda not found. Initialize it first, e.g.: conda init bash && source ~/.bashrc"
  echo "Or install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

echo "[1/4] Creating conda environment dreamzero (Python 3.11)..."
conda create -n dreamzero python=3.11 -y

echo "[2/4] Installing DreamZero and PyTorch 2.8 with CUDA 12.9..."
# Activate in subshell so we can run pip
eval "$(conda shell.bash hook)"
conda activate dreamzero
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129

echo "[3/4] Installing flash-attn (build from source; may take 10-30 min)..."
MAX_JOBS=8 pip install --no-build-isolation flash-attn

echo "[4/4] Skipping Transformer Engine (H100; only needed for GB200)."
# For GB200 only: pip install --no-build-isolation transformer_engine[pytorch]

echo ""
echo "Setup complete. Activate with: conda activate dreamzero"
echo "Then run: bash scripts/download_weights.sh  # optional, to pre-download model weights"
