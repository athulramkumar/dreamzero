#!/usr/bin/env bash
# Download DreamZero and WAN model weights in parallel using HF token.
# Optionally sources ~/.bash_aliases for HF_TOKEN, HF_HOME, TRANSFORMERS_CACHE.
# Run from repo root: bash scripts/download_weights.sh [checkpoint_dir]

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load HF token and cache env (e.g. from .bash_aliases)
if [[ -f ~/.bash_aliases ]]; then
  echo "Sourcing ~/.bash_aliases for HF_TOKEN and cache settings..."
  # shellcheck source=/dev/null
  source ~/.bash_aliases
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Warning: HF_TOKEN not set. Set it or add to ~/.bash_aliases (e.g. export HF_TOKEN=...)"
  echo "Downloads may fail or be rate-limited."
fi

CHECKPOINT_DIR="${1:-$REPO_ROOT/checkpoints/DreamZero-DROID}"
mkdir -p "$CHECKPOINT_DIR"
export HF_HOME="${HF_HOME:-$REPO_ROOT/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME"

download_dreamzero() {
  echo "[DreamZero-DROID] Downloading to $CHECKPOINT_DIR ..."
  huggingface-cli download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir "$CHECKPOINT_DIR" --token "${HF_TOKEN:-}"
  echo "[DreamZero-DROID] Done."
}

download_wan() {
  echo "[Wan2.1-I2V-14B-480P] Downloading to HF cache (HF_HOME=$HF_HOME) ..."
  python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='Wan-AI/Wan2.1-I2V-14B-480P',
    token=os.environ.get('HF_TOKEN'),
)
print('[Wan2.1-I2V-14B-480P] Done.')
"
}

echo "Starting parallel downloads (DreamZero-DROID + Wan2.1-I2V-14B-480P)..."
download_dreamzero &
PID1=$!
download_wan &
PID2=$!
wait $PID1
wait $PID2
echo "All downloads finished. DreamZero checkpoint: $CHECKPOINT_DIR"
