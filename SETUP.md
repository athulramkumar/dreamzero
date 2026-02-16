# DreamZero – Reproducible environment setup

This document describes how to reproduce the DreamZero environment on a new machine. Tested on **H100** with **CUDA 12.9** and **PyTorch 2.8**.

**Recreating on a different instance:** follow the **Full steps to recreate on a new instance** section below (Steps 1–9). It lists every command in order.

---

## Full steps to recreate on a new instance

Use this as a single ordered checklist. Run from a fresh instance (e.g. new VM/container) with conda available and at least 2 GPUs (H100 or similar).

### Step 1: Prerequisites check

- **Hardware**: At least 2 GPUs (required for distributed inference; single-GPU will OOM).
- **Conda** on PATH (e.g. `conda init bash` and `source ~/.bashrc` if needed).
- **Hugging Face token** for weight downloads (create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

```bash
nvidia-smi                    # expect CUDA 12.x or 13.x
conda --version               # must succeed
```

### Step 2: Clone the repository

```bash
git clone --recurse-submodules https://github.com/dreamzero0/dreamzero.git
cd dreamzero
```

### Step 3: Create conda environment and install dependencies

```bash
conda create -n dreamzero python=3.11 -y
conda activate dreamzero
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
```

### Step 4: Install Flash Attention (build from source)

```bash
MAX_JOBS=8 pip install --no-build-isolation flash-attn
```

Adjust `MAX_JOBS` if needed (e.g. fewer for low-memory machines). This may take 10–30 minutes.

**Skip Transformer Engine** on H100 (only needed for GB200).

### Step 5: Set Hugging Face environment variables

Set these before downloading weights and before running the server (e.g. in `~/.bash_aliases` or `~/.bashrc`, or export in the same shell):

```bash
export HF_TOKEN="hf_xxxxxxxx"                    # your token; do not commit
export HF_HOME="/workspace/hf_cache"             # or another path you prefer
export TRANSFORMERS_CACHE="$HF_HOME"
```

Create the cache dir so downloads and the server use it:

```bash
mkdir -p "$HF_HOME"
```

### Step 6: Download model weights

DreamZero needs two assets: the **DreamZero-DROID** checkpoint (policy) and the **Wan2.1** backbone (image encoder, VAE, diffusion). Both can be pre-downloaded so the first server run is fast.

**Option A – use the script (downloads both in parallel):**

```bash
conda activate dreamzero
source ~/.bash_aliases   # or wherever you set HF_TOKEN and HF_HOME
bash scripts/download_weights.sh
```

Default checkpoint path: `./checkpoints/DreamZero-DROID`. To use another path:

```bash
bash scripts/download_weights.sh /path/to/checkpoint_dir
```

**Option B – manual downloads:**

```bash
conda activate dreamzero
# 1) DreamZero-DROID checkpoint (~50GB)
huggingface-cli download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir ./checkpoints/DreamZero-DROID --token "$HF_TOKEN"

# 2) Wan2.1 backbone (~77GB) into HF cache
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P', token='$HF_TOKEN')"
```

Ensure `HF_HOME` is set so Wan2.1 lands in your chosen cache (e.g. `/workspace/hf_cache`).

### Step 7: Verify installation

```bash
conda activate dreamzero
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"
python -c "import flash_attn; print('flash_attn OK')"
ls checkpoints/DreamZero-DROID/model.safetensors.index.json   # must exist
```

### Step 8: Run the inference server (requires 2+ GPUs)

In the same environment, set `HF_HOME`/`TRANSFORMERS_CACHE` so the server finds the Wan2.1 cache, then start the server:

```bash
conda activate dreamzero
export HF_HOME="/workspace/hf_cache"             # same path as in Step 5
export TRANSFORMERS_CACHE="$HF_HOME"

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
  --model-path ./checkpoints/DreamZero-DROID
```

Wait until the log shows the server is listening. First startup can take several minutes (model + WAN loading).

### Step 9: Run the test client (in a second terminal)

```bash
cd dreamzero
conda activate dreamzero
python test_client_AR.py --port 5000
```

The client uses videos in `debug_image/` (e.g. `exterior_image_1_left.mp4`, `wrist_image_left.mp4`) and sends frames to the server. First inferences may be slow (warmup); then ~3 s per inference on H100.

---

## Prerequisites (reference)

- **Hardware**: Multi-GPU (e.g. H100); minimum 2 GPUs for distributed inference.
- **OS**: Linux.
- **Conda**: Miniconda or Anaconda.
- **CUDA**: 12.9 (driver and toolkit).
- **Hugging Face**: Account and token for downloading model weights (required for download script; server will also use HF cache for Wan2.1).

---

## Alternative: one-shot setup script

If you prefer a single script instead of the step-by-step above:

From the repo root:

```bash
bash scripts/setup_env.sh
```

This will:

1. Create a conda env `dreamzero` with Python 3.11.
2. Install the project in editable mode with PyTorch 2.8 and CUDA 12.9 wheels.
3. Install `flash-attn` (no Transformer Engine; for H100 only).

Then activate and optionally download weights:

```bash
conda activate dreamzero
bash scripts/download_weights.sh   # optional; see section 4
```

## 3. Manual setup (for reproducibility)

If you prefer to run steps by hand or need to adapt them:

### 3.1 Conda environment

```bash
conda create -n dreamzero python=3.11
conda activate dreamzero
```

### 3.2 PyTorch 2.8 + CUDA 12.9 and project deps

Use the PyTorch CUDA 12.9 index so that `torch`, `torchvision`, and `torchaudio` match the stack below:

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
```

- **Python**: 3.11  
- **PyTorch**: 2.8.x  
- **CUDA**: 12.9  
- **torchvision / torchaudio**: versions pinned in `pyproject.toml`

### 3.3 Flash Attention

```bash
MAX_JOBS=8 pip install --no-build-isolation flash-attn
```

Adjust `MAX_JOBS` if needed (e.g. fewer for low-memory machines). Build may take 10–30 minutes.

**What `--no-build-isolation` does**

- **Default (with isolation):** pip creates a temporary environment, installs the package’s *build* dependencies (e.g. `ninja`, `packaging`) there, then builds the package. Your main env stays unchanged during the build.
- **With `--no-build-isolation`:** pip does **not** create that temporary env. It uses your **current** environment to build. So:
  - Build dependencies (e.g. `ninja`, `packaging`, and for flash-attn often a matching CUDA/PyTorch) must already be installed.
  - The build can see your installed PyTorch/CUDA and link against them correctly, which is why flash-attn docs often recommend it.
  - If something is missing, the build fails and pip won’t auto-install those build deps.
  - It’s useful when the isolated build would use wrong versions or when you need explicit control over the build environment.

### 3.4 Transformer Engine (GB200 only; skip on H100)

On **H100**, do **not** install Transformer Engine. On **GB200** only:

```bash
pip install --no-build-isolation transformer_engine[pytorch]
```

## 4. Downloading model weights

Two sets of weights are used:

1. **DreamZero-DROID** – main policy checkpoint (required for inference).
2. **Wan2.1-I2V-14B-480P** – video/diffusion model (can be downloaded on first run via `huggingface_hub`, or pre-downloaded).

Pre-downloading both in parallel (recommended so the first run is fast):

- **Hugging Face token**: Set `HF_TOKEN` or source a file that exports it (e.g. `~/.bash_aliases` with `export HF_TOKEN=...`). Do **not** commit the token.
- **Cache (optional)**: Set `HF_HOME` and/or `TRANSFORMERS_CACHE` to a directory (e.g. `/workspace/hf_cache`) so all Hugging Face assets use one place.

Example:

```bash
# Optional: source env that sets HF_TOKEN and cache dirs
[ -f ~/.bash_aliases ] && source ~/.bash_aliases

conda activate dreamzero
bash scripts/download_weights.sh
```

Default checkpoint directory: `./checkpoints/DreamZero-DROID`. Override with:

```bash
bash scripts/download_weights.sh /path/to/checkpoint_dir
```

The script runs the DreamZero-DROID and Wan2.1 downloads in parallel. Wan2.1 is stored in the Hugging Face cache (e.g. `$HF_HOME`); the runtime code will use it from there.

Manual download (if you prefer not to use the script):

```bash
# DreamZero-DROID (required for --model-path)
huggingface-cli download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir ./checkpoints/DreamZero-DROID

# Wan2.1 (optional; will be downloaded on first run if missing)
# Set HF_TOKEN and HF_HOME as above, then:
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P')"
```

## 5. Verify installation

```bash
conda activate dreamzero
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import flash_attn; print('flash_attn OK')"
```

You should see a PyTorch 2.8.x version, `True` for CUDA, and `flash_attn OK`.

## 6. Run the inference server

After weights are in place (e.g. `./checkpoints/DreamZero-DROID`):

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
  --model-path ./checkpoints/DreamZero-DROID
```

Then in another terminal:

```bash
python test_client_AR.py --port 5000
```

## Summary: versions for reproducibility

| Component   | Version / note        |
|------------|------------------------|
| Python     | 3.11                   |
| CUDA       | 12.9                   |
| PyTorch    | 2.8.x                  |
| torchvision| 0.23.x                 |
| torchaudio | 2.8.x                  |
| flash-attn | from PyPI (no-build-isolation) |
| Transformer Engine | Not used on H100; only on GB200 |

## Troubleshooting

- **CUDA / driver**: Ensure `nvidia-smi` shows a driver that supports CUDA 12.9 and that the CUDA toolkit 12.9 is available (e.g. `nvcc --version` if installed).
- **flash-attn build**: If the build fails, try reducing `MAX_JOBS` or installing in a clean env with only PyTorch and CUDA installed first.
- **Missing weights**: If the server fails loading the model, ensure `--model-path` points at the downloaded DreamZero-DROID directory and that `HF_HOME` (or default cache) contains the Wan2.1 repo if that backend is used.
