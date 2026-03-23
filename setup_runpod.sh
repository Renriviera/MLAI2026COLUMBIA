#!/usr/bin/env bash
# =============================================================================
# ONE-TIME RunPod setup for MLAI2026COLUMBIA
# Run once after creating a new pod. Installs env on network volume so it
# persists across pod restarts.
#
# Assumptions:
#   - RunPod pytorch template (CUDA + PyTorch pre-installed in base image)
#   - Network volume mounted at /workspace
#   - A100 PCIe 80GB, 125GB RAM, 16 vCPU
#
# Usage:
#   chmod +x setup_runpod.sh && ./setup_runpod.sh
# =============================================================================
set -euo pipefail

NETWORK_VOL="/workspace"
REPO_DIR="${NETWORK_VOL}/MLAI2026COLUMBIA"
MODEL_DIR="${NETWORK_VOL}/models"
ENV_DIR="${NETWORK_VOL}/envs/iris"
HF_CACHE="${NETWORK_VOL}/.cache/huggingface"
DEFAULT_MODEL="Qwen/Qwen2-7B-Instruct"

echo "============================================"
echo " MLAI2026COLUMBIA — RunPod Setup"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# ── 0. Verify GPU ───────────────────────────────────────────────────────────
echo ""
echo "[0/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── 1. Directory structure on network volume ────────────────────────────────
echo "[1/5] Setting up directory structure on network volume..."
mkdir -p "${MODEL_DIR}"
mkdir -p "${HF_CACHE}"
mkdir -p "${ENV_DIR}"

# Symlink HuggingFace cache so downloads persist across restarts
if [ ! -L "$HOME/.cache/huggingface" ]; then
    rm -rf "$HOME/.cache/huggingface" 2>/dev/null || true
    ln -sf "${HF_CACHE}" "$HOME/.cache/huggingface"
    echo "  Symlinked ~/.cache/huggingface -> ${HF_CACHE}"
fi

# ── 2. Clone / update repo ─────────────────────────────────────────────────
echo "[2/5] Setting up repository..."
if [ -d "${REPO_DIR}/.git" ] || [ -f "${REPO_DIR}/attack_llm_core_base.py" ]; then
    echo "  Repo already exists at ${REPO_DIR}, pulling latest..."
    cd "${REPO_DIR}"
    git pull 2>/dev/null || echo "  (not a git repo or no remote — skipping pull)"
else
    echo "  Repo not found. Copy or clone your code to ${REPO_DIR}"
    echo "  Example: rsync -avz /local/path/MLAI2026COLUMBIA/ ${REPO_DIR}/"
    mkdir -p "${REPO_DIR}"
fi
cd "${REPO_DIR}"

# ── 3. Python environment ──────────────────────────────────────────────────
echo "[3/5] Setting up Python environment..."

if command -v conda &>/dev/null; then
    echo "  Conda available — creating persistent env at ${ENV_DIR}"
    if [ ! -f "${ENV_DIR}/bin/python" ]; then
        conda create -y -p "${ENV_DIR}" python=3.10
    fi
    # shellcheck disable=SC1091
    source activate "${ENV_DIR}" 2>/dev/null || conda activate "${ENV_DIR}"

    pip install --upgrade pip
    pip install -e "${REPO_DIR}"
else
    echo "  No conda found — using system Python + pip"
    pip install --upgrade pip
    pip install -e "${REPO_DIR}"
fi

# Verify torch sees CUDA
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ── 4. Download model weights ──────────────────────────────────────────────
echo "[4/5] Downloading model weights (if needed)..."

# Set HF_TOKEN env var or log in via `huggingface-cli login` for gated models.
if [ -n "${HF_TOKEN:-}" ]; then
    echo "  HF_TOKEN detected — using for gated model access"
    huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || true
fi

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, sys

model_id = '${DEFAULT_MODEL}'
cache_dir = '${MODEL_DIR}'

if os.path.isdir(os.path.join(cache_dir, model_id.split('/')[-1])):
    print(f'  Model already cached at {cache_dir}/{model_id.split(\"/\")[-1]}')
    sys.exit(0)

print(f'  Downloading {model_id} to {cache_dir}...')
print('  (This may take 5-10 minutes on first run)')
try:
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype='auto',
    )
    print(f'  Model downloaded successfully')
except Exception as e:
    print(f'  WARNING: Could not download model: {e}')
    print(f'  You may need to set HF_TOKEN for gated models')
    sys.exit(0)
"

# ── 5. Patch model paths in scripts ────────────────────────────────────────
echo "[5/5] Creating local config..."

# Write a small env file that start_session.sh and scripts can source
cat > "${REPO_DIR}/.env.runpod" <<ENVEOF
export REPO_DIR="${REPO_DIR}"
export MODEL_PATH="${MODEL_DIR}"
export DEFAULT_MODEL_PATH="${MODEL_DIR}/models--Qwen--Qwen2-7B-Instruct/snapshots"
export HF_HOME="${HF_CACHE}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"  # fallback; prefer pip install -e
ENVEOF

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Quick reference:"
echo "   source ${REPO_DIR}/.env.runpod"
echo "   source ${REPO_DIR}/start_session.sh"
echo ""
echo " Model path for scripts:"
echo "   --model_path \${DEFAULT_MODEL_PATH}/\$(ls \${DEFAULT_MODEL_PATH} | head -1)"
echo "   (or use the resolved path printed by start_session.sh)"
echo "============================================"
