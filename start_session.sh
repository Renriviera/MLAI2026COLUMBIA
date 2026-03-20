#!/usr/bin/env bash
# =============================================================================
# QUICK START — run at the beginning of every RunPod session (or agent session).
# Sources the persisted environment, activates conda, resolves model path,
# and prints a ready-to-go status dashboard.
#
# Usage:
#   source start_session.sh          # sources into current shell
#   # OR
#   bash start_session.sh            # prints status only (no env activation)
# =============================================================================

NETWORK_VOL="/workspace"
REPO_DIR="${NETWORK_VOL}/MLAI2026COLUMBIA"
ENV_FILE="${REPO_DIR}/.env.runpod"

# ── Source env ──────────────────────────────────────────────────────────────
if [ -f "${ENV_FILE}" ]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
else
    echo "WARNING: ${ENV_FILE} not found. Run setup_runpod.sh first."
fi

cd "${REPO_DIR}" 2>/dev/null || true

# ── Activate conda env if available ─────────────────────────────────────────
IRIS_ENV="${NETWORK_VOL}/envs/iris"
if [ -f "${IRIS_ENV}/bin/python" ]; then
    if command -v conda &>/dev/null; then
        source activate "${IRIS_ENV}" 2>/dev/null || conda activate "${IRIS_ENV}" 2>/dev/null || true
    else
        export PATH="${IRIS_ENV}/bin:${PATH}"
    fi
fi

# ── Resolve model path ─────────────────────────────────────────────────────
MODEL_DIR="${NETWORK_VOL}/models"
RESOLVED_MODEL_PATH=""

# Check HuggingFace cache structure
HF_SNAP="${MODEL_DIR}/models--meta-llama--Llama-2-7b-chat-hf/snapshots"
if [ -d "${HF_SNAP}" ]; then
    LATEST_SNAP=$(ls -t "${HF_SNAP}" 2>/dev/null | head -1)
    if [ -n "${LATEST_SNAP}" ]; then
        RESOLVED_MODEL_PATH="${HF_SNAP}/${LATEST_SNAP}"
    fi
fi

# Check if model was saved directly
if [ -z "${RESOLVED_MODEL_PATH}" ] && [ -d "${MODEL_DIR}/Llama-2-7b-chat-hf" ]; then
    RESOLVED_MODEL_PATH="${MODEL_DIR}/Llama-2-7b-chat-hf"
fi

export RESOLVED_MODEL_PATH

# ── GPU dashboard ───────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MLAI2026COLUMBIA — Session Ready                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Repo:        ${REPO_DIR}"
echo "  Python:      $(python --version 2>&1 || echo 'not found')"
echo "  PyTorch:     $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA:        $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU:         ${GPU_NAME} (${GPU_MEM})"
    echo "  GPU util:    ${GPU_UTIL} | VRAM used: ${GPU_MEM_USED}"
else
    echo "  GPU:         nvidia-smi not found"
fi

if [ -n "${RESOLVED_MODEL_PATH}" ]; then
    echo "  Model:       ${RESOLVED_MODEL_PATH}"
else
    echo "  Model:       NOT FOUND — run setup_runpod.sh or set HF_TOKEN"
fi

echo ""
echo "  Quick commands:"
echo "    make help            # Show all targets"
echo "    make status          # GPU + disk usage"
echo "    make smoke-test      # Fast sanity check (~30 min, 5 behaviors x 100 steps)"
echo "    make quick-eval      # Quick ASR estimate (~3 h, 15 behaviors x 200 steps)"
echo "    make attack-single   # Run 1 base GCG behavior (BEHAVIOR_ID=1)"
echo "    make robust-a        # Run 1 robust GCG-A behavior (BEHAVIOR_ID=1)"
echo "    make backup-results  # Copy results to network volume"
echo ""
echo "  Subset methods:  make smoke-test METHODS=A,B"
echo ""

# Export convenience alias for scripts
export MODEL_FLAG="--model_path ${RESOLVED_MODEL_PATH}"
