#!/bin/bash
# Shared inference environment — mirrors train/_shared/common_env_memory.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ── Base model ────────────────────────────────────────────────────────
WAN_BASE_MODEL="${WAN_BASE_MODEL:-}"
if [ -z "${WAN_BASE_MODEL}" ]; then
  for _d in \
    "${REPO_ROOT}/checkpoints/Wan2.1-T2V-1.3B" \
    "${REPO_ROOT}/models/Wan2.1-T2V-1.3B" \
    "${REPO_ROOT}/Wan2.1-T2V-1.3B"; do
    [ -d "${_d}" ] && { WAN_BASE_MODEL="${_d}"; break; }
  done
fi
if [ -z "${WAN_BASE_MODEL}" ]; then
  echo "[common_env_infer] ERROR: WAN_BASE_MODEL not set." >&2
  echo "[common_env_infer] HINT: export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B" >&2
  exit 2
fi
for _f in diffusion_pytorch_model.safetensors models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth; do
  [ -f "${WAN_BASE_MODEL}/${_f}" ] || {
    echo "[common_env_infer] ERROR: missing ${WAN_BASE_MODEL}/${_f}" >&2; exit 2; }
done
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
if [ -z "${TOKENIZER_PATH}" ] && [ -d "${WAN_BASE_MODEL}/google/umt5-xxl" ]; then
  TOKENIZER_PATH="${WAN_BASE_MODEL}/google/umt5-xxl"
fi
export WAN_BASE_MODEL
export TOKENIZER_PATH

# ── Checkpoint ────────────────────────────────────────────────────────
# CKPT must be set by the caller or the wrapper script.
# Inference scripts validate this themselves.

# ── Output ────────────────────────────────────────────────────────────
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-${REPO_ROOT}/inference_outputs}"
mkdir -p "${INFER_OUTPUT_ROOT}"
export INFER_OUTPUT_ROOT

# ── Defaults ──────────────────────────────────────────────────────────
PROMPT="${PROMPT:-A game scene, the camera moves through the environment}"
CONTEXT_IMAGE="${CONTEXT_IMAGE:-}"
ACTION_PATH="${ACTION_PATH:-${REPO_ROOT}/env/action_rotation_left_45.json}"
SEED="${SEED:-0}"
HEIGHT="${HEIGHT:-352}"
WIDTH="${WIDTH:-640}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SIGMA_SHIFT="${SIGMA_SHIFT:-15.0}"
CFG_SCALE="${CFG_SCALE:-5.0}"
FPS="${FPS:-15}"

echo "[common_env_infer] WAN_BASE_MODEL=${WAN_BASE_MODEL}"
echo "[common_env_infer] INFER_OUTPUT_ROOT=${INFER_OUTPUT_ROOT}"
cd "${REPO_ROOT}"
