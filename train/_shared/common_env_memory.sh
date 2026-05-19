#!/bin/bash
# Shared memory-training environment fragment.
# Source this file from train/*/common_env.sh before setting output_base.
set -euo pipefail

if [ -f "${CONDA_SH:-}" ]; then
  source "${CONDA_SH}"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
if [ -n "${ECHO_MEMORY_CONDA_ENV:-}" ]; then
  conda activate "${ECHO_MEMORY_CONDA_ENV}" 2>/dev/null || true
elif [ -n "${CAM_CONDA_ENV:-}" ]; then
  conda activate "${CAM_CONDA_ENV}" 2>/dev/null || true
elif [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  conda activate echo-memory 2>/dev/null || true
fi

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
  _default_ifname="$(ip -o -4 route show to default 2>/dev/null | awk '{print $5; exit}')"
  export NCCL_SOCKET_IFNAME="${_default_ifname:-eth0}"
fi
export CONTEXT_POSITION=suffix
export USE_CONCATENATION_INFERENCE=true
export USE_RT_RELATIVE=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

_first_existing_dir() {
  local c
  for c in "$@"; do
    [ -n "${c}" ] || continue
    [ -d "${c}" ] && { echo "${c}"; return 0; }
  done
  return 1
}

_WAN_BASE_DEFAULT="$(_first_existing_dir \
  "${REPO_ROOT}/checkpoints/Wan2.1-T2V-1.3B" \
  "${REPO_ROOT}/checkpoints/wan2.1-t2v-1.3b" \
  "${REPO_ROOT}/models/Wan2.1-T2V-1.3B" \
 || true)"
_WAN_BASE_DEFAULT="${_WAN_BASE_DEFAULT:-}"
WAN_BASE_MODEL="${WAN_BASE_MODEL:-${_WAN_BASE_DEFAULT:-}}"

_DATASET_BASE_DEFAULT="$(_first_existing_dir \
  "${REPO_ROOT}/data/Context-as-Memory-Dataset" \
  "${REPO_ROOT}/data/Context-as-Memory-Dataset/videos/Context-as-Memory-Dataset" \
 || true)"
_DATASET_BASE_DEFAULT="${_DATASET_BASE_DEFAULT:-}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-${_DATASET_BASE_DEFAULT:-}}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
export WAN_BASE_MODEL DATASET_BASE_PATH LOG_DIR REPO_ROOT

_require_file() {
  local p="$1"
  [ -f "${p}" ] || {
    echo "[common_env_memory][ERROR] missing file: ${p}" >&2
    echo "[common_env_memory][HINT] set WAN_BASE_MODEL to your Wan2.1 base model directory." >&2
    exit 2
  }
}

_require_dir() {
  local p="$1"
  [ -d "${p}" ] || {
    echo "[common_env_memory][ERROR] missing dir: ${p}" >&2
    echo "[common_env_memory][HINT] set DATASET_BASE_PATH to your dataset root." >&2
    exit 2
  }
}

dataset_base_path="${DATASET_BASE_PATH}"
METADATA_NAME=metadata_full.csv
model_paths="[\"${WAN_BASE_MODEL}/diffusion_pytorch_model.safetensors\",\"${WAN_BASE_MODEL}/models_t5_umt5-xxl-enc-bf16.pth\",\"${WAN_BASE_MODEL}/Wan2.1_VAE.pth\"]"
remove_prefix_in_ckpt="pipe.dit."

if [ -z "${WAN_BASE_MODEL}" ]; then
  echo "[common_env_memory][ERROR] WAN_BASE_MODEL is not set." >&2
  echo "[common_env_memory][HINT] export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B" >&2
  exit 2
fi
if [ -z "${DATASET_BASE_PATH}" ]; then
  echo "[common_env_memory][ERROR] DATASET_BASE_PATH is not set." >&2
  echo "[common_env_memory][HINT] export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset" >&2
  exit 2
fi
_require_file "${WAN_BASE_MODEL}/diffusion_pytorch_model.safetensors"
_require_file "${WAN_BASE_MODEL}/models_t5_umt5-xxl-enc-bf16.pth"
_require_file "${WAN_BASE_MODEL}/Wan2.1_VAE.pth"
_require_dir "${DATASET_BASE_PATH}"
echo "[common_env_memory] WAN_BASE_MODEL=${WAN_BASE_MODEL}"
echo "[common_env_memory] DATASET_BASE_PATH=${DATASET_BASE_PATH}"

action_dir="${ACTION_DIR:-${REPO_ROOT}/env}"
sampling_action_path="${action_dir}/action_rotation_left_45.json"
[ ! -f "${sampling_action_path}" ] && (python3 "${action_dir}/generate_rotation_actions.py" 2>/dev/null || true)
cd "${REPO_ROOT}"
