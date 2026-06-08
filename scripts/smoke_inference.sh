#!/usr/bin/env bash
# Minimal inference smoke test for Echo-Memory eval/inference wiring.
#
# Required:
#   WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
#   DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
#   CKPT=/path/to/epoch-0.safetensors
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f "${CONDA_SH:-}" ]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/pfs/weiyang/Miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/pfs/weiyang/Miniconda3/etc/profile.d/conda.sh"
fi

if [ -n "${ECHO_MEMORY_CONDA_ENV:-}" ]; then
  conda activate "$ECHO_MEMORY_CONDA_ENV"
elif [ -n "${CAM_CONDA_ENV:-}" ]; then
  conda activate "$CAM_CONDA_ENV"
fi

: "${WAN_BASE_MODEL:?Set WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B}"
: "${DATASET_BASE_PATH:?Set DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset}"
: "${CKPT:?Set CKPT=/path/to/epoch-0.safetensors}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WAN_BASE_MODEL DATASET_BASE_PATH

OUT_DIR="${OUT_DIR:-$ROOT/outputs/smoke_inference/$(date +%Y%m%d_%H%M%S)}"
VIDEO_NAME="${VIDEO_NAME:-AncientTempleEnv_0}"
START_FRAME="${START_FRAME:-0}"
CHUNK_FRAMES="${CHUNK_FRAMES:-21}"
HEIGHT="${HEIGHT:-128}"
WIDTH="${WIDTH:-256}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-2}"
CFG_SCALE="${CFG_SCALE:-1.0}"

mkdir -p "$OUT_DIR"

python eval/v2/basic/replay_gt_error.py \
  --ckpt "$CKPT" \
  --dataset_base "$DATASET_BASE_PATH" \
  --video_name "$VIDEO_NAME" \
  --start_frame "$START_FRAME" \
  --num_chunks 1 \
  --chunk_frames "$CHUNK_FRAMES" \
  --context_frames "${CONTEXT_FRAMES:-1}" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_inference_steps "$NUM_INFERENCE_STEPS" \
  --cfg_scale "$CFG_SCALE" \
  --output_dir "$OUT_DIR" \
  --no_lpips \
  --write_csv

echo "[smoke_inference] OK: $OUT_DIR"
