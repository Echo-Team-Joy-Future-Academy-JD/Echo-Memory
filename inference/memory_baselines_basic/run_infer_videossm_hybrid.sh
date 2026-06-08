#!/bin/bash
# Inference: videossm_hybrid
# Corresponds to: train/memory_baselines_basic/run_videossm_hybrid_baseline.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

: "${CKPT:?ERROR: set CKPT to your videossm_hybrid checkpoint path}"
OUTPUT="${INFER_OUTPUT_ROOT}/videossm_hybrid_$(date +%Y%m%d_%H%M%S).mp4"

EXTRA_ARGS=()
if [ -n "${CONTEXT_IMAGE}" ]; then
  EXTRA_ARGS+=(--context_image "${CONTEXT_IMAGE}")
fi
if [ -n "${ACTION_PATH}" ]; then
  EXTRA_ARGS+=(--action_path "${ACTION_PATH}")
fi

python inference/unified_inference.py \
  --ckpt "${CKPT}" \
  --memory_type videossm_hybrid \
  --base_model "${WAN_BASE_MODEL}" \
  --prompt "${PROMPT}" \
  "${EXTRA_ARGS[@]}" \
  --height "${HEIGHT}" --width "${WIDTH}" --num_frames "${NUM_FRAMES}" \
  --seed "${SEED}" --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --sigma_shift "${SIGMA_SHIFT}" --cfg_scale "${CFG_SCALE}" --fps "${FPS}" \
  --output_path "${OUTPUT}"
