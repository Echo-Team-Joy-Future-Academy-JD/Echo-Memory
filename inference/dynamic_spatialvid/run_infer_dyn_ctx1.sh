#!/bin/bash
# Dynamic SpatialVID inference: Context K=1.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
: "${CKPT:?ERROR: set CKPT to your dynamic ctx1 checkpoint path}"
OUTPUT="${INFER_OUTPUT_ROOT}/dyn_ctx1_$(date +%Y%m%d_%H%M%S).mp4"
EXTRA_ARGS=()
[ -n "${CONTEXT_IMAGE}" ] && EXTRA_ARGS+=(--context_image "${CONTEXT_IMAGE}")
[ -n "${ACTION_PATH}" ] && EXTRA_ARGS+=(--action_path "${ACTION_PATH}")
[ -n "${TOKENIZER_PATH}" ] && EXTRA_ARGS+=(--tokenizer_path "${TOKENIZER_PATH}")

python inference/unified_inference.py \
  --ckpt "${CKPT}" --memory_type context_k1 --base_model "${WAN_BASE_MODEL}" \
  --prompt "${PROMPT}" "${EXTRA_ARGS[@]}" \
  --height "${HEIGHT}" --width "${WIDTH}" --num_frames "${NUM_FRAMES}" \
  --seed "${SEED}" --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --sigma_shift "${SIGMA_SHIFT:-5.0}" --cfg_scale "${CFG_SCALE}" --fps "${FPS}" \
  --output_path "${OUTPUT}"
