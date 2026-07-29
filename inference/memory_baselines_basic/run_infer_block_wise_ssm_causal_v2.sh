#!/usr/bin/env bash
# Multi-chunk inference for the released causal Block-SSM v2 checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

: "${CKPT:?ERROR: set CKPT to block_wise_ssm_causal_v2/epoch-0.safetensors}"
: "${CONTEXT_IMAGE:?ERROR: set CONTEXT_IMAGE for the first chunk}"
: "${ACTION_PATH:?ERROR: set ACTION_PATH to an 81-frame camera trajectory JSON}"
OUTPUT="${INFER_OUTPUT_ROOT}/block_wise_ssm_causal_v2_$(date +%Y%m%d_%H%M%S).mp4"

python inference/unified_inference.py \
  --ckpt "${CKPT}" --memory_type block_wise_ssm_causal_v2 \
  --base_model "${WAN_BASE_MODEL}" --prompt "${PROMPT}" \
  --context_image "${CONTEXT_IMAGE}" --action_path "${ACTION_PATH}" \
  --num_chunks "${NUM_CHUNKS:-2}" \
  --height "${HEIGHT}" --width "${WIDTH}" --num_frames "${NUM_FRAMES}" \
  --seed "${SEED}" --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --sigma_shift "${SIGMA_SHIFT}" --cfg_scale "${CFG_SCALE}" --fps "${FPS}" \
  --output_path "${OUTPUT}"
