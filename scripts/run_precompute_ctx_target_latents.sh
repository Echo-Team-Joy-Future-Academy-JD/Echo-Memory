#!/bin/bash
# Precompute ctx (1 latent/frame) and target (1 latent/4 frames) latents for Context-as-Memory dataset.
#
# No metadata needed: auto-discovers segments from frames/ + captions.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

DATASET_BASE="${DATASET_BASE_PATH:-${REPO_ROOT}/data/Context-as-Memory-Dataset}"
OUTPUT_DIR="${DATASET_BASE}/latents"
WAN_BASE_MODEL="${WAN_BASE_MODEL:-${REPO_ROOT}/checkpoints/Wan2.1-T2V-1.3B}"
MODEL_PATHS="${MODEL_PATHS:-[\"${WAN_BASE_MODEL}/diffusion_pytorch_model.safetensors\",\"${WAN_BASE_MODEL}/models_t5_umt5-xxl-enc-bf16.pth\",\"${WAN_BASE_MODEL}/Wan2.1_VAE.pth\"]}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-20}"

cd "${REPO_ROOT}"

accelerate launch --num_processes "${NUM_PROCESSES}" scripts/precompute_ctx_target_latents.py \
  --dataset_base_path "${DATASET_BASE}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_paths "${MODEL_PATHS}" \
  --height 352 --width 640 \
  --context_frames "${CONTEXT_FRAMES}" \
  --target_frames_per_latent 4 \
  --no_metadata \
  --use_overlap_labels \
  --overlap_labels_dense \
  --skip_existing

echo "Done. Latents saved to ${OUTPUT_DIR}/ctx_latents/ and ${OUTPUT_DIR}/target_latents/"
