#!/bin/bash
# Dynamic SpatialVID: Context K=1 baseline.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

EXTRA_ARGS=()
[ -n "${MAX_TRAIN_STEPS:-}" ] && EXTRA_ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}")
[ -n "${PROGRESS_TOTAL_STEPS:-}" ] && EXTRA_ARGS+=(--progress_total_steps "${PROGRESS_TOTAL_STEPS}")

accelerate launch src/model_training/train.py \
  --dataset_base_path "${dataset_base_path}" --dataset_metadata_path "${dataset_base_path}/${METADATA_NAME}" \
  --enable_fov_retrieval --fov_top_k 0 --context_memory_frames 1 --use_rt_relative --height 352 --width 640 \
  --dataset_repeat 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS:-16}" \
  --model_paths "${model_paths}" --tokenizer_path "${TOKENIZER_PATH:-${WAN_BASE_MODEL}/google/umt5-xxl}" \
  --learning_rate "${LEARNING_RATE:-5e-5}" --num_epochs "${NUM_EPOCHS:-1}" --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}" \
  --output_path "${output_base}_ctx1" --trainable_models dit --ckpt_interval "${CKPT_INTERVAL:-1000}" --save_full_model \
  --wandb_run_name "${WANDB_RUN_NAME:-dyn_spatialvid_ctx1}" \
  --enable_context_memory --training_mode context --context_drop_prob 0.1 --cfg_target_only \
  --train_cam_pose --add_action_attn --action_use_temporal_attention \
  --timestep_shift "${TIMESTEP_SHIFT:-5}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/dyn_spatialvid_ctx1_$(date +%Y%m%d_%H%M%S).log"
