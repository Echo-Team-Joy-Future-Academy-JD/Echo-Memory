#!/bin/bash
# Geometry-grounded Spatial Memory.
# Requires metadata column GEOMETRY_MEMORY_COLUMN whose values point to
# TSDF/point-cloud-rendered static videos (for example Vid_masktarget.mp4).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

GEOMETRY_MEMORY_COLUMN="${GEOMETRY_MEMORY_COLUMN:-geometry_memory}"
GEOMETRY_MEMORY_ROOT="${GEOMETRY_MEMORY_ROOT:-${dataset_base_path}}"
EXTRA_ARGS=()
[ -n "${MAX_TRAIN_STEPS:-}" ] && EXTRA_ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}")
[ -n "${PROGRESS_TOTAL_STEPS:-}" ] && EXTRA_ARGS+=(--progress_total_steps "${PROGRESS_TOTAL_STEPS}")

accelerate launch src/model_training/train.py \
  --dataset_base_path "${dataset_base_path}" --dataset_metadata_path "${dataset_base_path}/${METADATA_NAME}" \
  --geometry_memory_column "${GEOMETRY_MEMORY_COLUMN}" --geometry_memory_root "${GEOMETRY_MEMORY_ROOT}" \
  --context_source prev_chunk_tail --context_memory_frames 1 --use_rt_relative --height 352 --width 640 \
  --dataset_repeat 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS:-16}" \
  --model_paths "${model_paths}" --learning_rate 5e-5 --num_epochs 1 --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}" \
  --output_path "${output_base}_geometry_spatial_mem" --trainable_models dit --ckpt_interval "${CKPT_INTERVAL:-1000}" --save_full_model \
  --wandb_run_name "memory_baseline_geometry_spatial_mem" \
  --enable_context_memory --training_mode context --condition_t2v_ratio 0.10 --condition_i2v_ratio 0.10 --cfg_target_only \
  --train_cam_pose --add_action_attn --action_use_temporal_attention \
  --spike_threshold "${SPIKE_THRESHOLD:-15.0}" \
  --use_moc --moc_temperature 1.0 \
  --use_geometry_spatial_memory --geometry_spatial_memory_tokens 64 \
  --geometry_spatial_memory_grid 8 --geometry_spatial_memory_temporal_bins 4 \
  --geometry_spatial_memory_inject_mode "${GEOMETRY_MEMORY_INJECT_MODE:-concat_text}" \
  --timestep_shift "${TIMESTEP_SHIFT:-15}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/memory_baseline_geometry_spatial_mem_$(date +%Y%m%d_%H%M%S).log"

