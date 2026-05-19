#!/bin/bash
# Ablation: SpatialGridMemory stored but not injected into the generator.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
source "${SCRIPT_DIR}/common_sampling_two_chunk.sh"
accelerate launch src/model_training/train.py \
  --dataset_base_path "${dataset_base_path}" --dataset_metadata_path "${dataset_base_path}/${METADATA_NAME}" \
  --context_source prev_chunk_tail --context_memory_frames 1 --use_rt_relative --height 352 --width 640 \
  --dataset_repeat 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_workers 16 \
  --model_paths "${model_paths}" --learning_rate 5e-5 --num_epochs 1 --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}" \
  --output_path "${output_base}_abl_spatial_inject_none_two_chunk" --trainable_models dit --ckpt_interval "${CKPT_INTERVAL:-1000}" --save_full_model \
  --wandb_run_name "abl_spatial_inject_none_two_chunk" \
  --enable_context_memory --training_mode context --condition_t2v_ratio 0.10 --condition_i2v_ratio 0.10 --cfg_target_only \
  --train_cam_pose --add_action_attn --action_use_temporal_attention \
  --spike_threshold "${SPIKE_THRESHOLD:-15.0}" \
  --use_moc --moc_temperature 1.0 \
  --use_spatial_memory --spatial_memory_tokens 64 --spatial_memory_inject_mode none \
  --timestep_shift "${TIMESTEP_SHIFT:-15}" \
  "${SAMPLING_TWO_CHUNK_FLAGS[@]}" \
  2>&1 | tee "${LOG_DIR}/abl_spatial_inject_none_two_chunk_$(date +%Y%m%d_%H%M%S).log"
