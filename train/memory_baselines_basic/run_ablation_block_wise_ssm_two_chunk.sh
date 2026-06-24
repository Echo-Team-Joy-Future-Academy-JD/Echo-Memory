#!/bin/bash
# Ablation: paper-style block-wise SSM (arXiv:2505.20171) + 2-chunk monitor. Prefer over legacy VideoSSM hybrid for SSM table.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
source "${SCRIPT_DIR}/common_sampling_two_chunk.sh"
accelerate launch src/model_training/train.py \
  --dataset_base_path "${dataset_base_path}" --dataset_metadata_path "${dataset_base_path}/${METADATA_NAME}" \
  --context_source replay --prev_chunk_frames 81 --context_memory_frames 5 --use_rt_relative --height 352 --width 640 \
  --dataset_repeat 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_workers 16 \
  --model_paths "${model_paths}" --learning_rate 5e-5 --num_epochs 1 --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}" \
  --output_path "${output_base}_abl_block_wise_ssm_two_chunk" --trainable_models dit --ckpt_interval "${CKPT_INTERVAL:-1000}" --save_full_model \
  --wandb_run_name "abl_block_wise_ssm_two_chunk" \
  --enable_context_memory --training_mode context --context_drop_prob 0.1 --cfg_target_only \
  --train_cam_pose --add_action_attn --action_use_temporal_attention \
  --use_moc --moc_temperature 1.0 \
  --use_block_wise_ssm --ssm_every_n_blocks 4 --ssm_num_blocks_hint 21 \
  --timestep_shift "${TIMESTEP_SHIFT:-15}" \
  "${SAMPLING_TWO_CHUNK_FLAGS[@]}" \
  2>&1 | tee "${LOG_DIR}/abl_block_wise_ssm_two_chunk_$(date +%Y%m%d_%H%M%S).log"
