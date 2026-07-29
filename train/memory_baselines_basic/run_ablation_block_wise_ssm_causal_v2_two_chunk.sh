#!/usr/bin/env bash
# Released causal Block-SSM v2 recipe: leak-free prefix history, no MoC.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
source "${SCRIPT_DIR}/common_sampling_two_chunk.sh"
export CONTEXT_POSITION=prefix

accelerate launch src/model_training/train.py \
  --dataset_base_path "${dataset_base_path}" \
  --dataset_metadata_path "${dataset_base_path}/${METADATA_NAME}" \
  --context_source causal_prev_prefix --context_memory_frames 5 \
  --context_per_frame_vae --use_rt_relative \
  --height 352 --width 640 \
  --dataset_repeat 1 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS:-2}" \
  --model_paths "${model_paths}" --learning_rate 5e-5 \
  --num_epochs "${NUM_EPOCHS:-2}" --max_train_steps "${MAX_TRAIN_STEPS:-30000}" \
  --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}" \
  --output_path "${output_base}_block_wise_ssm_causal_v2" \
  --trainable_models dit --ckpt_interval "${CKPT_INTERVAL:-1000}" \
  --save_full_model --wandb_run_name "block_wise_ssm_causal_v2" \
  --enable_context_memory --training_mode context \
  --condition_t2v_ratio 0 --condition_i2v_ratio 0.10 \
  --train_cam_pose --add_action_attn --action_use_temporal_attention \
  --use_block_wise_ssm --block_wise_ssm_causal_v2 \
  --ssm_every_n_blocks 4 --ssm_num_blocks_hint 26 \
  --timestep_shift "${TIMESTEP_SHIFT:-15}" \
  "${SAMPLING_TWO_CHUNK_FLAGS[@]}" \
  2>&1 | tee "${LOG_DIR}/block_wise_ssm_causal_v2_$(date +%Y%m%d_%H%M%S).log"
