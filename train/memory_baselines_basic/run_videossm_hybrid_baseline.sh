#!/bin/bash
# Legacy VideoSSM hybrid (depthwise conv). For paper-aligned SSM narrative prefer --use_block_wise_ssm in a separate script.
# VideoSSM baseline: hybrid state-space memory in DiT blocks
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
accelerate launch src/model_training/train.py \
  --dataset_base_path "${dataset_base_path}" --dataset_metadata_path "${dataset_base_path}/${METADATA_NAME}" \
  --enable_fov_retrieval --fov_top_k 4 --context_memory_frames 5 --use_rt_relative --height 352 --width 640 \
  --dataset_repeat 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_workers 16 \
  --model_paths "${model_paths}" --learning_rate 5e-5 --num_epochs 1 --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}" \
  --output_path "${output_base}_videossm_hybrid" --trainable_models dit --ckpt_interval "${CKPT_INTERVAL:-1000}" --save_full_model \
  --wandb_run_name "memory_baseline_videossm_hybrid" \
  --enable_context_memory --training_mode context --condition_t2v_ratio 0.10 --condition_i2v_ratio 0.10 --cfg_target_only \
  --train_cam_pose --add_action_attn --action_use_temporal_attention \
  --spike_threshold "${SPIKE_THRESHOLD:-15.0}" \
  --use_moc --moc_temperature 1.0 \
  --use_videossm_hybrid --videossm_every_n_blocks 4 --videossm_kernel_size 3 --videossm_expand 2 \
  --timestep_shift "${TIMESTEP_SHIFT:-15}" --enable_video_sampling --sampling_atomic_left_right --sampling_interval_steps "${SAMPLING_INTERVAL_STEPS:-1000}" \
  --sampling_num_inference_steps "${SAMPLING_NUM_INFERENCE_STEPS:-50}" --sampling_negative_prompt "oversaturated colors, overexposed, static, blurry details" \
  --sampling_height "${SAMPLING_HEIGHT:-352}" --sampling_width "${SAMPLING_WIDTH:-640}" --sampling_num_frames "${SAMPLING_NUM_FRAMES:-81}" --samples_per_epoch 0 --sampling_action_path "${sampling_action_path}" \
  2>&1 | tee "${LOG_DIR}/memory_baseline_videossm_hybrid_$(date +%Y%m%d_%H%M%S).log"

