#!/bin/bash
# FramePack/FAR baseline: weight-only (no length compression)
# Optional 2-chunk train monitor: use --sampling_two_chunk_memory [--sampling_two_chunk_action_path ...] instead of
# --sampling_atomic_left_right when you want replay-aligned two-chunk videos (modes are mutually prioritized in ModelLogger).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
export CONTEXT_POSITION=prefix
accelerate launch src/model_training/train.py \
  --dataset_base_path "${dataset_base_path}" --dataset_metadata_path "${dataset_base_path}/${METADATA_NAME}" \
  --context_source prev_chunk_tail --context_memory_frames 81 --use_rt_relative --height 352 --width 640 \
  --dataset_repeat 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_workers 16 \
  --model_paths "${model_paths}" --learning_rate 5e-5 --num_epochs 1 --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}" \
  --output_path "${output_base}_framepack_weight_only" --trainable_models dit --ckpt_interval 5000 --save_full_model \
  --wandb_run_name "memory_baseline_framepack_weight_only" \
  --enable_context_memory --training_mode context --context_drop_prob 0.1 --cfg_target_only \
  --train_cam_pose --add_action_attn --action_use_temporal_attention \
  --use_moc --moc_temperature 1.0 \
  --use_framepack_memory --context_temporal_decay 0.9 --context_attention_weight 1.0 \
  --timestep_shift 5 --enable_video_sampling --sampling_atomic_left_right --sampling_interval_steps 1000 \
  --sampling_num_inference_steps 50 --sampling_negative_prompt "oversaturated colors, overexposed, static, blurry details" \
  --sampling_height 352 --sampling_width 640 --sampling_num_frames 81 --samples_per_epoch 0 --sampling_action_path "${sampling_action_path}" \
  2>&1 | tee "${LOG_DIR}/memory_baseline_framepack_weight_only_$(date +%Y%m%d_%H%M%S).log"

