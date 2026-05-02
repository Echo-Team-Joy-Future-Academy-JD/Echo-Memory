#!/bin/bash
# 消融：无 action 的 ctx=5 训练，用于检查跳变（对比有 action 的 baseline）
# 仅 Context Memory，无 camera_encoder / action 注入
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
accelerate launch src/model_training/train.py \
  --dataset_base_path "${dataset_base_path}" --dataset_metadata_path "${dataset_base_path}/${METADATA_NAME}" \
  --enable_fov_retrieval --fov_top_k 4 --context_memory_frames 5 --context_per_frame_vae --use_rt_relative --height 352 --width 640 \
  --dataset_repeat 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_workers 16 \
  --model_paths "${model_paths}" --learning_rate 5e-5 --num_epochs 1 --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}" \
  --output_path "${output_base}_ctx_5_no_action_ablation" --trainable_models dit --ckpt_interval 5000 --save_full_model \
  --wandb_run_name "exp1_4_4_ctx_5_no_action_ablation" \
  --enable_context_memory --training_mode context --context_drop_prob 0.1 --cfg_target_only \
  --timestep_shift 5 --enable_video_sampling --sampling_four_prompts --sampling_interval_steps 1000 \
  --verify_high_noise_first_steps 5 --verify_ckpt_step 5 \
  --sampling_num_inference_steps 50 --sampling_negative_prompt "oversaturated colors, overexposed, static, blurry details" \
  --sampling_height 352 --sampling_width 640 --sampling_num_frames 81 --samples_per_epoch 0 \
  2>&1 | tee "${LOG_DIR}/exp1_4_4_ctx_5_no_action_ablation_$(date +%Y%m%d_%H%M%S).log"
