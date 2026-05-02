#!/bin/bash
# pre_qkv 系列：action 在第一个 layer norm 之后、3D self-attn 之前与 frame 交互后输入。ctx=1（1 帧 context，无需 per_frame_vae）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
accelerate launch src/model_training/train.py \
  --dataset_base_path "${dataset_base_path}" --dataset_metadata_path "${dataset_base_path}/${METADATA_NAME}" \
  --enable_fov_retrieval --fov_top_k 0 --context_memory_frames 1 --use_rt_relative --height 352 --width 640 \
  --dataset_repeat 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_workers 16 \
  --model_paths "${model_paths}" --learning_rate 5e-5 --num_epochs 1 --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}" \
  --output_path "${output_base}_merged_cam_ctx_1_noise_5_atomic_cam_inject_pre_qkv_seperate_t_r" --trainable_models dit --ckpt_interval 5000 --save_full_model \
  --wandb_run_name "exp1_4_4_ctx_1_noise_5_atomic_cam_inject_pre_qkv" \
  --enable_context_memory --training_mode context --context_drop_prob 0.1 --cfg_target_only \
  --train_cam_pose --add_action_attn --action_use_temporal_attention \
  --timestep_shift 5 --enable_video_sampling --sampling_atomic_left_right --sampling_interval_steps 1000 \
  --verify_high_noise_first_steps 5 --verify_ckpt_step 5 \
  --sampling_num_inference_steps 50 --sampling_negative_prompt "oversaturated colors, overexposed, static, blurry details" \
  --sampling_height 352 --sampling_width 640 --sampling_num_frames 81 --samples_per_epoch 0 --sampling_action_path "${sampling_action_path}" \
  2>&1 | tee "${LOG_DIR}/exp1_4_4_ctx_1_pre_qkv_$(date +%Y%m%d_%H%M%S).log"
