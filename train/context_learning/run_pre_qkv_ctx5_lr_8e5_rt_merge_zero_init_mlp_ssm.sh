#!/bin/bash
# GF-style 整层0初始化 + Diffusion Block SSM：ctx=5 + 合并RT + pre_qkv + 单层MLP + Linear全零初始化 + Block-wise SSM (Long-Context State-Space Video World Models, https://ryanpo.com/ssm_wm/)
# SSM 增强 diffusion block 的 temporal memory，参考 arXiv:2505.20171
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
accelerate launch src/model_training/train.py \
  --dataset_base_path "${dataset_base_path}" --dataset_metadata_path "${dataset_base_path}/${METADATA_NAME}" \
  --enable_fov_retrieval --fov_top_k 4 --context_memory_frames 5 --context_per_frame_vae --use_rt_relative --height 352 --width 640 \
  --dataset_repeat 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_workers 16 \
  --model_paths "${model_paths}" --learning_rate 8e-5 --num_epochs 1 --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}" \
  --output_path "${output_base}_merged_cam_ctx_5_pre_qkv_zero_init_mlp_ssm_lr_8e5" --trainable_models dit --ckpt_interval 5000 --save_full_model \
  --wandb_run_name "exp1_4_4_ctx_5_pre_qkv_zero_init_mlp_ssm_lr8e5" \
  --enable_context_memory --training_mode context --context_drop_prob 0.1 --cfg_target_only \
  --train_cam_pose --add_action_attn --action_use_temporal_attention \
  --use_block_wise_ssm --ssm_num_blocks_hint 21 --ssm_every_n_blocks 4 \
  --timestep_shift 5 --enable_video_sampling --sampling_atomic_left_right --sampling_interval_steps 1000 \
  --verify_high_noise_first_steps 5 --verify_ckpt_step 5 \
  --sampling_num_inference_steps 50 --sampling_negative_prompt "oversaturated colors, overexposed, static, blurry details" \
  --sampling_height 352 --sampling_width 640 --sampling_num_frames 81 --samples_per_epoch 0 --sampling_action_path "${sampling_action_path}" \
  2>&1 | tee "${LOG_DIR}/exp1_4_4_ctx_5_pre_qkv_zero_init_mlp_ssm_lr8e5_$(date +%Y%m%d_%H%M%S).log"
