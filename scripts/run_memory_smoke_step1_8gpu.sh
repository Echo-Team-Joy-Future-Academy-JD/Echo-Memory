#!/bin/bash
set -euo pipefail
cd /pfs/weiyang/Echo-Memory

export WAN_BASE_MODEL="${WAN_BASE_MODEL:-/pfs/weiyang/Wan2.1-T2V-1.3B}"
export DATASET_BASE_PATH="${DATASET_BASE_PATH:-/pfs/weiyang/Context-as-Memory-Dataset/Context-as-Memory-Dataset}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CONTEXT_POSITION=suffix
export USE_CONCATENATION_INFERENCE=true
export USE_RT_RELATIVE=true
TOKENIZER_PATH="${TOKENIZER_PATH:-${WAN_BASE_MODEL}/google/umt5-xxl}"

RUN_ROOT="${RUN_ROOT:-$PWD/outputs/smoke_memory_step1_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOG_ROOT"
SUMMARY="$RUN_ROOT/summary.tsv"
echo -e "name\tstatus\tlog" > "$SUMMARY"
SMOKE_TOTAL_STEPS="${SMOKE_TOTAL_STEPS:-30000}"

MODEL_PATHS="[\"${WAN_BASE_MODEL}/diffusion_pytorch_model.safetensors\",\"${WAN_BASE_MODEL}/models_t5_umt5-xxl-enc-bf16.pth\",\"${WAN_BASE_MODEL}/Wan2.1_VAE.pth\"]"
METADATA="${DATASET_BASE_PATH}/metadata_full.csv"

wait_for_8gpu_idle() {
  echo "[smoke] waiting for 8 GPUs to be idle; current training process will not be touched"
  while true; do
    if pgrep -af "accelerate launch src/model_training/train.py|python.*src/model_training/train.py" | grep -v "run_memory_smoke_step1_8gpu" >/dev/null 2>&1; then
      sleep 60
      continue
    fi
    busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '$1 > 10 {c++} END{print c+0}')
    if [ "${busy}" -eq 0 ]; then
      break
    fi
    sleep 60
  done
  echo "[smoke] GPUs look idle; starting sequence"
}

run_one() {
  local name="$1"; shift
  if [ -n "${SMOKE_CASES:-}" ] && [[ ",${SMOKE_CASES}," != *",${name},"* ]]; then
    echo "[smoke][$name] SKIP (SMOKE_CASES=${SMOKE_CASES})"
    return
  fi
  local port="$1"; shift
  local log="$LOG_ROOT/${name}.log"
  local out="$RUN_ROOT/${name}"
  echo "[smoke][$name] START $(date)" | tee "$log"
  set +e
  accelerate launch --num_processes 8 --main_process_port "$port" src/model_training/train.py \
    --dataset_base_path "$DATASET_BASE_PATH" --dataset_metadata_path "$METADATA" \
    --use_rt_relative --height 352 --width 640 \
    --dataset_repeat 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_workers 0 \
    --model_paths "$MODEL_PATHS" --tokenizer_path "$TOKENIZER_PATH" --learning_rate 5e-5 --num_epochs 1 --remove_prefix_in_ckpt pipe.dit. \
    --output_path "$out" --trainable_models dit --ckpt_interval 999999 --save_full_model \
    --enable_context_memory --training_mode context --context_drop_prob 0.1 --cfg_target_only \
    --train_cam_pose --add_action_attn --action_use_temporal_attention \
    --spike_threshold 999999 --max_train_steps 1 --progress_total_steps "$SMOKE_TOTAL_STEPS" \
    "$@" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -eq 0 ] && grep -q "\[SMOKE\] Reached max_train_steps=1" "$log"; then
    echo -e "${name}\tPASS\t${log}" | tee -a "$SUMMARY"
    echo "[smoke][$name] PASS"
  else
    echo -e "${name}\tFAIL(rc=${rc})\t${log}" | tee -a "$SUMMARY"
    echo "[smoke][$name] FAIL rc=$rc"
    exit 1
  fi
}

if [ "${DIRECT_START:-0}" = "1" ]; then
  echo "[smoke] DIRECT_START=1; starting immediately without waiting for idle GPUs"
else
  wait_for_8gpu_idle
fi

run_one no_memory 29601 --context_source replay --prev_chunk_frames 81 --context_memory_frames 5 --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one framepack_weight 29602 --context_source prev_chunk_tail --context_memory_frames 81 --use_framepack_memory --context_temporal_decay 0.9 --context_attention_weight 1.0 --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one framepack_len_r2 29603 --context_source prev_chunk_tail --context_memory_frames 81 --use_framepack_length_compress --framepack_ratio 2 --framepack_length_strategy packed_multiscale --framepack_multiscale_w2 0.25 --framepack_multiscale_w4 0.15 --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one framepack_len_r4 29604 --context_source prev_chunk_tail --context_memory_frames 81 --use_framepack_length_compress --framepack_ratio 4 --framepack_length_strategy packed_multiscale --framepack_multiscale_w2 0.25 --framepack_multiscale_w4 0.15 --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one framepack_hybrid_r2 29605 --context_source prev_chunk_tail --context_memory_frames 81 --use_framepack_memory --context_temporal_decay 0.95 --context_attention_weight 1.1 --use_framepack_length_compress --framepack_ratio 2 --framepack_length_strategy packed_multiscale --framepack_multiscale_w2 0.25 --framepack_multiscale_w4 0.15 --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one framepack_hybrid_r4 29606 --context_source prev_chunk_tail --context_memory_frames 81 --use_framepack_memory --context_temporal_decay 0.95 --context_attention_weight 1.1 --use_framepack_length_compress --framepack_ratio 4 --framepack_length_strategy packed_multiscale --framepack_multiscale_w2 0.25 --framepack_multiscale_w4 0.15 --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one spatial_mem 29607 --context_source prev_chunk_tail --context_memory_frames 1 --use_spatial_memory --spatial_memory_tokens 64 --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one spatial_inject_none 29608 --context_source prev_chunk_tail --context_memory_frames 1 --use_spatial_memory --spatial_memory_tokens 64 --spatial_memory_inject_mode none --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one spatial_concat_text 29609 --context_source prev_chunk_tail --context_memory_frames 1 --use_spatial_memory --spatial_memory_tokens 64 --spatial_memory_inject_mode concat_text --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one spatial_cross_attn_readout 29610 --context_source prev_chunk_tail --context_memory_frames 1 --use_spatial_memory --spatial_memory_tokens 64 --spatial_memory_inject_mode cross_attn_readout --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one videossm_hybrid 29611 --enable_fov_retrieval --fov_top_k 4 --context_memory_frames 5 --use_videossm_hybrid --videossm_every_n_blocks 4 --videossm_kernel_size 3 --videossm_expand 2 --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one block_wise_ssm 29612 --context_source replay --prev_chunk_frames 81 --context_memory_frames 5 --use_block_wise_ssm --ssm_every_n_blocks 4 --ssm_num_blocks_hint 21 --use_moc --moc_temperature 1.0 --timestep_shift 15
run_one context_k1 29613 --enable_fov_retrieval --fov_top_k 0 --context_memory_frames 1 --timestep_shift 5
run_one context_k5 29614 --enable_fov_retrieval --fov_top_k 4 --context_memory_frames 5 --context_per_frame_vae --timestep_shift 5
run_one context_k20 29615 --enable_fov_retrieval --fov_top_k 19 --context_memory_frames 20 --context_per_frame_vae --timestep_shift 5

echo "[smoke] ALL_PASS summary=$SUMMARY"
