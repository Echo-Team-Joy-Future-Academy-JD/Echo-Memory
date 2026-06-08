#!/usr/bin/env bash
# 8-GPU one-step training smoke test for Echo-Memory.
#
# Required:
#   WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
#   DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
#
# Optional:
#   ECHO_MEMORY_CONDA_ENV=storymem
#   NUM_PROCESSES=8
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f "${CONDA_SH:-}" ]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/pfs/weiyang/Miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/pfs/weiyang/Miniconda3/etc/profile.d/conda.sh"
fi

if [ -n "${ECHO_MEMORY_CONDA_ENV:-}" ]; then
  conda activate "$ECHO_MEMORY_CONDA_ENV"
elif [ -n "${CAM_CONDA_ENV:-}" ]; then
  conda activate "$CAM_CONDA_ENV"
fi

: "${WAN_BASE_MODEL:?Set WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B}"
: "${DATASET_BASE_PATH:?Set DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export WAN_BASE_MODEL DATASET_BASE_PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OUTPUT_BASE_ROOT="${OUTPUT_BASE_ROOT:-$ROOT/outputs/smoke_train}"
mkdir -p "$OUTPUT_BASE_ROOT" "$ROOT/logs"

MODEL_PATHS="[\"${WAN_BASE_MODEL}/diffusion_pytorch_model.safetensors\",\"${WAN_BASE_MODEL}/models_t5_umt5-xxl-enc-bf16.pth\",\"${WAN_BASE_MODEL}/Wan2.1_VAE.pth\"]"
OUT_PATH="${OUT_PATH:-$OUTPUT_BASE_ROOT/spatial_mem_one_step_$(date +%Y%m%d_%H%M%S)}"

accelerate launch --num_processes "${NUM_PROCESSES:-8}" src/model_training/train.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "${DATASET_BASE_PATH}/${METADATA_NAME:-metadata_full.csv}" \
  --context_source prev_chunk_tail \
  --context_memory_frames "${CONTEXT_FRAMES:-1}" \
  --use_rt_relative \
  --height "${HEIGHT:-128}" \
  --width "${WIDTH:-256}" \
  --dataset_repeat 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_workers "${NUM_WORKERS:-0}" \
  --model_paths "$MODEL_PATHS" \
  --learning_rate 5e-5 \
  --num_epochs 1 \
  --max_train_steps "${MAX_TRAIN_STEPS:-1}" \
  --progress_total_steps "${PROGRESS_TOTAL_STEPS:-30000}" \
  --remove_prefix_in_ckpt "${REMOVE_PREFIX_IN_CKPT:-pipe.dit.}" \
  --output_path "$OUT_PATH" \
  --trainable_models dit \
  --ckpt_interval "${CKPT_INTERVAL:-999999}" \
  --wandb_run_name "${WANDB_RUN_NAME:-smoke_spatial_mem_one_step}" \
  --enable_context_memory \
  --training_mode context \
  --condition_t2v_ratio 0.10 \
  --condition_i2v_ratio 0.10 \
  --cfg_target_only \
  --train_cam_pose \
  --add_action_attn \
  --action_use_temporal_attention \
  --spike_threshold "${SPIKE_THRESHOLD:-1000000}" \
  --use_moc \
  --moc_temperature 1.0 \
  --use_spatial_memory \
  --spatial_memory_tokens 64 \
  --timestep_shift "${TIMESTEP_SHIFT:-15}" \
  2>&1 | tee "$ROOT/logs/smoke_train_8gpu_$(date +%Y%m%d_%H%M%S).log"

echo "[smoke_train_8gpu] OK: $OUT_PATH"
