#!/bin/bash
# =============================================================================
# 推理环境与训练对齐（务必在跑 ep0 / eval_v2 / replay_gt 前 source）
#
# 对应训练侧：
#   - train/memory_baselines_basic/common_env.sh
#   - train/ctx_5_20_per_frame_vae/common_env.sh 等
#
# 说明：
#   - CONTEXT_POSITION=suffix：与训练 CONTEXT_POSITION 一致；run_one_chunk 内也会显式传 context_position=suffix
#   - USE_CONCATENATION_INFERENCE：WanVideoPipeline 默认即为 true，此处显式导出避免子进程/调度环境缺失
#   - USE_RT_RELATIVE：训练 common_env 开启，评测脚本保持一致（供后续扩展或外部工具读取）
#
# 覆盖示例：CONTEXT_POSITION=prefix bash run_eval_...
# CAMERA_INJECT_MODE：默认 pre_qkv；若你的 ckpt 为 pre_norm/post 训练请显式覆盖。
# =============================================================================
: "${CONTEXT_POSITION:=suffix}"
: "${USE_CONCATENATION_INFERENCE:=true}"
: "${USE_RT_RELATIVE:=true}"
# 与 train/memory_baselines_basic/*.sh 一致：默认 pre_qkv（避免 replay_gt / combo 仅因 ckpt 路径无子串而落到 pre_norm 导致 action 注入错位）
: "${CAMERA_INJECT_MODE:=pre_qkv}"
export CONTEXT_POSITION USE_CONCATENATION_INFERENCE USE_RT_RELATIVE CAMERA_INJECT_MODE

if [ "${PRINT_EVAL_ALIGN_SUMMARY:-1}" = "1" ]; then
  echo "[eval_align_env] CONTEXT_POSITION=${CONTEXT_POSITION} USE_CONCATENATION_INFERENCE=${USE_CONCATENATION_INFERENCE} USE_RT_RELATIVE=${USE_RT_RELATIVE} CAMERA_INJECT_MODE=${CAMERA_INJECT_MODE}"
fi
