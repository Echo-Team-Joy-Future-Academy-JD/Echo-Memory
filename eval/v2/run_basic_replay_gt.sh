#!/bin/bash
# Basic capability（可选）：单条 GT 轨迹 replay。质量评测主路径见 static 中 long_horizon_gt_replay（多 chunk）。
# 默认 NUM_CHUNKS 与 NUM_CHUNKS_LONG（默认 3）对齐；单 chunk 不足以表征跨 chunk 记忆/误差累积。
# 与 eval_v2 static 对齐：PYTHONPATH、ctx、CAMERA_INJECT_MODE、MEM_ARGS。
# VIDEO_NAME:
#   - 显式设置时，要求该 video 在 DATASET/jsons 下有可用 GT pose（按 START_FRAME/NUM_CHUNKS/CHUNK_FRAMES 检查）
#   - 未设置时（或 VIDEO_NAME=AUTO），优先 AncientTempleEnv_0；不可用则自动回退到首个可用 video
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${EVAL_DIR}/../.." && pwd)"
ENV_DIR="${REPO_ROOT}/env"
# shellcheck disable=SC1091
[ -f "${REPO_ROOT}/env/eval_infer_alignment_env.sh" ] && source "${REPO_ROOT}/env/eval_infer_alignment_env.sh"
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT="${CKPT:?Set CKPT=/path/to/epoch-0.safetensors}"
CKPT_DIR="$(dirname "${CKPT}")"
DATASET="/threed-code/yorenchen/data/echo-memory/Context-as-Memory-Dataset/"

# --- Early path sanity (before any heavy Python) ---
if [ ! -d "${DATASET}" ]; then
  echo "[replay_gt] ERROR: DATASET 不是目录: ${DATASET}" >&2
  exit 1
fi
if [ ! -d "${DATASET}/jsons" ]; then
  echo "[replay_gt] ERROR: 缺少 DATASET/jsons: ${DATASET}/jsons" >&2
  exit 1
fi
echo "[replay_gt] DATASET=$(cd "${DATASET}" && pwd)"

VIDEO_NAME="${VIDEO_NAME:-AUTO}"
START_FRAME="${START_FRAME:-0}"
NUM_CHUNKS="${NUM_CHUNKS:-${NUM_CHUNKS_LONG:-3}}"
CHUNK_FRAMES_EFFECTIVE="${CHUNK_FRAMES:-81}"

_resolve_video_name() {
  local wanted="$1"
  DATASET="${DATASET}" \
  EVAL_DIR="${EVAL_DIR}" \
  VIDEO_NAME_IN="${wanted}" \
  START_FRAME="${START_FRAME}" \
  NUM_CHUNKS="${NUM_CHUNKS}" \
  CHUNK_FRAMES="${CHUNK_FRAMES_EFFECTIVE}" \
  python3 - <<'PY'
import os
import sys

dataset = os.environ["DATASET"]
eval_dir = os.environ["EVAL_DIR"]
wanted = os.environ.get("VIDEO_NAME_IN", "").strip()
start = int(os.environ.get("START_FRAME", "0"))
num_chunks = int(os.environ.get("NUM_CHUNKS", "1"))
chunk_frames = int(os.environ.get("CHUNK_FRAMES", "81"))

# 轻量解析 VIDEO_NAME：勿 import run_replay_loop_two_chunk（会拉 torch/train/flash_attn）
sys.path.insert(0, os.path.join(eval_dir, "basic"))
from gt_pose_minimal import build_gt_trajectory_actions  # noqa: E402

def valid(vn: str) -> bool:
    for ch in range(num_chunks):
        seg_start = start + ch * chunk_frames
        if build_gt_trajectory_actions(dataset, vn, seg_start, chunk_frames) is None:
            return False
    return True

if wanted and wanted.upper() != "AUTO":
    print(wanted if valid(wanted) else "")
    raise SystemExit(0)

cands = []
jsons_dir = os.path.join(dataset, "jsons")
if os.path.isdir(jsons_dir):
    for n in sorted(os.listdir(jsons_dir)):
        if n.endswith(".json"):
            cands.append(os.path.splitext(n)[0])
preferred = "AncientTempleEnv_0"
if preferred in cands:
    cands.remove(preferred)
    cands = [preferred] + cands
for vn in cands:
    if valid(vn):
        print(vn)
        raise SystemExit(0)
print("")
PY
}

RESOLVED_VIDEO_NAME="$(_resolve_video_name "${VIDEO_NAME}")"
if [ -z "${RESOLVED_VIDEO_NAME}" ]; then
  if [ -n "${VIDEO_NAME}" ] && [ "${VIDEO_NAME}" != "AUTO" ]; then
    echo "[replay_gt] ERROR: VIDEO_NAME=${VIDEO_NAME} 不可用：缺少 GT actions（json 或帧段不足）。" >&2
  else
    echo "[replay_gt] ERROR: 未找到可用 video（DATASET/jsons 下无可用 GT actions）。" >&2
  fi
  exit 1
fi
if [ "${VIDEO_NAME}" != "${RESOLVED_VIDEO_NAME}" ]; then
  echo "[replay_gt] 自动回退 VIDEO_NAME: ${VIDEO_NAME} -> ${RESOLVED_VIDEO_NAME}"
fi
VIDEO_NAME="${RESOLVED_VIDEO_NAME}"

python3 "${EVAL_DIR}/basic/check_dataset_gt_for_replay.py" \
  --dataset "${DATASET}" \
  --video "${VIDEO_NAME}" \
  --start_frame "${START_FRAME}" \
  --num_chunks "${NUM_CHUNKS}" \
  --chunk_frames "${CHUNK_FRAMES_EFFECTIVE}" || exit 1

OUT_ROOT="${OUT_ROOT:-${CKPT_DIR}/evals_v2/basic}"
OUT_DIR="${OUT_ROOT}/replay_gt/${VIDEO_NAME}_start${START_FRAME}"
mkdir -p "${OUT_DIR}"

eval "$(python3 "${ENV_DIR}/memory_baseline_runtime.py" bash-export "${CKPT}")"
_default_ctx=1
[[ "${CKPT}" =~ (ctx_20|context_k20|ctx20) ]] && _default_ctx=20
[[ "${CKPT}" =~ (ctx_5|context_k5|ctx5) ]] && _default_ctx=5
[ -n "${CONTEXT_FRAMES_MEM_OVERRIDE:-}" ] && _default_ctx="${CONTEXT_FRAMES_MEM_OVERRIDE}"
CONTEXT_FRAMES_EFFECTIVE="${CONTEXT_FRAMES:-$_default_ctx}"

python3 "${EVAL_DIR}/basic/replay_gt_error.py" \
  --ckpt "${CKPT}" \
  --dataset_base "${DATASET}" \
  --video_name "${VIDEO_NAME}" \
  --start_frame "${START_FRAME}" \
  --num_chunks "${NUM_CHUNKS}" \
  --chunk_frames "${CHUNK_FRAMES_EFFECTIVE}" \
  --context_frames "${CONTEXT_FRAMES_EFFECTIVE}" \
  --sigma_shift "${SIGMA_SHIFT:-5}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
  --cfg_scale "${CFG_SCALE:-5.0}" \
  --seed "${SEED:-42}" \
  --output_dir "${OUT_DIR}" \
  --write_csv

echo "Done. basic replay_gt output: ${OUT_DIR}"

