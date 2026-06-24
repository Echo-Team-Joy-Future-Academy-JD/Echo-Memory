#!/bin/bash
# Static consistency (v2): loop closure + composite-action revisit.
#
# Output root defaults to <ckpt_dir>/evals_v2/static_consistency.
# 与 run_evals_ep0_pre_qkv_rt_merge.sh 对齐：PYTHONPATH、ctx 帧数推断、metadata 采样、CAMERA_INJECT_MODE、MEM_ARGS、no_camera_encoder_separate_t_r。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="${SCRIPT_DIR}"
REPO_ROOT="${REPO_ROOT:-$(cd "${EVAL_DIR}/../.." && pwd)}"
ENV_DIR="${REPO_ROOT}/env"
# shellcheck disable=SC1091
[ -f "${REPO_ROOT}/env/eval_infer_alignment_env.sh" ] && source "${REPO_ROOT}/env/eval_infer_alignment_env.sh"

cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Ensure OpenCV runtime dependency exists (libGL.so.1), otherwise mp4 metric reading may fail.
if ! python3 - <<'PY'
import ctypes
ctypes.CDLL("libGL.so.1")
print("ok")
PY
then
  echo "[eval_v2] FATAL: libGL.so.1 missing. Install libgl1-mesa-glx or use a headless OpenCV build." >&2
  exit 2
fi

CKPT="${CKPT:?Set CKPT=/path/to/epoch-0.safetensors}"
CKPT_DIR="$(dirname "${CKPT}")"
if [ ! -f "${CKPT}" ]; then
  echo "[eval_v2] FATAL: CKPT 不是可读文件: ${CKPT}" >&2
  exit 1
fi
DATASET="${DATASET:-${DATASET_BASE_PATH:-${REPO_ROOT}/data/Context-as-Memory-Dataset}}"
if [ ! -d "${DATASET}" ]; then
  echo "[eval_v2] FATAL: DATASET 不是目录: ${DATASET}" >&2
  exit 1
fi
export CAMERA_INJECT_MODE="${CAMERA_INJECT_MODE:-pre_qkv}"

METADATA_NAME="${METADATA_NAME:-metadata_full.csv}"
METADATA_PATH="${METADATA_PATH:-${DATASET}/${METADATA_NAME}}"
SAMPLING_ACTION_DIR="${SAMPLING_ACTION_DIR:-${ENV_DIR}}"
LOOP_EXTRA_META=()
if [ -f "${METADATA_PATH}" ]; then
  LOOP_EXTRA_META=(--dataset_metadata_path "${METADATA_PATH}")
else
  echo "[eval_v2] WARN: METADATA_PATH 不存在，loop 将用 frames 随机采样（与 ep0 有 metadata 时不一致）: ${METADATA_PATH}" >&2
fi

EVALS_ROOT="${EVALS_ROOT:-${CKPT_DIR}/evals_v2/static_consistency}"
IN_DOMAIN="${EVALS_ROOT}/in_domain"
OPEN_DOMAIN_ROOT="${EVALS_ROOT}/open_domain"
mkdir -p "${IN_DOMAIN}"

# Infer memory baseline runtime flags from CKPT path (single source: memory_baseline_runtime.py).
MEM_ARGS=()
eval "$(python3 "${ENV_DIR}/memory_baseline_runtime.py" bash-export "${CKPT}")"
# Infer context length from both legacy ctx_* names and released HF context_k* folders.
_default_ctx=1
[[ "${CKPT}" =~ (ctx_20|context_k20|ctx20) ]] && _default_ctx=20
[[ "${CKPT}" =~ (ctx_5|context_k5|ctx5) ]] && _default_ctx=5
[ -n "${CONTEXT_FRAMES_MEM_OVERRIDE:-}" ] && _default_ctx="${CONTEXT_FRAMES_MEM_OVERRIDE}"
CONTEXT_FRAMES_EFFECTIVE="${CONTEXT_FRAMES:-$_default_ctx}"
SEED_EFFECTIVE="${SEED:-42}"

echo "[eval_v2] CONTEXT_FRAMES_EFFECTIVE=${CONTEXT_FRAMES_EFFECTIVE} CAMERA_INJECT_MODE=${CAMERA_INJECT_MODE} SAMPLING_ACTION_DIR=${SAMPLING_ACTION_DIR}"
echo "[eval_v2] MEM_ARGS=${MEM_ARGS[*]:-none}"
[ "${#LOOP_EXTRA_META[@]}" -gt 0 ] && echo "[eval_v2] loop dataset_metadata_path=${METADATA_PATH}"

# Fail-fast：MEM_ARGS 须能被 multiview 的 argparse.REMAINDER 吞掉（避免跑完全部 in-domain 后才报 unrecognized arguments）
python3 "${EVAL_DIR}/tools/verify_static_eval_prereqs.py" remainder-mem-args "${MEM_ARGS[@]}"
# Wan2.1 底座权重（combo / loop 均依赖）
python3 "${EVAL_DIR}/tools/verify_static_eval_prereqs.py" wan-base

# ---------- (A) Loop closure (in-domain) ----------
python3 "${ENV_DIR}/run_replay_loop_two_chunk.py" \
  --ckpt "${CKPT}" \
  --context_frames "${CONTEXT_FRAMES_EFFECTIVE}" \
  --sampling_action_dir "${SAMPLING_ACTION_DIR}" \
  --dataset_base "${DATASET}" \
  "${LOOP_EXTRA_META[@]}" \
  --output_dir "${IN_DOMAIN}/loop_closure" \
  --num_samples "${NUM_SAMPLES_LOOP:-8}" \
  --seed "${SEED_EFFECTIVE}" \
  --sigma_shift "${SIGMA_SHIFT:-5}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
  --cfg_scale "${CFG_SCALE:-5.0}" \
  "${MEM_ARGS[@]}"

# ---------- (B) Symmetric random-action combo revisit (in-domain: training prompt + first frame) ----------
# 随机幅度、对称闭环：左转→直行→后退→右转（与固定 45° 同结构），最后 revisit 到首帧视角（像素上首末帧 MSE 见 revisit_closure_metrics.json）。
# 可选 STATIC_USE_OPEN_DOMAIN_COMBO=1：回退到单条 open-domain（FIRST_FRAME_IMG + PROMPT）。
CHUNK_FRAMES_EFFECTIVE="${CHUNK_FRAMES:-81}"
NUM_CHUNKS_LONG="${NUM_CHUNKS_LONG:-3}"
MIN_SPAN=$(( CHUNK_FRAMES_EFFECTIVE * NUM_CHUNKS_LONG ))
NUM_SAMPLES_STATIC="${NUM_SAMPLES_STATIC:-6}"
COMBO_BASE="${IN_DOMAIN}/action_combos_random_symmetric"
COMBO_OUT="${IN_DOMAIN}/combo_revisit_in_domain"
mkdir -p "${COMBO_BASE}" "${COMBO_OUT}"

if [ "${STATIC_USE_OPEN_DOMAIN_COMBO:-0}" = "1" ]; then
  echo "[eval_v2] STATIC_USE_OPEN_DOMAIN_COMBO=1 -> legacy single combo + external first frame"
  COMBO_DIR="${IN_DOMAIN}/action_combo_rot_trans_rev"
  python3 "${EVAL_DIR}/actions/build_action_combo.py" \
    --exp_dir "${ENV_DIR}" \
    --out_dir "${COMBO_DIR}" \
    --chunk_frames "${CHUNK_FRAMES_EFFECTIVE}" \
    --translation_delta "${TRANSLATION_DELTA:-0.1}" >/dev/null
  FIRST_FRAME_IMG="${FIRST_FRAME_IMG:-${REPO_ROOT}/assets/first_frame.png}"
    python3 "${EVAL_DIR}/static/run_combo_revisit_fixed_first.py" \
      --ckpt "${CKPT}" \
      --first_frame_image "${FIRST_FRAME_IMG}" \
      --output_dir "${IN_DOMAIN}/combo_revisit_fixed_first" \
      --prompt "${PROMPT:-A scene.}" \
      --action_combo_dir "${COMBO_DIR}" \
      --context_frames "${CONTEXT_FRAMES_EFFECTIVE}" \
      --chunk_frames "${CHUNK_FRAMES_EFFECTIVE}" \
      --sigma_shift "${SIGMA_SHIFT:-5}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
      --cfg_scale "${CFG_SCALE:-5.0}" \
      --seed "${SEED_EFFECTIVE}" \
      "${MEM_ARGS[@]}"
else
  SAMPLES_TSV="${IN_DOMAIN}/_static_eval_samples.tsv"
  python3 "${EVAL_DIR}/tools/emit_dataset_samples.py" \
    --dataset "${DATASET}" \
    --num_samples "${NUM_SAMPLES_STATIC}" \
    --min_frames "${MIN_SPAN}" \
    --seed "${SEED_EFFECTIVE}" > "${SAMPLES_TSV}"
  _idx=0
  _seed_base="${SEED_EFFECTIVE}"
  while IFS=$'\t' read -r _vn _st; do
    [ -z "${_vn:-}" ] && continue
    _combo_sub="${COMBO_BASE}/${_vn}_start${_st}"
    mkdir -p "${_combo_sub}"
    python3 "${EVAL_DIR}/actions/build_action_combo.py" \
      --random_symmetric \
      --combo_seed "$((_seed_base + _idx))" \
      --out_dir "${_combo_sub}" \
      --chunk_frames "${CHUNK_FRAMES_EFFECTIVE}" \
      --yaw_min "${COMBO_YAW_MIN:-20}" \
      --yaw_max "${COMBO_YAW_MAX:-55}" \
      --translation_min "${COMBO_TRANS_MIN:-0.05}" \
      --translation_max "${COMBO_TRANS_MAX:-0.18}" >/dev/null
    python3 "${EVAL_DIR}/static/run_combo_revisit_fixed_first.py" \
      --ckpt "${CKPT}" \
      --dataset_base "${DATASET}" \
      --video_name "${_vn}" \
      --start_frame "${_st}" \
      --output_dir "${COMBO_OUT}/${_vn}_start${_st}" \
      --action_combo_dir "${_combo_sub}" \
      --context_frames "${CONTEXT_FRAMES_EFFECTIVE}" \
      --chunk_frames "${CHUNK_FRAMES_EFFECTIVE}" \
      --sigma_shift "${SIGMA_SHIFT:-5}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
      --cfg_scale "${CFG_SCALE:-5.0}" \
      --seed "${SEED_EFFECTIVE}" \
      "${MEM_ARGS[@]}"
    _idx=$((_idx + 1))
  done < "${SAMPLES_TSV}"
  python3 "${EVAL_DIR}/metrics/aggregate_combo_closure_metrics.py" \
    --root "${COMBO_OUT}" \
    --output_json "${EVALS_ROOT}/metrics/combo_closure_summary.json"
fi

if [ "${RUN_GEOMETRY_DIAG:-0}" = "1" ] && [ "${STATIC_USE_OPEN_DOMAIN_COMBO:-0}" = "1" ]; then
  echo "[eval_v2] WARN: RUN_GEOMETRY_DIAG=1 但 STATIC_USE_OPEN_DOMAIN_COMBO=1 时不会生成 _static_eval_samples.tsv / long_horizon_gt_replay，点云段 (F) 无输入。若要 3D 诊断请关掉 STATIC_USE_OPEN_DOMAIN_COMBO 或设 RUN_BASIC_REPLAY_GT=1 后自行对齐 mp4 与数据集 pose。" >&2
fi

# ---------- (D) Long-horizon GT trajectory replay (>=3 chunks), visual quality as primary ----------
LONG_ROOT="${IN_DOMAIN}/long_horizon_gt_replay"
mkdir -p "${LONG_ROOT}"
if [ "${STATIC_USE_OPEN_DOMAIN_COMBO:-0}" != "1" ] && [ -f "${IN_DOMAIN}/_static_eval_samples.tsv" ]; then
  while IFS=$'\t' read -r _vn _st; do
    [ -z "${_vn:-}" ] && continue
    _out="${LONG_ROOT}/${_vn}_start${_st}"
    mkdir -p "${_out}"
    python3 "${EVAL_DIR}/basic/check_dataset_gt_for_replay.py" \
      --dataset "${DATASET}" \
      --video "${_vn}" \
      --start_frame "${_st}" \
      --num_chunks "${NUM_CHUNKS_LONG}" \
      --chunk_frames "${CHUNK_FRAMES_EFFECTIVE}" || continue
    python3 "${EVAL_DIR}/basic/replay_gt_error.py" \
      --ckpt "${CKPT}" \
      --dataset_base "${DATASET}" \
      --video_name "${_vn}" \
      --start_frame "${_st}" \
      --num_chunks "${NUM_CHUNKS_LONG}" \
      --chunk_frames "${CHUNK_FRAMES_EFFECTIVE}" \
      --context_frames "${CONTEXT_FRAMES_EFFECTIVE}" \
      --sigma_shift "${SIGMA_SHIFT:-5}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
      --cfg_scale "${CFG_SCALE:-5.0}" \
      --seed "${SEED_EFFECTIVE}" \
      --output_dir "${_out}" \
      --write_csv
  done < "${IN_DOMAIN}/_static_eval_samples.tsv"
  python3 "${EVAL_DIR}/metrics/aggregate_long_horizon_fid_fvd.py" \
    --root "${LONG_ROOT}" \
    --dataset_base "${DATASET}" \
    --device "${DEVICE:-cuda}" \
    --output_json "${EVALS_ROOT}/metrics/long_horizon_fid_fvd_summary.json"
  # 可选：生成序列相邻帧一致性（无 GT，仅表征生成序列是否时序平滑；与 replay_gt_metrics 对 GT 指标互补）
  if [ "${RUN_TEMPORAL_ADJ:-0}" = "1" ]; then
    python3 "${EVAL_DIR}/metrics/aggregate_long_horizon_temporal_adjacency.py" \
      --root "${LONG_ROOT}" \
      --output_json "${EVALS_ROOT}/metrics/long_horizon_temporal_adjacency_summary.json"
  fi
fi

# ---------- (C) Revisit metrics: randomly sample a subset and average ----------
# 说明：metrics 目前使用 first-vs-last 作为 revisit proxy；LPIPS 依赖可选。
METRIC_SAMPLE_NUM="${METRIC_SAMPLE_NUM:-5}"
WRITE_VIZ="${WRITE_VIZ:-1}"
python3 "${EVAL_DIR}/metrics/aggregate_revisit_metrics.py" \
  --evals_root "${IN_DOMAIN}" \
  --num_samples "${METRIC_SAMPLE_NUM}" \
  --seed "${SEED_EFFECTIVE}" \
  --device "${DEVICE:-cuda}" \
  --output_dir "${EVALS_ROOT}/metrics/revisit_subset" \
  $( [ "${WRITE_VIZ}" != "0" ] && echo "--write_viz" || true )

# ---------- (E) Optional: multiview revisit from edited first-frame list ----------
# 产物目录：open_domain/multiview_revisit/（与 in_domain 分离）
# 方式 1：MULTIVIEW_FIRSTFRAME_LIST=/path/to/views.txt（每行：图片绝对路径[\t prompt]）
# 方式 2：MULTIVIEW_FIRSTFRAME_DIR=/path/to/dir（扫描 png/jpg/webp，生成列表到 open_domain/）
if [ -n "${MULTIVIEW_FIRSTFRAME_DIR:-}" ] && [ -d "${MULTIVIEW_FIRSTFRAME_DIR}" ]; then
  mkdir -p "${OPEN_DOMAIN_ROOT}"
  MULTIVIEW_FIRSTFRAME_LIST="${OPEN_DOMAIN_ROOT}/_multiview_firstframes_from_dir.txt"
  _mv_prompt_args=()
  if [ -n "${MULTIVIEW_PROMPT:-}" ]; then
    _mv_prompt_args=(--prompt "${MULTIVIEW_PROMPT}")
  fi
  python3 "${EVAL_DIR}/tools/build_multiview_firstframe_list_from_dir.py" \
    --image_dir "${MULTIVIEW_FIRSTFRAME_DIR}" \
    --output "${MULTIVIEW_FIRSTFRAME_LIST}" \
    "${_mv_prompt_args[@]}"
fi
if [ -n "${MULTIVIEW_FIRSTFRAME_LIST:-}" ] && [ -f "${MULTIVIEW_FIRSTFRAME_LIST}" ]; then
  mkdir -p "${OPEN_DOMAIN_ROOT}"
  MV_OUT="${OPEN_DOMAIN_ROOT}/multiview_revisit"
  MV_ACTION_DIR="${MULTIVIEW_ACTION_DIR:-${COMBO_BASE}/multiview_shared_combo}"
  mkdir -p "${MV_ACTION_DIR}" "${MV_OUT}"
  python3 "${EVAL_DIR}/actions/build_action_combo.py" \
    --random_symmetric \
    --combo_seed "${SEED_EFFECTIVE}" \
    --out_dir "${MV_ACTION_DIR}" \
    --chunk_frames "${CHUNK_FRAMES_EFFECTIVE}" \
    --yaw_min "${COMBO_YAW_MIN:-20}" \
    --yaw_max "${COMBO_YAW_MAX:-55}" \
    --translation_min "${COMBO_TRANS_MIN:-0.05}" \
    --translation_max "${COMBO_TRANS_MAX:-0.18}" >/dev/null
  python3 "${EVAL_DIR}/tools/verify_static_eval_prereqs.py" multiview-preflight \
    --ckpt "${CKPT}" \
    --firstframe-list "${MULTIVIEW_FIRSTFRAME_LIST}" \
    --action-combo-dir "${MV_ACTION_DIR}" \
    --runner "${EVAL_DIR}/static/run_combo_revisit_fixed_first.py" \
    -- "${MEM_ARGS[@]}"
  python3 "${EVAL_DIR}/static/run_multiview_revisit_from_firstframes.py" \
    --ckpt "${CKPT}" \
    --firstframe_list "${MULTIVIEW_FIRSTFRAME_LIST}" \
    --action_combo_dir "${MV_ACTION_DIR}" \
    --output_root "${MV_OUT}" \
    --chunk_frames "${CHUNK_FRAMES_EFFECTIVE}" \
    --context_frames "${CONTEXT_FRAMES_EFFECTIVE}" \
    --sigma_shift "${SIGMA_SHIFT:-5}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
    --cfg_scale "${CFG_SCALE:-5.0}" \
    --seed "${SEED_EFFECTIVE}" \
    --extra_args "${MEM_ARGS[@]}"
  if [ "${RUN_MULTIVIEW_AGGREGATE_METRICS:-1}" = "1" ] && [ -d "${MV_OUT}" ]; then
    python3 "${EVAL_DIR}/metrics/aggregate_multiview_open_domain_metrics.py" \
      --multiview_root "${MV_OUT}" \
      --ref_view "${MULTIVIEW_REF_VIEW:-0}" \
      --device "${DEVICE:-cuda}" \
      --output_json "${MV_OUT}/multiview_closure_and_cross_view.json"
  fi
fi

# ---------- (F) Optional: offline point-cloud rendering diagnostics ----------
# GEOMETRY_DEPTH_MODE=pseudo|npy_dir|npz_dir|midas；外部深度目录 GEOMETRY_DEPTH_DIR（每帧 {abs_idx:04d}.npy/.npz）
# GEOMETRY_DEPTH_IS_INVERSE=1、GEOMETRY_DEPTH_KEY、GEOMETRY_DEPTH_SCALE、MIDAS_DEVICE 可选
if [ "${RUN_GEOMETRY_DIAG:-0}" = "1" ] && [ -f "${IN_DOMAIN}/_static_eval_samples.tsv" ]; then
  GEOM_ROOT="${EVALS_ROOT}/geometry_diagnostics"
  mkdir -p "${GEOM_ROOT}"
  _geom_depth_args=(--depth_mode "${GEOMETRY_DEPTH_MODE:-pseudo}")
  if [ -n "${GEOMETRY_DEPTH_DIR:-}" ]; then
    _geom_depth_args+=(--depth_dir "${GEOMETRY_DEPTH_DIR}")
  fi
  if [ -n "${GEOMETRY_DEPTH_KEY:-}" ]; then
    _geom_depth_args+=(--depth_key "${GEOMETRY_DEPTH_KEY}")
  fi
  if [ "${GEOMETRY_DEPTH_IS_INVERSE:-0}" = "1" ]; then
    _geom_depth_args+=(--depth_is_inverse)
  fi
  if [ -n "${GEOMETRY_DEPTH_SCALE:-}" ]; then
    _geom_depth_args+=(--depth_scale "${GEOMETRY_DEPTH_SCALE}")
  fi
  if [ -n "${MIDAS_DEVICE:-}" ]; then
    _geom_depth_args+=(--midas_device "${MIDAS_DEVICE}")
  fi
  _geom_count=0
  while IFS=$'\t' read -r _vn _st; do
    [ -z "${_vn:-}" ] && continue
    _run="${LONG_ROOT}/${_vn}_start${_st}"
    _gen="${_run}/replay_gt_gen_only.mp4"
    [ ! -f "${_gen}" ] && continue
    python3 "${EVAL_DIR}/geometry/render_multiview_pointcloud_offline.py" \
      --gen_video "${_gen}" \
      --dataset_base "${DATASET}" \
      --video_name "${_vn}" \
      --start_frame "${_st}" \
      --num_frames "${CHUNK_FRAMES_EFFECTIVE}" \
      --output_dir "${GEOM_ROOT}/${_vn}_start${_st}" \
      "${_geom_depth_args[@]}"
    _geom_count=$((_geom_count + 1))
  done < "${IN_DOMAIN}/_static_eval_samples.tsv"
  if [ "${_geom_count}" -eq 0 ]; then
    echo "[eval_v2] WARN: RUN_GEOMETRY_DIAG=1 但未渲染任何点云（${LONG_ROOT} 下缺少 replay_gt_gen_only.mp4）。长时程 (D) 可能未跑或全部样本被 skip。" >&2
  else
    echo "[eval_v2] geometry diagnostics: ${_geom_count} run(s) -> ${GEOM_ROOT}"
  fi
  python3 "${EVAL_DIR}/metrics/aggregate_geometry_consistency.py" \
    --root "${GEOM_ROOT}" \
    --output_json "${EVALS_ROOT}/metrics/geometry_consistency_summary.json"
fi

echo "Done. static_consistency root: ${EVALS_ROOT}"

