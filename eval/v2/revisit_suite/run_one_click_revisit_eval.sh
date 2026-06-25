#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${ENV_DIR:-${REPO_ROOT}/env}:${EVAL_DIR}:${PYTHONPATH:-}"
export WAN_BASE_MODEL="${WAN_BASE_MODEL:?Set WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B}"
export DATASET_BASE="${DATASET_BASE:-${DATASET_BASE_PATH:-${REPO_ROOT}/data/Context-as-Memory-Dataset}}"
export DATASET_METADATA="${DATASET_METADATA:-${DATASET_BASE}/metadata_full.csv}"
export OOD_DIR="${OOD_DIR:-${REPO_ROOT}/assets/opendomain_revisit}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_ROOT}/outputs}"

EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/eval_outputs/revisit_suite_$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${MANIFEST:-${EVAL_ROOT}/manifest.json}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-}"
TRAIN_LIMIT="${TRAIN_LIMIT:-8}"
OOD_LIMIT="${OOD_LIMIT:-8}"
MODES="${MODES:-rot180_4chunk,rot360_8chunk,right45_return_2chunk}"
PHASE="${PHASE:-all}"  # all | stage1 | vlm
SUMMARY_DIR_NAME="${SUMMARY_DIR_NAME:-revisit_eval_summary}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"

mkdir -p "${EVAL_ROOT}"

python3 "${SCRIPT_DIR}/prepare_eval_manifest.py" \
  --outputs-root "${OUTPUTS_ROOT}" \
  --dataset-base "${DATASET_BASE}" \
  --metadata "${DATASET_METADATA}" \
  --ood-dir "${OOD_DIR}" \
  --train-limit "${TRAIN_LIMIT}" \
  --ood-limit "${OOD_LIMIT}" \
  --out "${MANIFEST}" \
  ${INCLUDE_DYNMEMBENCH:+--include-dynmembench}

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "stage1" ]; then
  if [ -n "${GPU_IDS}" ]; then
    IFS=',' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
    NUM_GPUS="${#GPU_ID_LIST[@]}"
  else
    GPU_ID_LIST=()
    for rank in $(seq 0 $((NUM_GPUS - 1))); do
      GPU_ID_LIST+=("${rank}")
    done
  fi
  IFS=',' read -r -a MODE_LIST <<< "${MODES}"
  TOTAL_STAGE1_CASES="$(python3 - "${MANIFEST}" "${#MODE_LIST[@]}" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(len(m.get("ckpts", [])) * len(m.get("samples", [])) * int(sys.argv[2]))
PY
)"
  pids=()
  for rank in $(seq 0 $((NUM_GPUS - 1))); do
    (
      export CUDA_VISIBLE_DEVICES="${GPU_ID_LIST[$rank]}"
      python3 "${SCRIPT_DIR}/run_revisit_stage1.py" \
        --manifest "${MANIFEST}" \
        --output-root "${EVAL_ROOT}/stage1" \
        --rank "${rank}" \
        --world-size "${NUM_GPUS}" \
        --modes "${MODES}" \
        --base-model "${WAN_BASE_MODEL}" \
        --sigma-shift "${SIGMA_SHIFT:-15}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS:-50}" \
        ${FORCE_STAGE1:+--force}
    ) > "${EVAL_ROOT}/stage1_rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  if [ "${SHOW_PROGRESS}" = "1" ]; then
    while true; do
      running=0
      for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          running=$((running + 1))
        fi
      done
      completed="$(python3 - "${EVAL_ROOT}/stage1" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
print(sum(1 for _ in root.rglob("stage1_metrics.json")) if root.exists() else 0)
PY
)"
      echo "[one_click_revisit_eval][stage1] progress ${completed}/${TOTAL_STAGE1_CASES} cases, running_ranks=${running}/${NUM_GPUS}, logs=${EVAL_ROOT}/stage1_rank*.log"
      if [ "${running}" -eq 0 ]; then
        break
      fi
      sleep "${PROGRESS_INTERVAL}"
    done
  fi
  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if [ "${failed}" -ne 0 ]; then
    echo "[one_click_revisit_eval][ERROR] one or more stage1 ranks failed. Check ${EVAL_ROOT}/stage1_rank*.log" >&2
    exit 1
  fi
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "vlm" ]; then
  python3 "${SCRIPT_DIR}/qwen_vlm_score.py" \
    --stage1-root "${EVAL_ROOT}/stage1" \
    --api-base "${VLM_API_BASE:-http://127.0.0.1:8000/v1}" \
    --model "${VLM_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct}" \
    ${VLM_DRY_RUN:+--dry-run} \
    ${FORCE_VLM:+--force}
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "stage1" ] || [ "${PHASE}" = "summary" ] || [ "${PHASE}" = "vlm" ]; then
  python3 "${SCRIPT_DIR}/summarize_revisit_results.py" \
    --stage1-root "${EVAL_ROOT}/stage1" \
    --manifest "${MANIFEST}" \
    --out-dir-name "${SUMMARY_DIR_NAME}"
fi

echo "[one_click_revisit_eval] done: ${EVAL_ROOT}"
