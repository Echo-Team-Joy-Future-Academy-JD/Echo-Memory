#!/bin/bash
# Run all memory baseline inferences sequentially.
# Requires: CKPT_DIR pointing to a directory with per-row checkpoint folders
# (same layout as HF repo: spatial_mem/epoch-0.safetensors, etc.)
#
# Usage:
#   CKPT_DIR=./ckpts bash inference/memory_baselines_basic/run_infer_all.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${CKPT_DIR:?ERROR: set CKPT_DIR to your checkpoints directory}"

declare -A CKPT_MAP=(
  [run_infer_no_memory.sh]="no_memory_extra_two_chunk"
  [run_infer_framepack_weight.sh]="framepack_weight_only"
  [run_infer_framepack_len_r2.sh]="framepack_lencompress_r2"
  [run_infer_framepack_len_r4.sh]="framepack_lencompress_r4"
  [run_infer_framepack_hybrid_r2.sh]="framepack_hybrid_r2_weight_two_chunk"
  [run_infer_framepack_hybrid_r4.sh]="framepack_hybrid_r4_weight_two_chunk"
  [run_infer_spatial_mem.sh]="spatial_mem"
  [run_infer_spatial_concat_text.sh]="spatial_concat_text_two_chunk"
  [run_infer_spatial_inject_none.sh]="spatial_inject_none_two_chunk"
  [run_infer_spatial_cross_attn_readout.sh]="spatial_cross_attn_readout_two_chunk"
  [run_infer_videossm_hybrid.sh]="videossm_hybrid"
  [run_infer_block_wise_ssm.sh]="block_wise_ssm_two_chunk"
)

for script in "${SCRIPT_DIR}"/run_infer_*.sh; do
  name="$(basename "${script}")"
  [[ "${name}" == "run_infer_all.sh" ]] && continue
  ckpt_folder="${CKPT_MAP[${name}]:-}"
  if [ -z "${ckpt_folder}" ]; then
    echo "[run_infer_all] SKIP: no checkpoint mapping for ${name}"
    continue
  fi
  ckpt_file="${CKPT_DIR}/${ckpt_folder}/epoch-0.safetensors"
  if [ ! -f "${ckpt_file}" ]; then
    echo "[run_infer_all] SKIP: ${ckpt_file} not found"
    continue
  fi
  echo "========================================"
  echo "[run_infer_all] Running ${name} with ${ckpt_file}"
  echo "========================================"
  CKPT="${ckpt_file}" bash "${script}"
done

echo "[run_infer_all] Done."
