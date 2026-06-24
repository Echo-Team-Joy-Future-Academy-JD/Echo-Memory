#!/bin/bash
# Run all dynamic SpatialVID inference rows from CKPT_DIR.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CKPT_DIR:?ERROR: set CKPT_DIR to a directory with dynamic row checkpoints}"

run_one() {
  local row="$1"
  local script="$2"
  CKPT="${CKPT_DIR}/${row}/epoch-0.safetensors" bash "${SCRIPT_DIR}/${script}"
}

run_one ctx1 run_infer_dyn_ctx1.sh
run_one ctx5 run_infer_dyn_ctx5.sh
run_one ctx20 run_infer_dyn_ctx20.sh
run_one spatial_mem run_infer_dyn_spatial_mem.sh
run_one block_wise_ssm run_infer_dyn_block_wise_ssm.sh
run_one videossm_hybrid run_infer_dyn_videossm_hybrid.sh
