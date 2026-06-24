#!/bin/bash
# Launch all public dynamic SpatialVID rows.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_dyn_ctx1.sh"
bash "${SCRIPT_DIR}/run_dyn_ctx5.sh"
bash "${SCRIPT_DIR}/run_dyn_ctx20.sh"
bash "${SCRIPT_DIR}/run_dyn_spatial_mem.sh"
bash "${SCRIPT_DIR}/run_dyn_block_wise_ssm.sh"
bash "${SCRIPT_DIR}/run_dyn_videossm_hybrid.sh"
