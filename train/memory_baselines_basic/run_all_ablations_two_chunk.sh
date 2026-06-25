#!/bin/bash
# Convenience launcher for the paper ablation matrix. This is long-running.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_ablation_no_memory_baseline_two_chunk.sh"
bash "${SCRIPT_DIR}/run_ablation_framepack_weight_two_chunk.sh"
bash "${SCRIPT_DIR}/run_ablation_framepack_len_r2_two_chunk.sh"
bash "${SCRIPT_DIR}/run_ablation_framepack_len_r4_two_chunk.sh"
bash "${SCRIPT_DIR}/run_ablation_framepack_hybrid_r2_weight_two_chunk.sh"
bash "${SCRIPT_DIR}/run_ablation_framepack_hybrid_r4_weight_two_chunk.sh"
bash "${SCRIPT_DIR}/run_ablation_spatial_inject_none_two_chunk.sh"
bash "${SCRIPT_DIR}/run_ablation_spatial_concat_text_two_chunk.sh"
bash "${SCRIPT_DIR}/run_ablation_spatial_cross_attn_readout_two_chunk.sh"
bash "${SCRIPT_DIR}/run_ablation_videossm_hybrid_two_chunk.sh"
bash "${SCRIPT_DIR}/run_ablation_block_wise_ssm_two_chunk.sh"
