#!/bin/bash
# ctx=5 / ctx=20 + 每帧单独 VAE 注入（无时序降采样），context_actions 每帧一条；baseline 为 run_01_post / run_02_pre_norm
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_shared/common_env_memory.sh"
OUTPUT_BASE_ROOT="${OUTPUT_BASE_ROOT:-${REPO_ROOT}/outputs}"
mkdir -p "${OUTPUT_BASE_ROOT}"
output_base="${output_base:-${OUTPUT_BASE_ROOT}/context_learning}"
