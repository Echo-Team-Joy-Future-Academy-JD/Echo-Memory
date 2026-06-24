#!/bin/bash
# memory_baselines_basic: FAR/FramePack / Spatial Memory / VideoSSM baseline 共用环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_shared/common_env_memory.sh"
OUTPUT_BASE_ROOT="${OUTPUT_BASE_ROOT:-${REPO_ROOT}/outputs}"
mkdir -p "${OUTPUT_BASE_ROOT}"
output_base="${output_base:-${OUTPUT_BASE_ROOT}/memory_baselines_basic}"
