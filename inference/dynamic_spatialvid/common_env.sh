#!/bin/bash
# Shared dynamic SpatialVID inference environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_shared/common_env_infer.sh"

INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-${REPO_ROOT}/inference_outputs/dynamic_spatialvid}"
mkdir -p "${INFER_OUTPUT_ROOT}"
export INFER_OUTPUT_ROOT
