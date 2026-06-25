#!/bin/bash
# Shared environment for dynamic SpatialVID training recipes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${DATASET_BASE_PATH:-}" ]; then
  REPO_ROOT_PRE="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
  for _d in \
    "${REPO_ROOT_PRE}/data/dynamic-spatialvid-motion60/mixed" \
    "${REPO_ROOT_PRE}/data/DynMemBench-SpatialVID-motion60/mixed" \
    "${REPO_ROOT_PRE}/data/camcl_spatialvid_motion60_ready/mixed"; do
    if [ -d "${_d}" ]; then
      export DATASET_BASE_PATH="${_d}"
      break
    fi
  done
fi

source "${SCRIPT_DIR}/../_shared/common_env_memory.sh"

if [ "${DATASET_BASE_PATH:-}" = "${REPO_ROOT}/data/Context-as-Memory-Dataset" ]; then
  for _d in \
    "${REPO_ROOT}/data/dynamic-spatialvid-motion60/mixed" \
    "${REPO_ROOT}/data/DynMemBench-SpatialVID-motion60/mixed" \
    "${REPO_ROOT}/data/camcl_spatialvid_motion60_ready/mixed"; do
    if [ -d "${_d}" ]; then
      DATASET_BASE_PATH="${_d}"
      dataset_base_path="${DATASET_BASE_PATH}"
      break
    fi
  done
fi

METADATA_NAME="${METADATA_NAME:-metadata_train.csv}"
OUTPUT_BASE_ROOT="${OUTPUT_BASE_ROOT:-${REPO_ROOT}/outputs/dynamic_spatialvid}"
mkdir -p "${OUTPUT_BASE_ROOT}"
output_base="${output_base:-${OUTPUT_BASE_ROOT}/dyn_spatialvid}"

echo "[dynamic_spatialvid] DATASET_BASE_PATH=${DATASET_BASE_PATH}"
echo "[dynamic_spatialvid] METADATA_NAME=${METADATA_NAME}"
