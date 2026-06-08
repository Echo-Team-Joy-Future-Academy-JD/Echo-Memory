#!/bin/bash
# Common inference wrapper for context_learning.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIGMA_SHIFT="${SIGMA_SHIFT:-5.0}"
source "${SCRIPT_DIR}/../_shared/common_env_infer.sh"
