#!/bin/bash
# Run the official arXiv:2506.05284 TSDF renderer on one reconstructed clip.
# INPUT_NPZ must contain: images, depths, intrinsic, cam_c2w.
set -euo pipefail

SPMEM_ROOT="${SPMEM_ROOT:?Set SPMEM_ROOT to a checkout of https://github.com/spmem/spmem}"
INPUT_NPZ="${INPUT_NPZ:?Set INPUT_NPZ to the reconstructed clip .npz}"
CLIP_NAME="${CLIP_NAME:?Set CLIP_NAME to the output sample name}"
GEOMETRY_OUTPUT_ROOT="${GEOMETRY_OUTPUT_ROOT:?Set GEOMETRY_OUTPUT_ROOT}"

RUN_DATA="${SPMEM_ROOT}/tsdf/run_data.py"
if [ ! -f "${RUN_DATA}" ]; then
  echo "Missing official TSDF entrypoint: ${RUN_DATA}" >&2
  exit 2
fi
if [ ! -f "${INPUT_NPZ}" ]; then
  echo "Missing reconstructed clip: ${INPUT_NPZ}" >&2
  exit 2
fi

python "${RUN_DATA}" \
  --input_dir "${INPUT_NPZ}" \
  --name "${CLIP_NAME}" \
  --save_dir "${GEOMETRY_OUTPUT_ROOT}" \
  --fast_mode

OUTPUT_VIDEO="${GEOMETRY_OUTPUT_ROOT}/${CLIP_NAME}/Vid_masktarget.mp4"
if [ ! -f "${OUTPUT_VIDEO}" ]; then
  echo "TSDF preprocessing completed without expected output: ${OUTPUT_VIDEO}" >&2
  exit 3
fi
echo "${OUTPUT_VIDEO}"

