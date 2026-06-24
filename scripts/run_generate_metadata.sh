#!/bin/bash
# Generate metadata CSV for a context-based memory dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

DETECTED_CPUS=$(python3 -c "import os; print(os.cpu_count())")
OPTIMAL_WORKERS=$((DETECTED_CPUS > 4 ? DETECTED_CPUS - 2 : DETECTED_CPUS))

NUM_WORKERS=${NUM_WORKERS:-$OPTIMAL_WORKERS}
DATASET_BASE_PATH="${DATASET_BASE_PATH:-${REPO_ROOT}/data/Context-as-Memory-Dataset}"
OUTPUT_CSV="${OUTPUT_CSV:-${DATASET_BASE_PATH}/metadata_full.csv}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-81}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-5}"
METADATA_MAX_ROWS="${METADATA_MAX_ROWS:-${DATASET_SIZE_ROWS:-0}}"

echo "=========================================="
echo "Context-as-Memory Metadata Generation"
echo "=========================================="
echo "Detected CPUs: $DETECTED_CPUS"
echo "Using workers: $NUM_WORKERS"
echo "Dataset: ${DATASET_BASE_PATH}"
echo "Output: ${OUTPUT_CSV}"
echo "Max rows: ${METADATA_MAX_ROWS} (0 = full metadata)"
echo "=========================================="
echo ""

python3 src/data/preprocess_cam_dataset.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --output_csv "${OUTPUT_CSV}" \
  --segment_length "${SEGMENT_LENGTH}" \
  --context_frames "${CONTEXT_FRAMES}"

if [ "${METADATA_MAX_ROWS}" != "0" ]; then
    python3 - "${OUTPUT_CSV}" "${METADATA_MAX_ROWS}" <<'PY'
import csv
import os
import sys

path = sys.argv[1]
max_rows = int(sys.argv[2])
if max_rows < 0:
    raise SystemExit("METADATA_MAX_ROWS must be >= 0")
if max_rows > 0:
    tmp_path = f"{path}.tmp"
    with open(path, newline="", encoding="utf-8") as src, open(tmp_path, "w", newline="", encoding="utf-8") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader, None)
        if header is not None:
            writer.writerow(header)
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            writer.writerow(row)
    os.replace(tmp_path, path)
    print(f"Truncated metadata to {max_rows} rows: {path}")
PY
fi

echo ""
echo "Generation complete! Checking results..."
if [ -f "${OUTPUT_CSV}" ]; then
    echo "CSV file generated successfully"
    wc -l "${OUTPUT_CSV}"
else
    echo "CSV file generation failed"
fi
