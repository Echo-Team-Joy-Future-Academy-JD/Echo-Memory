#!/usr/bin/env bash
# Upload Echo-Memory paper baseline checkpoints to Hugging Face (Echo-Team/Echo-Memory).
#
# Requires a token with write access to the Echo-Team org (Wayne-King / Weiyang Jin):
#   export HF_TOKEN=hf_...
#   hf auth login --token "$HF_TOKEN"
#   bash scripts/upload_hf_checkpoints.sh
#
# Optional:
#   CHECKPOINT_SOURCE_ROOT=/path/to/training/outputs
#   HF_REPO_ID=Echo-Team/Echo-Memory
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${REPO_ROOT}/scripts/hf_checkpoints_manifest.json"
SOURCE_ROOT="${CHECKPOINT_SOURCE_ROOT:-/pfs/weiyang/CAM-CL/outputs}"
STAGING="${REPO_ROOT}/.hf-upload"
HF_REPO_ID="${HF_REPO_ID:-Echo-Team/Echo-Memory}"
CKPT_NAME="epoch-0.safetensors"

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[upload_hf_checkpoints] Install huggingface_hub: pip install -U huggingface_hub" >&2
  exit 1
fi

HF_CMD=(hf)
if ! hf --help >/dev/null 2>&1; then
  HF_CMD=(huggingface-cli)
fi

echo "[upload_hf_checkpoints] Auth:"
"${HF_CMD[@]}" auth whoami || true

echo "[upload_hf_checkpoints] Create repo ${HF_REPO_ID} (if needed)"
if [[ "${HF_CMD[0]}" == "hf" ]]; then
  hf repo create "${HF_REPO_ID}" --repo-type model --exist-ok
else
  huggingface-cli repo create "${HF_REPO_ID}" --exist-ok
fi

rm -rf "${STAGING}"
mkdir -p "${STAGING}"

python3 - <<'PY' "${MANIFEST}" "${SOURCE_ROOT}" "${STAGING}" "${CKPT_NAME}"
import json, os, sys
from pathlib import Path

manifest_path, source_root, staging, ckpt_name = sys.argv[1:5]
manifest = json.loads(Path(manifest_path).read_text())
rows = []
for item in manifest["checkpoints"]:
    src = Path(source_root) / item["source_subdir"] / ckpt_name
    if not src.is_file():
        raise SystemExit(f"Missing checkpoint: {src}")
    dest_dir = Path(staging) / Path(item["hf_path"]).parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ckpt_name
    os.symlink(src.resolve(), dest)
    rows.append({**item, "training_steps": manifest["training_steps"], "backbone": manifest["backbone"]})

out = {
    "repo_id": manifest["repo_id"],
    "backbone": manifest["backbone"],
    "checkpoints": rows,
}
Path(staging, "checkpoints.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"Staged {len(rows)} checkpoints under {staging}")
PY

cp "${REPO_ROOT}/scripts/hf_model_card_README.md" "${STAGING}/README.md"

echo "[upload_hf_checkpoints] Uploading to ${HF_REPO_ID} (~11 × 3.2 GB; may take a while)"
if [[ "${HF_CMD[0]}" == "hf" ]]; then
  hf upload "${HF_REPO_ID}" "${STAGING}/." . --repo-type model --commit-message "Release Echo-Memory Wan 2.1 1.3B memory baseline checkpoints (epoch-0)"
else
  huggingface-cli upload "${HF_REPO_ID}" "${STAGING}" . --repo-type model
fi

echo "[upload_hf_checkpoints] Done: https://huggingface.co/${HF_REPO_ID}"
