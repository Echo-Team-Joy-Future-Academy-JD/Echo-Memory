# Context-as-Memory dataset — download & preprocessing

Echo-Memory training and evaluation expect the public **Context-as-Memory** dataset from Kling Team (SIGGRAPH Asia 2025; [arXiv:2506.03141](https://arxiv.org/abs/2506.03141)).

**Hugging Face (recommended):** [KlingTeam/Context-as-Memory-Dataset](https://huggingface.co/datasets/KlingTeam/Context-as-Memory-Dataset)

Total size is about **340 GB**. Plan disk space before downloading.

---

## 1. Download

### Option A — Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"

# login if the repo requires authentication
# huggingface-cli login

huggingface-cli download KlingTeam/Context-as-Memory-Dataset \
  --repo-type dataset \
  --local-dir ./data/Context-as-Memory-Dataset-hf
```

If the dataset is shipped as split zip parts on Hugging Face, merge them first (see Option B), then extract into a single root folder.

### Option B — merge split zip parts (official layout)

From the [dataset card](https://huggingface.co/datasets/KlingTeam/Context-as-Memory-Dataset):

```bash
mkdir -p data
cd data

# after all Context-as-Memory-Dataset_* parts are downloaded into this directory:
cat Context-as-Memory-Dataset_* > Context-as-Memory-Dataset.zip
unzip Context-as-Memory-Dataset.zip -d .
```

You should end up with a directory named `Context-as-Memory-Dataset/` (adjust the path below if your folder name differs).

---

## 2. Expected layout

After extraction, the dataset root should look like this:

```text
Context-as-Memory-Dataset/
├── frames/              # 100 scene folders, ~7601 PNGs each
│   ├── AncientTempleEnv_0/
│   │   ├── 0000.png
│   │   └── ...
│   └── ...
├── jsons/               # per-scene camera pose JSON (one file per scene)
│   ├── AncientTempleEnv_0.json
│   └── ...
├── overlap_labels/      # per-frame overlap indices (used by context retrieval / latent precompute)
│   ├── AncientTempleEnv_0/
│   │   ├── 0.json
│   │   └── ...
│   └── ...
└── captions.txt         # segment captions (optional for some workflows)
```

Quick sanity check:

```bash
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset

test -d "${DATASET_BASE_PATH}/frames" && echo "frames OK"
test -d "${DATASET_BASE_PATH}/jsons" && echo "jsons OK"
test -d "${DATASET_BASE_PATH}/overlap_labels" && echo "overlap_labels OK"
ls "${DATASET_BASE_PATH}/frames" | head
ls "${DATASET_BASE_PATH}/jsons" | head
```

---

## 3. Point Echo-Memory at the dataset

```bash
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

Training scripts also accept `data/Context-as-Memory-Dataset` under the repo root if `DATASET_BASE_PATH` is unset.

---

## 4. Generate training metadata (required)

Build `metadata_full.csv` — segment list with context/target frame indices and camera RT fields used by Echo-Memory loaders:

```bash
cd /path/to/Echo-Memory
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset

bash scripts/run_generate_metadata.sh
```

Defaults (override via env vars):

| Variable | Default | Meaning |
| --- | --- | --- |
| `OUTPUT_CSV` | `${DATASET_BASE_PATH}/metadata_full.csv` | Output metadata path |
| `SEGMENT_LENGTH` | `81` | Frames per training segment |
| `CONTEXT_FRAMES` | `5` | Context window used when building metadata |
| `NUM_WORKERS` | CPU count − 2 | Parallel workers |

Verify:

```bash
wc -l "${DATASET_BASE_PATH}/metadata_full.csv"
head -n 3 "${DATASET_BASE_PATH}/metadata_full.csv"
```

---

## 5. Precompute latents (optional, speeds training)

If you train with precomputed VAE latents:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
NUM_PROCESSES=8 bash scripts/run_precompute_ctx_target_latents.sh
```

Latents are written under `${DATASET_BASE_PATH}/latents/`. The script can use `overlap_labels/` when `--use_overlap_labels` is enabled (see `scripts/run_precompute_ctx_target_latents.sh`).

---

## 6. What Echo-Memory uses vs. open-domain assets

| Asset | Location | Purpose |
| --- | --- | --- |
| Context-as-Memory (full) | `DATASET_BASE_PATH` | Training, in-domain replay/revisit, metadata |
| Open-domain first frames | `assets/opendomain_revisit/` | Held-out OOD revisit probes (already in repo) |

You do **not** need to rebuild open-domain anchors for the released revisit suite.

---

## 7. Troubleshooting

**`DATASET_BASE_PATH is not set`** — export the variable or place data at `data/Context-as-Memory-Dataset` relative to the repo root.

**Missing `frames/` or `jsons/`** — re-check unzip path; the root folder name must match what you pass to `DATASET_BASE_PATH`.

**Metadata script missing** — ensure you are on the latest Echo-Memory `main` branch; metadata generation is invoked via `scripts/run_generate_metadata.sh`.

**Disk space** — keep ~340 GB for raw frames plus extra space for `metadata_full.csv`, `latents/`, and training outputs.

---

## Reference

- Static in-domain pool: [dataset_preprocessing.md](dataset_preprocessing.md)
- Dynamic training data: [dynamic_dataset_preprocessing.md](dynamic_dataset_preprocessing.md)
