# Static in-domain pool — download & preprocessing

Echo-Memory’s **static in-domain pool** is released through [Echo-Team/Echo-Memory-Data](https://huggingface.co/datasets/Echo-Team/Echo-Memory-Data) as tar parts under `static_pool_tar_parts/`. The underlying pool is sourced from [KlingTeam/Context-as-Memory-Dataset](https://huggingface.co/datasets/KlingTeam/Context-as-Memory-Dataset) on Hugging Face (Kling Team, SIGGRAPH Asia 2025; [arXiv:2506.03141](https://arxiv.org/abs/2506.03141)). Total size is about **340 GB** — plan disk space before downloading and unpacking.

---

## 1. Download

### Option A — Echo-Team packaged release

```bash
pip install -U "huggingface_hub[cli]"

mkdir -p data

huggingface-cli download Echo-Team/Echo-Memory-Data \
  --repo-type dataset \
  --include "static_pool_tar_parts/*" \
  --local-dir ./data/echo-memory-data-release

cat ./data/echo-memory-data-release/static_pool_tar_parts/echo-memory-data.tar.part-* | tar -xf - -C ./data
```

You should end up with `data/Context-as-Memory-Dataset/`.

### Option B — original KlingTeam source

If you prefer the upstream release, download or merge the original parts from the [KlingTeam dataset card](https://huggingface.co/datasets/KlingTeam/Context-as-Memory-Dataset):

```bash
mkdir -p data
cd data

# after all Context-as-Memory-Dataset_* parts are downloaded into this directory:
cat Context-as-Memory-Dataset_* > Context-as-Memory-Dataset.zip
unzip Context-as-Memory-Dataset.zip -d .
```

You should end up with a directory named `Context-as-Memory-Dataset/` (adjust the path below if your folder name differs).

---

## 2. Expected layout (static in-domain pool)

After extraction, point `DATASET_BASE_PATH` at the pool root (default: `data/Context-as-Memory-Dataset/`):

```text
data/Context-as-Memory-Dataset/
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
├── captions.txt         # segment captions (optional for some workflows)
└── metadata_full.csv    # released Echo-Memory segment metadata
```

Quick sanity check:

```bash
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset

test -d "${DATASET_BASE_PATH}/frames" && echo "frames OK"
test -d "${DATASET_BASE_PATH}/jsons" && echo "jsons OK"
test -d "${DATASET_BASE_PATH}/overlap_labels" && echo "overlap_labels OK"
ls "${DATASET_BASE_PATH}/frames" | head
ls "${DATASET_BASE_PATH}/jsons" | head
```

---

## 3. Point Echo-Memory at the static in-domain pool

```bash
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

Training scripts also accept `data/Context-as-Memory-Dataset` under the repo root if `DATASET_BASE_PATH` is unset.

---

## 4. Metadata (required)

`metadata_full.csv` is included in the Echo-Team packaged release. If you downloaded the upstream KlingTeam source instead, fetch the released metadata into the pool root:

```bash
cd /path/to/Echo-Memory
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset

huggingface-cli download Echo-Team/Echo-Memory-Data metadata_full.csv \
  --repo-type dataset \
  --local-dir "${DATASET_BASE_PATH}"
```

If you modify the pool or need to rebuild metadata locally, regenerate it from `frames/` and `captions.txt`:

```bash
bash scripts/run_generate_metadata.sh
```

You can also generate a smaller custom index for ablations or reduced-size training:

```bash
OUTPUT_CSV="${DATASET_BASE_PATH}/metadata_1000.csv" \
METADATA_MAX_ROWS=1000 \
bash scripts/run_generate_metadata.sh
```

Pass the custom CSV to training/evaluation with `--dataset_metadata_path "${DATASET_BASE_PATH}/metadata_1000.csv"`.

Defaults (override via env vars):

| Variable | Default | Meaning |
| --- | --- | --- |
| `OUTPUT_CSV` | `${DATASET_BASE_PATH}/metadata_full.csv` | Output metadata path |
| `SEGMENT_LENGTH` | `81` | Frames per training segment |
| `CONTEXT_FRAMES` | `5` | Context window used when building metadata |
| `NUM_WORKERS` | CPU count − 2 | Parallel workers |
| `METADATA_MAX_ROWS` / `DATASET_SIZE_ROWS` | `0` | Keep only the first N metadata rows after generation; `0` keeps the full CSV |

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
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
NUM_PROCESSES=8 bash scripts/run_precompute_ctx_target_latents.sh
```

Latents are written under `${DATASET_BASE_PATH}/latents/`. The script can use `overlap_labels/` when `--use_overlap_labels` is enabled (see `scripts/run_precompute_ctx_target_latents.sh`).

---

## 6. Training pools vs. open-domain assets

| Echo pool / asset | Location | Purpose |
| --- | --- | --- |
| Static in-domain pool | `DATASET_BASE_PATH` → `data/Context-as-Memory-Dataset` | Training, in-domain replay/revisit, metadata |
| Dynamic training pool | `DATASET_BASE_PATH` → `data/dynamic-memory-dataset` | Training on the dynamic pool ([guide](dynamic_dataset_preprocessing.md)) |
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
- Dynamic training pool: [dynamic_dataset_preprocessing.md](dynamic_dataset_preprocessing.md)
