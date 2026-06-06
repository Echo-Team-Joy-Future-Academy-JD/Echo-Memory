---
name: echo-memory-train
description: >-
  Run Echo-Memory memory-baseline and context training recipes on Wan 2.1 1.3B.
  Use when training spatial/SSM/compression/context rows, editing train/*.sh
  launchers, or configuring DATASET_BASE_PATH for static or dynamic pools.
---

# Echo-Memory training

## Required env (repo root)

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset   # static in-domain pool
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export OUTPUT_BASE_ROOT=$PWD/outputs
```

- **Static in-domain pool** — default `data/Context-as-Memory-Dataset`; see `doc/dataset_preprocessing.md`
- **Dynamic training pool** — e.g. `data/dynamic-memory-dataset`; see `doc/dynamic_dataset_preprocessing.md`

## Entry scripts

| Family | Directory | Example |
| --- | --- | --- |
| Spatial / SSM / compression | `train/memory_baselines_basic/` | `run_spatial_memory_baseline.sh`, `run_ablation_block_wise_ssm_two_chunk.sh` |
| Context K=1/5/20 | `train/context_learning/` | `run_pre_qkv_ctx1.sh`, `run_pre_qkv_ctx20.sh` |

Run from repository root: `bash train/memory_baselines_basic/run_spatial_memory_baseline.sh`

Shared env: `train/_shared/common_env_memory.sh` — no private paths baked in.

## Agent checklist

1. Confirm which **memory row** (paper family + row id) the user wants.
2. Set `DATASET_BASE_PATH` to the correct **training pool** name in docs (Echo terms only in public markdown).
3. Prefer editing existing `run_*.sh` patterns over new one-off Python entrypoints.
4. Outputs go to `outputs/` unless `OUTPUT_BASE_ROOT` is set.
5. Do not commit `data/`, `outputs/`, checkpoints, or machine-local paths.

## Docs

- `train/README.md` — launcher index
- `train/memory_baselines_basic/README.md` — ablation set
