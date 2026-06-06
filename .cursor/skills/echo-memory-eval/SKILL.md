---
name: echo-memory-eval
description: >-
  Run Echo-Memory replay, in-domain loop/revisit, and open-domain eval v2 scripts.
  Use when smoke-testing HF checkpoints, editing eval/v2, or tracing CKPT → memory
  profile via env/memory_baseline_runtime.py.
---

# Echo-Memory evaluation

## Required env

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CKPT=./ckpts/spatial_mem/epoch-0.safetensors
```

**CKPT path must include the row folder name** (e.g. `spatial_mem/`, `context_k20/`) so `env/memory_baseline_runtime.py` selects the right memory profile.

## Main scripts

| Goal | Script |
| --- | --- |
| GT replay smoke (PSNR/SSIM/LPIPS) | `eval/v2/run_basic_replay_gt.sh` |
| Loop + in-domain + open-domain revisit | `eval/v2/run_static_consistency_loop_and_revisit.sh` |
| Revisit suite assets | `eval/v2/revisit_suite/` |

Download a baseline:

```bash
huggingface-cli download Echo-Team/Echo-Memory spatial_mem/epoch-0.safetensors --local-dir ./ckpts
```

Index: `doc/checkpoints.md` · HF repo: `Echo-Team/Echo-Memory`

## Eval branches (name these in prompts)

- **Replay** — short-horizon pixel fidelity
- **In-domain** — 180° loop / held layouts
- **Open-domain** — edited first frames, return probes (`assets/opendomain_revisit/`)

## Agent checklist

1. Ask which branch (replay / in-domain / open-domain) unless obvious from context.
2. Verify `CKPT` row folder matches the memory family under test.
3. Prefer extending existing `eval/v2/*.sh` over duplicating inference loops.
4. Optional vars: `EVALS_ROOT`, `NUM_SAMPLES_LOOP`, `MULTIVIEW_FIRSTFRAME_DIR` — see `eval/v2/README.md`.
