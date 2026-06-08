# Runtime Smoke Tests

This note records the runtime bug fixed after the public cleanup and the smoke
tests used to verify eval, inference, and 8-GPU training.

## Bug fixed

`diffsynth/models/` was missing from the public tree while the pipeline imports
it directly:

```python
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.memory.spatial_grid_memory import inject_spatial_memory
```

Without the vendored DiffSynth model source modules, eval and inference fail at
import time before any checkpoint can be loaded. `diffsynth/data/` is also
required because `diffsynth/__init__.py` imports it.

The fix keeps only source code in Git:

- `diffsynth/models/**/*.py`
- `diffsynth/data/**/*.py`

Checkpoint files remain ignored by `.gitignore`:

- `*.safetensors`
- `*.pth`
- `*.pt`
- `*.ckpt`

## Smoke scripts

Two scripts are provided for maintainers:

```bash
bash scripts/smoke_inference.sh
bash scripts/smoke_train_8gpu.sh
```

Both scripts are environment-variable driven. Example local setup:

```bash
export ECHO_MEMORY_CONDA_ENV=storymem
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
export CKPT=/path/to/epoch-0.safetensors
```

`scripts/smoke_inference.sh` runs a minimal replay/inference path with reduced
resolution and denoising steps. `scripts/smoke_train_8gpu.sh` launches 8 GPU
training and exits after `MAX_TRAIN_STEPS=1` without writing an epoch
checkpoint. The training smoke still reports the full training target in the
progress bar via `PROGRESS_TOTAL_STEPS=30000`; `MAX_TRAIN_STEPS` only controls
where the smoke run exits.

## Verified result

The local validation used:

- conda env: `storymem`
- base model: Wan 2.1 T2V 1.3B
- dataset: Static in-domain pool
- checkpoint: paper-aligned context K=1 epoch-0 checkpoint

Inference smoke completed and wrote output under `outputs/smoke_inference/`.

8-GPU training smoke completed with:

```text
[SMOKE] Reached max_train_steps=1
[smoke_train_8gpu] OK
train/loss 0.68457
```

The first two training attempts were expected setup failures:

1. `--spike_threshold` was referenced by legacy launchers but missing from the
   current parser. The parser now accepts it.
2. All 8 GPUs were occupied by vLLM workers. After freeing the GPUs, the 8-GPU
   smoke run completed.

