# Dynamic SpatialVID Inference

Wrappers for checkpoints trained on the dynamic SpatialVID motion-filtered pool. They call `inference/unified_inference.py` and mirror the six public dynamic rows:

| Script | `--memory_type` | Training recipe |
| --- | --- | --- |
| `run_infer_dyn_ctx1.sh` | `context_k1` | `train/dynamic_spatialvid/run_dyn_ctx1.sh` |
| `run_infer_dyn_ctx5.sh` | `context_k5` | `train/dynamic_spatialvid/run_dyn_ctx5.sh` |
| `run_infer_dyn_ctx20.sh` | `context_k20` | `train/dynamic_spatialvid/run_dyn_ctx20.sh` |
| `run_infer_dyn_spatial_mem.sh` | `spatial_mem` | `train/dynamic_spatialvid/run_dyn_spatial_mem.sh` |
| `run_infer_dyn_block_wise_ssm.sh` | `block_wise_ssm` | `train/dynamic_spatialvid/run_dyn_block_wise_ssm.sh` |
| `run_infer_dyn_videossm_hybrid.sh` | `videossm_hybrid` | `train/dynamic_spatialvid/run_dyn_videossm_hybrid.sh` |

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export CKPT=./ckpts/dynamic_spatialvid/spatial_mem/epoch-0.safetensors
PROMPT="A dynamic outdoor scene with a smooth camera move" \
  bash inference/dynamic_spatialvid/run_infer_dyn_spatial_mem.sh
```

To run all six rows:

```bash
CKPT_DIR=./ckpts/dynamic_spatialvid bash inference/dynamic_spatialvid/run_infer_all_dyn.sh
```

## Demo Selection

Dynamic demos should be selected from training scenes by random replay, then manually picked:

1. Sample candidate rows from `metadata_train.csv` or a small public subset such as `metadata_train_sample.csv`.
2. Replay the same scene/prompt/action with all six dynamic checkpoints.
3. Pick representative successes for the public preview grid.

Evaluation for the dynamic benchmark is intentionally TODO for now; only training and inference wrappers are public.
