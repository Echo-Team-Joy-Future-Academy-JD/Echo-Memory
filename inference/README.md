# Inference Recipes

Bash-level inference scripts mirroring `train/` — one script per memory row, all calling `inference/unified_inference.py`.

## Usage

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B

# Single memory type
CKPT=./ckpts/spatial_mem/epoch-0.safetensors \
  bash inference/memory_baselines_basic/run_infer_spatial_mem.sh

# With custom prompt and context image
CKPT=./ckpts/spatial_mem/epoch-0.safetensors \
PROMPT="A toy bear on a table" \
CONTEXT_IMAGE=assets/opendomain_revisit/1774363417.png \
  bash inference/memory_baselines_basic/run_infer_spatial_mem.sh

# All memory baselines (needs CKPT_DIR with per-row folders)
CKPT_DIR=./ckpts bash inference/memory_baselines_basic/run_infer_all.sh

# Dynamic SpatialVID row
CKPT=./ckpts/dynamic_spatialvid/spatial_mem/epoch-0.safetensors \
  bash inference/dynamic_spatialvid/run_infer_dyn_spatial_mem.sh
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CKPT` | (required) | Path to `.safetensors` checkpoint |
| `WAN_BASE_MODEL` | (required) | Wan 2.1 base model directory |
| `PROMPT` | Generic game scene prompt | Text prompt |
| `CONTEXT_IMAGE` | (none) | First-frame context image path |
| `ACTION_PATH` | `env/action_rotation_left_45.json` | Camera trajectory JSON |
| `SEED` | `0` | Random seed |
| `HEIGHT` / `WIDTH` | `352` / `640` | Resolution |
| `NUM_FRAMES` | `81` | Frames per chunk |
| `NUM_INFERENCE_STEPS` | `50` | Denoising steps |
| `SIGMA_SHIFT` | `15.0` (memory baselines) / `5.0` (context learning) | Timestep shift |
| `INFER_OUTPUT_ROOT` | `inference_outputs/` | Output directory |

## Script Mapping

### Memory Baselines (`inference/memory_baselines_basic/`)

| Inference script | `--memory_type` | Training script |
|---|---|---|
| `run_infer_no_memory.sh` | `no_memory` | `run_ablation_no_memory_baseline_two_chunk.sh` |
| `run_infer_framepack_weight.sh` | `framepack_weight` | `run_ablation_framepack_weight_two_chunk.sh` |
| `run_infer_framepack_len_r2.sh` | `framepack_len_r2` | `run_ablation_framepack_len_r2_two_chunk.sh` |
| `run_infer_framepack_len_r4.sh` | `framepack_len_r4` | `run_ablation_framepack_len_r4_two_chunk.sh` |
| `run_infer_framepack_hybrid_r2.sh` | `framepack_hybrid_r2` | `run_ablation_framepack_hybrid_r2_weight_two_chunk.sh` |
| `run_infer_framepack_hybrid_r4.sh` | `framepack_hybrid_r4` | `run_ablation_framepack_hybrid_r4_weight_two_chunk.sh` |
| `run_infer_spatial_mem.sh` | `spatial_mem` | `run_spatial_memory_baseline.sh` |
| `run_infer_spatial_concat_text.sh` | `spatial_concat_text` | `run_ablation_spatial_concat_text_two_chunk.sh` |
| `run_infer_spatial_inject_none.sh` | `spatial_inject_none` | `run_ablation_spatial_inject_none_two_chunk.sh` |
| `run_infer_spatial_cross_attn_readout.sh` | `spatial_cross_attn_readout` | `run_ablation_spatial_cross_attn_readout_two_chunk.sh` |
| `run_infer_videossm_hybrid.sh` | `videossm_hybrid` | `run_videossm_hybrid_baseline.sh` |
| `run_infer_block_wise_ssm.sh` | `block_wise_ssm` | `run_ablation_block_wise_ssm_two_chunk.sh` |

### Context Learning (`inference/context_learning/`)

| Inference script | `--memory_type` | Training script |
|---|---|---|
| `run_infer_ctx1.sh` | `context_k1` | `run_pre_qkv_ctx1.sh` |
| `run_infer_ctx5.sh` | `context_k5` | `run_pre_qkv_ctx5.sh` |
| `run_infer_ctx20.sh` | `context_k20` | `run_pre_qkv_ctx20.sh` |

### Dynamic SpatialVID (`inference/dynamic_spatialvid/`)

Dynamic wrappers mirror the six dynamic training rows in `train/dynamic_spatialvid/`. They are intended for qualitative replay and demo generation; dynamic evaluation scripts are TODO.
