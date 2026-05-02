# Training Recipes

All launchers assume they are run from the repository root or can derive it from their script path.

Before training:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
export OUTPUT_BASE_ROOT=$PWD/outputs
```

## Memory Baselines

`train/memory_baselines_basic/` contains the public memory baseline recipes:

- `run_spatial_memory_baseline.sh`: spatial memory tokens.
- `run_videossm_hybrid_baseline.sh`: VideoSSM hybrid memory.
- `run_ablation_block_wise_ssm_two_chunk.sh`: block-wise SSM recipe.
- `run_framepack_baseline.sh`: FramePack/FAR-style context memory.
- `run_framepack_lencompress_r2.sh`: FramePack length compression.
- `run_ablation_no_memory_baseline_two_chunk.sh`: no-memory baseline.

## Context Learning

`train/context_learning/` keeps context-frame recipes:

- `run_pre_qkv_ctx1.sh`
- `run_pre_qkv_ctx5.sh`
- `run_pre_qkv_ctx20.sh`
- `run_pre_qkv_ctx5_lr_8e5_rt_merge_zero_init_mlp_ssm.sh`
- `run_ctx5_no_action_ablation.sh`

The shared environment file is `train/_shared/common_env_memory.sh`. It does not set private paths or credentials; configure Weights & Biases in your shell if you use it.
