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

- `run_ablation_no_memory_baseline_two_chunk.sh`: anchor/no-extra-memory reference.
- `run_ablation_framepack_weight_two_chunk.sh`: token weighting without length reduction.
- `run_ablation_framepack_len_r2_two_chunk.sh`: temporal length compression, ratio 2.
- `run_ablation_framepack_len_r4_two_chunk.sh`: temporal length compression, ratio 4.
- `run_ablation_framepack_hybrid_r2_weight_two_chunk.sh`: ratio-2 length compression plus token weighting.
- `run_ablation_framepack_hybrid_r4_weight_two_chunk.sh`: ratio-4 length compression plus token weighting.
- `run_spatial_memory_baseline.sh`: representative spatial memory tokens.
- `run_ablation_spatial_inject_none_two_chunk.sh`: spatial storage with withheld read-out.
- `run_ablation_spatial_concat_text_two_chunk.sh`: spatial memory read through text KV concatenation.
- `run_ablation_spatial_cross_attn_readout_two_chunk.sh`: spatial memory read through dedicated cross-attention.
- `run_videossm_hybrid_baseline.sh`: legacy VideoSSM hybrid memory.
- `run_ablation_videossm_hybrid_two_chunk.sh`: legacy VideoSSM with the paper two-chunk monitor.
- `run_ablation_block_wise_ssm_two_chunk.sh`: paper-aligned block-wise SSM recipe.
- `run_all_ablations_two_chunk.sh`: convenience launcher for the full ablation set.

The two-chunk scripts share `common_sampling_two_chunk.sh` and expose `CKPT_INTERVAL`, `TIMESTEP_SHIFT`, `SAMPLING_INTERVAL_STEPS`, `SAMPLING_NUM_INFERENCE_STEPS`, `SAMPLING_HEIGHT`, `SAMPLING_WIDTH`, and `SAMPLING_NUM_FRAMES` as environment overrides.

## Context Learning

`train/context_learning/` keeps context-frame recipes:

- `run_pre_qkv_ctx1.sh`
- `run_pre_qkv_ctx5.sh`
- `run_pre_qkv_ctx20.sh`
- `run_pre_qkv_ctx5_lr_8e5_rt_merge_zero_init_mlp_ssm.sh`
- `run_ctx5_no_action_ablation.sh`

The shared environment file is `train/_shared/common_env_memory.sh`. It does not set private paths or credentials; configure Weights & Biases in your shell if you use it.
