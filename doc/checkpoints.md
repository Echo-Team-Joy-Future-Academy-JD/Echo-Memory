# Checkpoints (Hugging Face)

All released weights live in one model repo:

**[Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory)**

Backbone: **Wan2.1-T2V-1.3B** · checkpoint file: **`epoch-0.safetensors`** · training: **30,000 steps** (1 epoch on the static in-domain pool).

## Paper baselines

| Family | Paper row | HF weight | Train recipe |
| --- | --- | --- | --- |
| Raw context | Context K=1 | [`context_k1/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/context_k1) | `train/context_learning/run_pre_qkv_ctx1.sh` |
| Raw context | Context K=20 | [`context_k20/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/context_k20) | `train/context_learning/run_pre_qkv_ctx20.sh` |
| Spatial | Spatial Memory | [`spatial_mem/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/spatial_mem) | `train/memory_baselines_basic/run_spatial_memory_baseline.sh` |
| State-space | Block-wise SSM | [`block_wise_ssm_two_chunk/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/block_wise_ssm_two_chunk) | `train/memory_baselines_basic/run_ablation_block_wise_ssm_two_chunk.sh` |
| State-space | Legacy Hybrid (VideoSSM) | [`videossm_hybrid/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/videossm_hybrid) | `train/memory_baselines_basic/run_videossm_hybrid_baseline.sh` |

## Extended spatial / SSM ablations

| Family | Row | HF weight | Train recipe |
| --- | --- | --- | --- |
| Spatial | concat text readout | [`spatial_concat_text_two_chunk/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/spatial_concat_text_two_chunk) | `run_ablation_spatial_concat_text_two_chunk.sh` |
| Spatial | inject none | [`spatial_inject_none_two_chunk/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/spatial_inject_none_two_chunk) | `run_ablation_spatial_inject_none_two_chunk.sh` |
| Spatial | cross-attn readout (t32) | [`spatial_cross_attn_readout_t32_g4_two_chunk/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/spatial_cross_attn_readout_t32_g4_two_chunk) | `run_ablation_spatial_cross_attn_readout_two_chunk.sh` |
| State-space | SSM ctx=1, every 4, hint 21 | [`ssm_ablation_ctx1_every4_hint21/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/ssm_ablation_ctx1_every4_hint21) | internal ablation |
| State-space | SSM ctx=5, every 1, hint 21 | [`ssm_ablation_ctx5_every1_hint21/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/ssm_ablation_ctx5_every1_hint21) | internal ablation |
| State-space | SSM ctx=5, every 4, hint 81 | [`ssm_ablation_ctx5_every4_hint81/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/ssm_ablation_ctx5_every4_hint81) | internal ablation |

Context K=5 and FramePack compression rows are not yet published as `epoch-0` checkpoints in this release.

## Upload (maintainers)

```bash
hf auth login   # Wayne-King token with Echo-Team write access
bash scripts/upload_hf_checkpoints.sh
```

Manifest: `scripts/hf_checkpoints_manifest.json`.
