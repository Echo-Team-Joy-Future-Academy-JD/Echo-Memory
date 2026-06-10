# Memory Mechanisms

This note keeps repository names aligned with the paper tables. It is intentionally short; use the training scripts for exact hyperparameters.

| Paper family | Paper row / repo name | What is stored or read | Main code path | Training entry |
| --- | --- | --- | --- | --- |
| Raw context | `context_k1`, `context_k5`, `context_k20` | Uncompressed retrieved context frames. `K=1` is the anchor/I2V floor; `K=5/20` are context-learning capacity rows. | `diffsynth/pipelines/wan_video_new.py` context latent path | `train/context_learning/run_pre_qkv_ctx{1,5,20}.sh` |
| Compression | `framepack_weight` | Context tokens are kept at the same length but temporally reweighted. | `diffsynth/models/memory/framepack_weight.py` | `train/memory_baselines_basic/run_ablation_framepack_weight_two_chunk.sh` |
| Compression | `framepack_len_r2`, `framepack_len_r4` | Context latents and matched RT actions are pooled along time. | `diffsynth/models/memory/framepack_length.py` | `train/memory_baselines_basic/run_ablation_framepack_len_r{2,4}_two_chunk.sh` |
| Compression | `framepack_hybrid_r2`, `framepack_hybrid_r4` | Length compression plus token reweighting. | `wan_video_new.py` + FramePack helpers | `train/memory_baselines_basic/run_ablation_framepack_hybrid_r*_weight_two_chunk.sh` |
| Spatial | `spatial_mem` | Context tokens are summarized into spatial grid memory tokens. | `diffsynth/models/memory/spatial_grid_memory.py` | `train/memory_baselines_basic/run_spatial_memory_baseline.sh` |
| Spatial | `spatial_inject_none`, `spatial_concat_text`, `spatial_cross_attn_readout` | Same storage, different read-out: withheld, text-KV concat, or dedicated cross-attention. | `spatial_grid_memory.py` read-out helpers | matching `run_ablation_spatial_*_two_chunk.sh` scripts |
| State-space | `block_wise_ssm` | Paper-aligned recurrent state attached to selected DiT blocks. Checkpoint keys contain `block_wise_ssm.*`. | `diffsynth/models/memory/block_wise_ssm.py` | `train/memory_baselines_basic/run_ablation_block_wise_ssm_two_chunk.sh` |
| State-space | `videossm_hybrid` | Legacy VideoSSM hybrid baseline: depthwise temporal-conv state-space-like module. Checkpoint keys contain `videossm_hybrid.*`. | `diffsynth/models/memory/videossm_hybrid.py` | `train/memory_baselines_basic/run_videossm_hybrid_baseline.sh` |

## Naming Rules

- Use **Block-wise SSM** only for `--use_block_wise_ssm` / `BlockWiseStateSpaceMemory`.
- Use **VideoSSM hybrid** only for the legacy `--use_videossm_hybrid` / `HybridStateSpaceMemory` baseline.
- Use **Context learning** for raw-context capacity rows (`K=1/5/20`), not for compact memory modules.
- Keep checkpoint folder names stable; `env/memory_baseline_runtime.py` and `inference/unified_inference.py` infer memory profiles from those names.
