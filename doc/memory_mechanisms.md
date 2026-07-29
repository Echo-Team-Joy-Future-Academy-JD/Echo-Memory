# Memory Mechanisms

This note maps the paper's memory rows to the repository implementation and explains the modeling role of each family. Echo-Memory treats memory as a controlled intervention on what information from chunk 1 is stored and how chunk 2 reads it back during denoising.

## Modeling View

All rows use the same action-conditioned Wan DiT backbone and the same two-chunk training/evaluation setup:

1. **Context chunk:** clean history frames are encoded into latent/context tokens, optionally with matched camera RT actions.
2. **Target chunk:** noisy target latents are denoised while the selected memory mechanism exposes information from the context chunk.
3. **Read-out:** memory is injected through raw context concatenation, compressed context tokens, spatial memory tokens, or recurrent state-space modules attached to DiT blocks.

The ablations are designed to change only the memory pathway while keeping the backbone, action conditioning, resolution, chunk length, and training schedule aligned.

## Paper Rows

| Paper family | Paper row / repo name | What is stored or read | Main code path | Training entry |
| --- | --- | --- | --- | --- |
| Raw context | `context_k1`, `context_k5`, `context_k20` | Uncompressed retrieved context frames. `K=1` is the anchor/I2V floor; `K=5/20` are context-learning capacity rows. | `diffsynth/pipelines/wan_video_new.py` context latent path | `train/context_learning/run_pre_qkv_ctx{1,5,20}.sh` |
| Compression | `framepack_weight` | Context tokens are kept at the same length but temporally reweighted. | `diffsynth/models/memory/framepack_weight.py` | `train/memory_baselines_basic/run_ablation_framepack_weight_two_chunk.sh` |
| Compression | `framepack_len_r2`, `framepack_len_r4`, `framepack_len_r8` | Context latents and matched RT actions are pooled along time. The released r8 row uses 81 newest-first history frames and packed-multiscale compression. | `diffsynth/models/memory/framepack_length.py` | `train/memory_baselines_basic/run_ablation_framepack_len_r{2,4,8}_two_chunk.sh` |
| Compression | `framepack_hybrid_r2`, `framepack_hybrid_r4` | Length compression plus token reweighting. | `wan_video_new.py` + FramePack helpers | `train/memory_baselines_basic/run_ablation_framepack_hybrid_r*_weight_two_chunk.sh` |
| Token-grid | `spatial_mem` | Context tokens are time-averaged and summarized into learned grid tokens. This is the implementation behind the currently reported `spatial_mem` row; it does **not** reconstruct depth or 3D geometry. | `diffsynth/models/memory/spatial_grid_memory.py` | `train/memory_baselines_basic/run_spatial_memory_baseline.sh` |
| Token-grid | `spatial_inject_none`, `spatial_concat_text`, `spatial_cross_attn_readout` | Same token-grid storage, different read-out: withheld, text-KV concat, or dedicated cross-attention. | `spatial_grid_memory.py` read-out helpers | matching `run_ablation_spatial_*_two_chunk.sh` scripts |
| Geometry-grounded spatial | `geometry_spatial_mem` | A static scene is reconstructed outside the DiT using depth, intrinsics, extrinsics, and TSDF fusion. The fused point cloud is rendered along the target trajectory, VAE-encoded, and converted into conditioning tokens. | `diffsynth/models/memory/geometry_spatial_memory.py` | `train/memory_baselines_basic/run_geometry_spatial_memory_baseline.sh` |
| State-space | `block_wise_ssm_causal_v2` | Causal recurrent state attached to selected DiT blocks. Context is a leak-free prefix and shares the target-first RT reference; a bounded residual path prevents collapse to a no-op. | `diffsynth/models/memory/block_wise_ssm.py` | `train/memory_baselines_basic/run_ablation_block_wise_ssm_causal_v2_two_chunk.sh` |
| State-space | `videossm_hybrid` | Legacy VideoSSM hybrid baseline: depthwise temporal-conv state-space-like module. Checkpoint keys contain `videossm_hybrid.*`. | `diffsynth/models/memory/videossm_hybrid.py` | `train/memory_baselines_basic/run_videossm_hybrid_baseline.sh` |

## Naming Rules

- Do not describe `SpatialGridMemory` or the existing `spatial_mem` results as the
  geometry-grounded method from arXiv:2506.05284. It is a token-grid baseline.
- Use **Geometry-grounded Spatial Memory** only when the metadata supplies
  rendered static geometry through `geometry_memory` (or a configured column).
  The geometry extractor is the external reconstruction pipeline: depth and
  cameras → TSDF-fused static point cloud → target-view renders. The model-side
  encoder does not estimate depth itself.
- Use **Block-wise SSM** only for `--use_block_wise_ssm` / `BlockWiseStateSpaceMemory`.
- Use **VideoSSM hybrid** only for the legacy `--use_videossm_hybrid` / `HybridStateSpaceMemory` baseline.
- Use **Context learning** for raw-context capacity rows (`K=1/5/20`), not for compact memory modules.
- Keep checkpoint folder names stable; `env/memory_baseline_runtime.py` and `inference/unified_inference.py` infer memory profiles from those names.
