# Memory Baselines Basic: FramePack, Spatial, and State-Space

This folder contains the public training recipes for the paper's controlled memory-design matrix. See `../../doc/memory_mechanisms.md` for the concise paper-row to implementation map. The scripts vary only the memory/context profile while keeping the backbone, optimizer, action conditioning, and evaluation interface aligned.

## Paper Rows and Code Mapping

| Paper row | Mechanism | Main implementation |
|-----------|-----------|---------------------|
| FramePack-Weight | Per-frame temporal decay and global scaling over context tokens; context length is unchanged. | `diffsynth/models/memory/framepack_weight.py` |
| FramePack-Length | Temporal mean pooling over context latents with matched RT-action padding and pooling. | `diffsynth/models/memory/framepack_length.py` |
| Hybrid FramePack | Length compression plus token weighting. | `wan_video_new.py` memory path plus FramePack helpers |
| Spatial Memory | Time-mean context summary to spatial grid tokens, injected by a selected read-out path. | `diffsynth/models/memory/spatial_grid_memory.py` |
| Block-wise SSM | Paper-aligned recurrent state inside selected DiT blocks. | `diffsynth/models/memory/block_wise_ssm.py` + `--use_block_wise_ssm` |
| VideoSSM hybrid | Legacy lightweight temporal-convolution state-space baseline; kept separate from Block-wise SSM. | `diffsynth/models/memory/videossm_hybrid.py` + `--use_videossm_hybrid` |

## Two-Chunk Ablation Scripts

The two-chunk scripts source `common_env.sh` and `common_sampling_two_chunk.sh`. The monitor uses `left_45` followed by `right_45`, writes `sampling_videos/step_*_two_chunk_memory*.mp4`, and stores metadata beside the videos.

| Script | Purpose |
|--------|---------|
| `run_ablation_no_memory_baseline_two_chunk.sh` | Anchor/no-extra-memory reference. |
| `run_ablation_framepack_weight_two_chunk.sh` | FramePack token weighting. |
| `run_ablation_framepack_len_r2_two_chunk.sh` | Length compression with ratio 2. |
| `run_ablation_framepack_len_r4_two_chunk.sh` | Length compression with ratio 4. |
| `run_ablation_framepack_hybrid_r2_weight_two_chunk.sh` | Ratio-2 length compression plus token weighting. |
| `run_ablation_framepack_hybrid_r4_weight_two_chunk.sh` | Ratio-4 length compression plus token weighting. |
| `run_ablation_spatial_inject_none_two_chunk.sh` | Spatial tokens are stored but not injected. |
| `run_ablation_spatial_concat_text_two_chunk.sh` | Spatial tokens are appended to text cross-attention keys/values. |
| `run_ablation_spatial_cross_attn_readout_two_chunk.sh` | Spatial tokens are read through a dedicated cross-attention read-out. |
| `run_ablation_videossm_hybrid_two_chunk.sh` | Legacy VideoSSM hybrid with the two-chunk monitor. |
| `run_ablation_block_wise_ssm_two_chunk.sh` | Paper-aligned block-wise SSM. |
| `run_all_ablations_two_chunk.sh` | Sequential launcher for the full ablation set. |

## Representative Baselines

The non-ablation baseline scripts are still useful for representative rows and quick training checks:

- `run_framepack_baseline.sh`: FramePack weight-only baseline.
- `run_framepack_lencompress_r2.sh`: FramePack length compression ratio 2.
- `run_framepack_lencompress_r4.sh`: FramePack length compression ratio 4.
- `run_spatial_memory_baseline.sh`: representative Spatial Memory baseline.
- `run_videossm_hybrid_baseline.sh`: legacy VideoSSM hybrid baseline.

These scripts expose common overrides through environment variables: `CKPT_INTERVAL`, `TIMESTEP_SHIFT`, `SAMPLING_INTERVAL_STEPS`, `SAMPLING_NUM_INFERENCE_STEPS`, `SAMPLING_HEIGHT`, `SAMPLING_WIDTH`, and `SAMPLING_NUM_FRAMES`.

## Context Construction

- `--context_source fov`: FOV-overlap retrieval over historical frames.
- `--context_source replay`: replay-style context construction aligned with `env/run_replay_loop_two_chunk.py`.
- `--context_source prev_chunk_tail`: continuous frames from `[start_frame - N, start_frame)` on disk.

The shared implementation lives in `src/model_training/context_chunk_utils.py`.

## Evaluation Alignment

Evaluation scripts source `env/eval_infer_alignment_env.sh` and call `env/memory_baseline_runtime.py` to infer the correct runtime memory flags from checkpoint paths. Keep output suffixes stable if you add a new script, or update `env/memory_baseline_runtime.py` so evaluation can recover the matching memory profile.

Generated training monitor videos are useful for fast visual checks, but paper-quality revisit panels should be produced through `eval/v2/revisit_suite`, which stores first frames, revisit-tail frames, change maps, and generated MP4 files for each case.
