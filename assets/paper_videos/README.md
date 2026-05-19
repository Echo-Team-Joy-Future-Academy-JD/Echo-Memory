# Paper Video Assets

This directory contains representative GT-replay videos copied from the paper materials for direct README display.

Included rows:

- `context_k1_replay_gt.mp4`
- `context_k5_replay_gt.mp4`
- `context_k20_replay_gt.mp4`
- `framepack_len_r4_replay_gt.mp4`
- `spatial_memory_replay_gt.mp4`
- `ssm_legacy_replay_gt.mp4`
- `ssm_blockwise_replay_gt.mp4`

These are static qualitative examples. To regenerate videos for a new checkpoint, run:

```bash
export CKPT=/path/to/epoch-0.safetensors
bash eval/v2/run_basic_replay_gt.sh
```
