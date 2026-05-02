# Eval v2

This folder keeps the public static and basic capability evaluations.

## Static Consistency

Run multi-chunk loop/revisit evaluation:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
export CKPT=/path/to/epoch-0.safetensors
bash eval/v2/run_static_consistency_loop_and_revisit.sh
```

Useful optional variables:

- `EVALS_ROOT`: output directory, defaults to `${CKPT_DIR}/evals_v2/static_consistency`.
- `NUM_SAMPLES_LOOP`: number of loop-closure samples.
- `NUM_SAMPLES_STATIC`: number of in-domain revisit samples.
- `MULTIVIEW_FIRSTFRAME_LIST`: text/jsonl/csv list for open-domain multiview revisit.
- `MULTIVIEW_FIRSTFRAME_DIR`: image directory used to generate the list automatically.
- `RUN_GEOMETRY_DIAG=1`: enable optional geometry diagnostics.

## Basic Capability

Run a single-video GT trajectory replay smoke test:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
export CKPT=/path/to/epoch-0.safetensors
bash eval/v2/run_basic_replay_gt.sh
```

Outputs include generated videos and `replay_gt_metrics.json` with MSE, PSNR, SSIM, and optional LPIPS.

Dynamic evaluation code is intentionally not included in this release.
