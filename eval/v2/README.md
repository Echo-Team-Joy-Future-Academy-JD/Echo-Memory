# Eval v2

This folder keeps the public replay, in-domain loop/revisit, and open-domain revisit evaluations.

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
- `MULTIVIEW_FIRSTFRAME_LIST`: text/jsonl/csv list for open-domain first frames.
- `MULTIVIEW_FIRSTFRAME_DIR`: image directory used to generate the list automatically, for example `assets/opendomain_revisit`.
- `RUN_GEOMETRY_DIAG=1`: enable optional geometry diagnostics.

The script writes in-domain outputs under `${EVALS_ROOT}/in_domain` and open-domain outputs under `${EVALS_ROOT}/open_domain`.

## Basic Capability

Run a single-video GT trajectory replay smoke test:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
export CKPT=/path/to/epoch-0.safetensors
bash eval/v2/run_basic_replay_gt.sh
```

Outputs include generated videos and `replay_gt_metrics.json` with MSE, PSNR, SSIM, and optional LPIPS.

## Open-Domain Revisit + VLM

For the paper-style open-domain return probe, use the one-click revisit suite:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
PHASE=stage1 OOD_DIR=assets/opendomain_revisit \
  bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```

Set `PHASE=vlm` with `EVAL_ROOT=/path/to/eval_root` to score an existing stage-1 run. The scorer expects an OpenAI-compatible endpoint via `VLM_API_BASE` and `VLM_MODEL`; use `VLM_DRY_RUN=1` for a wiring check.

## Visual Outputs

Every evaluation path keeps human-readable artifacts:

- GT replay writes `replay_gt_gen_only.mp4`, `replay_gt_metrics.json`, and optional per-frame CSV files.
- Static loop/revisit writes generated MP4 files under `in_domain/loop_closure` and `in_domain/combo_revisit_in_domain`.
- Open-domain multiview writes MP4 files under `open_domain/multiview_revisit`.
- The one-click revisit suite writes `stage1_frames/first_00.png`, `stage1_frames/revisit_tail_*.png`, optional first-vs-last chunk change maps, and `revisit_gen_only.mp4` for each case.

For paper figures, start from the one-click revisit suite outputs:

```bash
python eval/v2/revisit_suite/export_revisit_materials.py \
  --eval-root eval_outputs/revisit_suite_<timestamp> \
  --out-dir paper_case_materials \
  --prefix echo_memory_revisit
```

The exported material directory contains flattened case metadata plus image references that can be used to build qualitative grids.

Dynamic/private evaluation code is intentionally not included in this release.
