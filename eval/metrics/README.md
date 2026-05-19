# Memory Eval Metrics

This folder contains two evaluation layers:

1. **Numeric post-processing** for `evals_v2` or other generated-video folders.
2. **Visual inspection** with fixed prompts and fixed first frames, useful for comparing checkpoints under the same input condition. See [VISUAL_EVAL_DESIGN.md](VISUAL_EVAL_DESIGN.md) and [visual_eval_config.yaml](visual_eval_config.yaml).

---

## Visual Inspection

This route is intentionally human-readable. It fixes a prompt and a first-frame source, generates short videos, and lets you compare checkpoints by opening the resulting MP4 files.

- **Config**: `visual_eval_config.yaml` defines prompt sets and first-frame presets.
- **Design note**: `VISUAL_EVAL_DESIGN.md` explains recommended case groups and output layout.
- **Run**:
  ```bash
  python3 eval/metrics/run_visual_eval.py --ckpt /path/to/epoch-0.safetensors --output_root /path/to/ckpt_dir/evals_visual
  ```
  Outputs are written under `evals_visual/prompt_<id>_first_<id>/`; each case folder contains 2-chunk or 4-chunk MP4 files.

---

## Numeric Metrics

## Usage

```bash
export EVALS_ROOT=/path/to/ckpt_dir/evals_v2/static_consistency

# Run all six dimensions.
python eval/metrics/run_all_metrics.py --evals_root "$EVALS_ROOT"

# Run specific dimensions.
python eval/metrics/run_all_metrics.py --evals_root "$EVALS_ROOT" --dims 1 2 5

# Optional: dataset for loop-closure trajectory reference; CLIP for identity.
python eval/metrics/run_all_metrics.py --evals_root "$EVALS_ROOT" --dataset /path/to/Context-as-Memory-Dataset --use_clip --write_csv
```

Results are written to `evals_root/metrics/` by default (or `--output_dir`): per-dimension `*.json` and `all_metrics_summary.json`. Use `--write_csv` to also write `aggregate_summary.csv`.

## Dimensions

| Dim | Name | Metrics (Phase 1) | Optional |
|-----|------|-------------------|----------|
| 1 | Long-Horizon Consistency | Stable sequence length, frame-to-frame drift rate | User Study: see below |
| 2 | Loop Closure / Revisit | View Recall PSNR, View Recall SSIM | Trajectory ref error (when dataset provided) |
| 3 | Identity Preservation | CLIP consistency (or simple embedding) | Face Embedding, character ID (insightface/torchreid) |
| 4 | State Tracking | Consecutive displacement, large-jump fraction | Detection+tracking, VLM state accuracy |
| 5 | Temporal Coherence | Frame-to-frame PSNR | Optical flow consistency, FVD |
| 6 | Semantic/Logic Consistency | Rule-based physics violation rate | VLM common-sense, WorldModelBench |

## Paper Case and Video Access

For paper figures, prefer outputs from `eval/v2/revisit_suite` because each case stores the input frame, revisit-tail evidence frames, and the generated video in one directory:

```text
eval_outputs/revisit_suite_<timestamp>/stage1/<run_id>/<domain>/<sample_id>/<mode>/
```

Useful files:

- `revisit_gen_only.mp4`: generated return trajectory.
- `stage1_frames/first_00.png`: source view.
- `stage1_frames/revisit_tail_*.png`: final return frames.
- `stage1_frames/first_last_chunk_changes/*.png`: optional visual change maps.
- `stage1_metrics.json` and `vlm_score.json`: case-level metrics and VLM scores.

Serve the output folder when reviewing videos remotely:

```bash
python -m http.server 8000 --directory eval_outputs
```

## User Study (Long-Horizon Consistency)

To collect **User Study consistency scores** (1–5) for long sequences:

1. **Export list**: From `evals_root`, list all `*_gen_only.mp4` files, for example `find "$EVALS_ROOT" -name "*_gen_only.mp4" > video_list.txt`.
2. **Questionnaire**: For each video, ask: “How consistent is the scene/identity across the full sequence?” (1 = very inconsistent, 5 = very consistent).
3. **Summary**: Store responses in a CSV with columns e.g. `video_path,score`. Aggregate: mean and std of `score` per run or per model.

No automatic scoring is implemented; the pipeline only provides the list and this procedure.

## Optional Dependencies

- **Phase 1** (no extra deps): numpy, opencv-python, PIL; skimage for PSNR/SSIM (recommended).
- **Optional**: 
  - `scikit-image` — PSNR/SSIM in loop_closure and temporal_coherence.
  - CLIP (diffsynth ImageQualityMetric) — `--use_clip` in identity_preservation (requires model weights under `models/QualityMetric/`).
  - Face / ReID: `insightface`, `torchreid` — for identity_preservation Face Embedding and character ID (placeholders in code).
  - Optical flow: RAFT or `torchvision.optical_flow` — for temporal_coherence flow consistency (placeholder).
  - FVD: `pytorch-fvd` or I3D — for temporal_coherence FVD (placeholder).
  - VLM: local or API — for semantic_consistency common-sense/physics (placeholder).

Save optional deps to a separate file if needed, e.g. `requirements-optional.txt`:

```
scikit-image
# insightface
# torchreid
```

## WorldModelBench

For **WorldModelBench** or similar benchmarks: use their official data and evaluation protocol. This repo does not implement their scoring. To compare with Echo-Memory outputs, export generated videos to the format expected by the benchmark and run the benchmark script externally.

## Running a Single Dimension

Each module can be run standalone:

```bash
python eval/metrics/long_horizon_consistency.py --evals_root "$EVALS_ROOT" --output metrics/dim1.json
python eval/metrics/loop_closure.py --evals_root "$EVALS_ROOT" --output metrics/dim2.json
python eval/metrics/identity_preservation.py --evals_root "$EVALS_ROOT" --output metrics/dim3.json
python eval/metrics/state_tracking.py --evals_root "$EVALS_ROOT" --output metrics/dim4.json
python eval/metrics/temporal_coherence.py --evals_root "$EVALS_ROOT" --output metrics/dim5.json
python eval/metrics/semantic_consistency.py --evals_root "$EVALS_ROOT" --output metrics/dim6.json
```
