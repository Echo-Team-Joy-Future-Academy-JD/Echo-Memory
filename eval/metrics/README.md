# Memory Eval Metrics

本目录包含两类内容：

1. **数值 metric**：对 evals_ep0 已生成视频做后处理，产出六维度指标（见下）。
2. **可视化评测**：以**肉眼查看为主**，通过固定 **prompt** 与 **首 chunk/首帧**，在相同条件下生成视频，便于人工对比。见 [VISUAL_EVAL_DESIGN.md](VISUAL_EVAL_DESIGN.md) 与 [visual_eval_config.yaml](visual_eval_config.yaml)。

---

## 可视化评测（推荐先看）

不依赖 PSNR 等数值，通过设计 **prompt** 和 **首帧预设**，同一条件对比不同模型：

- **配置文件**：`visual_eval_config.yaml` — 定义多组 prompt（身份/长时/回环/状态等）与首帧预设（固定图或数据集某帧）。
- **设计说明**：`VISUAL_EVAL_DESIGN.md` — 设计思路、推荐组合、输出目录组织。
- **运行**：
  ```bash
  python3 -m eval_metrics.run_visual_eval --ckpt /path/to/epoch-0.safetensors --output_root /path/to/ckpt_dir/evals_visual
  ```
  输出在 `evals_visual/prompt_<id>_first_<id>/` 下，每个目录内有 2chunk/4chunk 的 MP4，直接打开文件夹查看即可。

---

## 数值 Metric（evals_ep0 后处理）

## Usage

```bash
# From repo root or exp1_4_4_cam_rt_paper_style
export EVALS_ROOT=/path/to/ckpt_dir/evals_ep0

# Run all six dimensions (default)
python -m eval_metrics.run_all_metrics --evals_root "$EVALS_ROOT"

# Run specific dimensions (e.g. 1, 2, 5)
python -m eval_metrics.run_all_metrics --evals_root "$EVALS_ROOT" --dims 1 2 5

# Optional: dataset for loop_closure trajectory reference; CLIP for identity
python -m eval_metrics.run_all_metrics --evals_root "$EVALS_ROOT" --dataset /path/to/Context-as-Memory-Dataset --use_clip --write_csv
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

## User Study (Long-Horizon Consistency)

To collect **User Study consistency scores** (1–5) for long sequences:

1. **Export list**: From `evals_root`, list all `*_gen_only.mp4` under `1_loop_4chunk/` and `3_multi_ctx_4chunk/` (e.g. `find "$EVALS_ROOT" - name "*_gen_only.mp4" > video_list.txt`).
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

For **WorldModelBench** or similar benchmarks: use their official data and evaluation protocol. This repo does not implement their scoring. To compare with evals_ep0 outputs, export generated videos to the format expected by the benchmark (e.g. fixed naming, resolution, length) and run the benchmark’s script externally. Document the mapping (which evals_ep0 run / ckpt corresponds to which benchmark run) in your own notes.

## Running a Single Dimension

Each module can be run standalone:

```bash
python -m eval_metrics.long_horizon_consistency --evals_root "$EVALS_ROOT" --output metrics/dim1.json
python -m eval_metrics.loop_closure --evals_root "$EVALS_ROOT" --output metrics/dim2.json
python -m eval_metrics.identity_preservation --evals_root "$EVALS_ROOT" --output metrics/dim3.json
python -m eval_metrics.state_tracking --evals_root "$EVALS_ROOT" --output metrics/dim4.json
python -m eval_metrics.temporal_coherence --evals_root "$EVALS_ROOT" --output metrics/dim5.json
python -m eval_metrics.semantic_consistency --evals_root "$EVALS_ROOT" --output metrics/dim6.json
```
