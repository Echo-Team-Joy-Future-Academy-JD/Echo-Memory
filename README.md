<div align="center">
<h1>🧠 Echo-Memory</h1>
<p><b>A Controlled Study of Memory in Action World Models</b></p>
<p><b>Echo Team @ Joy Future Academy, JD</b></p>
</div>

<div align="center">
<a href="#"><img src="https://img.shields.io/badge/arXiv-TBD-b31b1b.svg" alt="arXiv: TBD"></a>
<a href="#"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-TBD-blue" alt="Hugging Face: TBD"></a>
<a href="#"><img src="https://img.shields.io/badge/Project%20Page-TBD-green" alt="Project Page: TBD"></a>
<a href="https://github.com/WayneJin0918/Echo-Memory"><img src="https://img.shields.io/badge/GitHub-Echo--Memory-black" alt="GitHub repository"></a>
</div>

<div align="center">
<img src="assets/paper_cases/representative_sweep_panel.png" alt="Representative open-domain revisit sweep" width="92%">
</div>

<p align="center">
Representative open-domain revisit case: input/return evidence across no-memory, raw context, compression, spatial memory, and state-space memory variants.
</p>

Echo-Memory is the release code for controlled memory studies in action-conditioned video world models. It keeps the shared Wan video backbone, memory modules, training recipes, data utilities, open-domain revisit assets, and public replay/static evaluation suites used by the paper.

Private dynamic benchmarks, cluster submit files, logs, generated outputs, and machine-local paths are intentionally removed. Public GT replay, in-domain revisit, and open-domain revisit evaluation are retained.

## Authors and Release Statement

This repository is released by **Echo Team @ Joy Future Academy, JD**. The code and evaluation assets are intended to support reproducible memory-mechanism comparisons for action-conditioned video world models. If you use this repository, please cite the Echo-Memory paper or acknowledge the Echo Team release.

## Visual Assets Included

This release directly includes paper-facing visual assets:

```text
assets/opendomain_revisit/  Held-out first-frame sources for the open-domain toy-bear revisit probe
assets/paper_cases/         Representative qualitative panels and per-method first/tail evidence frames
assets/paper_videos/        Representative GT-replay MP4s for key memory rows
assets/ood_ssm_cases/       Open-domain right45 return videos for legacy and block-wise SSM
```

<div align="center">
<table>
  <tr>
    <td align="center"><img src="assets/opendomain_revisit/1774363417.png" width="180"><br>Open-domain source 1</td>
    <td align="center"><img src="assets/opendomain_revisit/1774363487.png" width="180"><br>Open-domain source 2</td>
    <td align="center"><img src="assets/opendomain_revisit/1774363572.png" width="180"><br>Open-domain source 3</td>
    <td align="center"><img src="assets/opendomain_revisit/1774369504.png" width="180"><br>Open-domain source 4</td>
  </tr>
  <tr>
    <td align="center"><img src="assets/opendomain_revisit/1774369548.png" width="180"><br>Open-domain source 5</td>
    <td align="center"><img src="assets/opendomain_revisit/1774369942.png" width="180"><br>Open-domain source 6</td>
    <td align="center"><img src="assets/opendomain_revisit/1774370005.png" width="180"><br>Open-domain source 7</td>
    <td align="center"><img src="assets/opendomain_revisit/1774370010.png" width="180"><br>Open-domain source 8</td>
  </tr>
</table>
</div>

Representative replay videos are also included as MP4 files:

<div align="center">
<table>
  <tr>
    <td align="center">
      <video src="assets/paper_videos/context_k1_replay_gt.mp4" controls muted width="260"></video><br>
      <a href="assets/paper_videos/context_k1_replay_gt.mp4">Context K=1 replay</a>
    </td>
    <td align="center">
      <video src="assets/paper_videos/context_k5_replay_gt.mp4" controls muted width="260"></video><br>
      <a href="assets/paper_videos/context_k5_replay_gt.mp4">Context K=5 replay</a>
    </td>
    <td align="center">
      <video src="assets/paper_videos/context_k20_replay_gt.mp4" controls muted width="260"></video><br>
      <a href="assets/paper_videos/context_k20_replay_gt.mp4">Context K=20 replay</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <video src="assets/paper_videos/framepack_len_r4_replay_gt.mp4" controls muted width="260"></video><br>
      <a href="assets/paper_videos/framepack_len_r4_replay_gt.mp4">FramePack length r=4 replay</a>
    </td>
    <td align="center">
      <video src="assets/paper_videos/spatial_memory_replay_gt.mp4" controls muted width="260"></video><br>
      <a href="assets/paper_videos/spatial_memory_replay_gt.mp4">Spatial Memory replay</a>
    </td>
    <td align="center">
      <video src="assets/paper_videos/ssm_blockwise_replay_gt.mp4" controls muted width="260"></video><br>
      <a href="assets/paper_videos/ssm_blockwise_replay_gt.mp4">Block-wise SSM replay</a>
    </td>
  </tr>
</table>
</div>

## Layout

```text
diffsynth/                  Core model, pipeline, trainer utilities
src/model_training/         Main training code and memory/context helpers
src/model_inference/        Stage-2 inference entrypoints
src/data/                   Dataset metadata construction utilities
train/                      Public training recipes
eval/v2/                    Static consistency and basic GT replay eval
eval/metrics/               Visual/basic capability metrics
scripts/                    Data construction and latent precompute scripts
assets/opendomain_revisit/  Held-out first frames for open-domain revisit
assets/paper_cases/         Paper qualitative panels and per-method visual evidence
assets/paper_videos/        Representative paper replay videos
assets/ood_ssm_cases/       OOD SSM revisit videos and evidence frames
env/                        Shared runtime helpers and action JSONs
tests/                      Focused checks for memory/context plumbing
```

## Installation

```bash
conda env create -f environment.yml
conda activate echo-memory
pip install -r requirements.txt
```

If your CUDA/Torch stack requires a custom `flash-attn` wheel, install it after the base environment is ready.

Configure `accelerate` for your machine before multi-GPU training:

```bash
accelerate config
```

## Required Paths

Most scripts are path-portable and use environment variables:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

`WAN_BASE_MODEL` should contain `diffusion_pytorch_model.safetensors`, `models_t5_umt5-xxl-enc-bf16.pth`, and `Wan2.1_VAE.pth`.

## Training

Memory baseline recipes live in `train/memory_baselines_basic/`. These map to the paper matrix:

```bash
bash train/memory_baselines_basic/run_ablation_no_memory_baseline_two_chunk.sh
bash train/memory_baselines_basic/run_ablation_framepack_weight_two_chunk.sh
bash train/memory_baselines_basic/run_ablation_framepack_len_r2_two_chunk.sh
bash train/memory_baselines_basic/run_ablation_framepack_len_r4_two_chunk.sh
bash train/memory_baselines_basic/run_ablation_framepack_hybrid_r2_weight_two_chunk.sh
bash train/memory_baselines_basic/run_ablation_framepack_hybrid_r4_weight_two_chunk.sh
bash train/memory_baselines_basic/run_spatial_memory_baseline.sh
bash train/memory_baselines_basic/run_ablation_spatial_inject_none_two_chunk.sh
bash train/memory_baselines_basic/run_ablation_spatial_concat_text_two_chunk.sh
bash train/memory_baselines_basic/run_ablation_spatial_cross_attn_readout_two_chunk.sh
bash train/memory_baselines_basic/run_videossm_hybrid_baseline.sh
bash train/memory_baselines_basic/run_ablation_block_wise_ssm_two_chunk.sh
```

Context learning recipes live in `train/context_learning/`:

```bash
bash train/context_learning/run_pre_qkv_ctx1.sh
bash train/context_learning/run_pre_qkv_ctx5.sh
bash train/context_learning/run_pre_qkv_ctx20.sh
```

Outputs default to `outputs/`. Override with `OUTPUT_BASE_ROOT=/path/to/outputs`.

## Data Construction

Generate metadata for a Context-as-Memory style dataset:

```bash
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
bash scripts/run_generate_metadata.sh
```

Precompute context and target latents:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
NUM_PROCESSES=8 bash scripts/run_precompute_ctx_target_latents.sh
```

## Evaluation

Run the paper evaluation bundle for a checkpoint:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
export CKPT=/path/to/epoch-0.safetensors
bash eval/v2/run_static_consistency_loop_and_revisit.sh
bash eval/v2/run_basic_replay_gt.sh
```

Run the open-domain revisit suite with the released first frames:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
PHASE=stage1 OOD_DIR=assets/opendomain_revisit \
  bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```

If an OpenAI-compatible VLM endpoint is available, add `PHASE=vlm` or run the default `PHASE=all` with `VLM_API_BASE` and `VLM_MODEL`.

## Visualization and Paper Cases

This repository directly includes the qualitative assets used to inspect the paper's representative open-domain case. The most compact panel is available at:

```text
assets/paper_cases/representative_sweep_panel.png
assets/paper_cases/representative_sweep_panel_highres.png
assets/paper_cases/manifest.csv
```

<div align="center">
<table>
  <tr>
    <th>Variant</th>
    <th>First frame</th>
    <th>Revisit tail</th>
  </tr>
  <tr>
    <td>No memory / I2V floor</td>
    <td><img src="assets/paper_cases/01_no_memory_i2v_floor_first.png" width="210"></td>
    <td><img src="assets/paper_cases/01_no_memory_i2v_floor_tail.png" width="210"></td>
  </tr>
  <tr>
    <td>Context K=5</td>
    <td><img src="assets/paper_cases/02_context_k5_first.png" width="210"></td>
    <td><img src="assets/paper_cases/02_context_k5_tail.png" width="210"></td>
  </tr>
  <tr>
    <td>Context K=20</td>
    <td><img src="assets/paper_cases/03_context_k20_first.png" width="210"></td>
    <td><img src="assets/paper_cases/03_context_k20_tail.png" width="210"></td>
  </tr>
  <tr>
    <td>FramePack length r=4</td>
    <td><img src="assets/paper_cases/04_framepack_len_r4_first.png" width="210"></td>
    <td><img src="assets/paper_cases/04_framepack_len_r4_tail.png" width="210"></td>
  </tr>
  <tr>
    <td>Spatial Memory</td>
    <td><img src="assets/paper_cases/05_spatial_memory_first.png" width="210"></td>
    <td><img src="assets/paper_cases/05_spatial_memory_tail.png" width="210"></td>
  </tr>
  <tr>
    <td>State-Space legacy hybrid</td>
    <td><img src="assets/paper_cases/06_ssm_legacy_first.png" width="210"></td>
    <td><img src="assets/paper_cases/06_ssm_legacy_tail.png" width="210"></td>
  </tr>
  <tr>
    <td>State-Space block-wise</td>
    <td><img src="assets/paper_cases/07_ssm_blockwise_first.png" width="210"></td>
    <td><img src="assets/paper_cases/07_ssm_blockwise_tail.png" width="210"></td>
  </tr>
</table>
</div>

The release also includes representative GT-replay videos copied from the paper materials:

```text
assets/paper_videos/context_k1_replay_gt.mp4
assets/paper_videos/context_k5_replay_gt.mp4
assets/paper_videos/context_k20_replay_gt.mp4
assets/paper_videos/framepack_len_r4_replay_gt.mp4
assets/paper_videos/spatial_memory_replay_gt.mp4
assets/paper_videos/ssm_legacy_replay_gt.mp4
assets/paper_videos/ssm_blockwise_replay_gt.mp4
```

Open-domain SSM revisit videos are included separately for the memory-sensitive `right45_return_2chunk` probe:

<div align="center">
<table>
  <tr>
    <th>Legacy VideoSSM Hybrid</th>
    <th>Block-wise SSM</th>
  </tr>
  <tr>
    <td align="center">
      <video src="assets/ood_ssm_cases/legacy_videossm/revisit_gen_only.mp4" controls muted width="360"></video><br>
      <a href="assets/ood_ssm_cases/legacy_videossm/revisit_gen_only.mp4">Open-domain VideoSSM hybrid return video</a>
    </td>
    <td align="center">
      <video src="assets/ood_ssm_cases/blockwise_ssm/revisit_gen_only.mp4" controls muted width="360"></video><br>
      <a href="assets/ood_ssm_cases/blockwise_ssm/revisit_gen_only.mp4">Open-domain block-wise SSM return video</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/ood_ssm_cases/legacy_videossm/first_00.png" width="170">
      <img src="assets/ood_ssm_cases/legacy_videossm/revisit_tail_03.png" width="170"><br>
      First frame and final revisit tail
    </td>
    <td align="center">
      <img src="assets/ood_ssm_cases/blockwise_ssm/first_00.png" width="170">
      <img src="assets/ood_ssm_cases/blockwise_ssm/revisit_tail_03.png" width="170"><br>
      First frame and final revisit tail
    </td>
  </tr>
</table>
</div>

The evaluation scripts also write videos and evidence images next to their metrics, so newly generated cases can be inspected without any extra conversion step.

GT replay videos:

```text
${CKPT_DIR}/evals_v2/basic/replay_gt/<video>_start<frame>/replay_gt_gen_only.mp4
${CKPT_DIR}/evals_v2/static_consistency/in_domain/long_horizon_gt_replay/*/replay_gt_gen_only.mp4
```

In-domain loop and open-domain revisit videos:

```text
${CKPT_DIR}/evals_v2/static_consistency/in_domain/loop_closure/**/*.mp4
${CKPT_DIR}/evals_v2/static_consistency/in_domain/combo_revisit_in_domain/**/*.mp4
${CKPT_DIR}/evals_v2/static_consistency/open_domain/multiview_revisit/**/*.mp4
eval_outputs/revisit_suite_<timestamp>/stage1/**/revisit_gen_only.mp4
```

Open-domain revisit evidence images, useful for paper figures and qualitative panels:

```text
eval_outputs/revisit_suite_<timestamp>/stage1/**/stage1_frames/first_00.png
eval_outputs/revisit_suite_<timestamp>/stage1/**/stage1_frames/revisit_tail_*.png
eval_outputs/revisit_suite_<timestamp>/stage1/**/stage1_frames/first_last_chunk_changes/*.png
```

To collect paper-ready case tables and image manifests from one or more revisit runs:

```bash
python eval/v2/revisit_suite/export_revisit_materials.py \
  --eval-root eval_outputs/revisit_suite_<timestamp> \
  --out-dir paper_case_materials \
  --prefix echo_memory_revisit
```

For lightweight browsing on a remote machine, serve an output folder and open the URL from your workstation:

```bash
python -m http.server 8000 --directory eval_outputs
```

The released first-frame sources for the open-domain paper cases live in `assets/opendomain_revisit/`. They are the input images behind the toy-bear revisit probe; generated videos and VLM evidence frames are produced by `eval/v2/revisit_suite`.

If MP4 files are generated on a remote machine, open them through the file browser or serve the output directory:

```bash
python -m http.server 8000 --directory eval_outputs
```

Then open `http://<host>:8000/` and navigate to `revisit_suite_<timestamp>/stage1/.../revisit_gen_only.mp4`.

Capability metrics:

```bash
python eval/metrics/run_all_metrics.py --help
python eval/metrics/run_visual_eval.py --help
```

Dynamic evaluation and training are not part of this release.
