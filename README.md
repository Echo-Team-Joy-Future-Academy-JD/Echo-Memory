<div align="center">
<h1>🧠 Echo-Memory</h1>
<p><b>A Controlled Study of Memory in Action World Models</b></p>
<p><b>Echo Team @ Joy Future Academy, JD</b></p>
</div>

<div align="center">
<a href="https://doi.org/10.13140/RG.2.2.19906.34248"><img src="https://img.shields.io/badge/ResearchGate-DOI-00CCBB.svg" alt="ResearchGate DOI"></a>
<a href="https://arxiv.org/abs/TBD"><img src="https://img.shields.io/badge/arXiv-TBD-b31b1b.svg" alt="arXiv: TBD"></a>
<a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="CC BY 4.0"></a>
<a href="https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/"><img src="https://img.shields.io/badge/Project%20Page-Echo--Memory-green" alt="Project Page"></a>
<a href="https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/developer.html"><img src="https://img.shields.io/badge/Developer%20Guide-EN%2F中文-blue" alt="Developer Guide"></a>
<a href="https://huggingface.co/Echo-Team/Echo-Memory"><img src="https://img.shields.io/badge/🤗%20Checkpoints-Echo--Team%2FEcho--Memory-yellow" alt="Hugging Face checkpoints"></a>
<a href="https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Memory"><img src="https://img.shields.io/badge/GitHub-Echo--Memory-black" alt="GitHub repository"></a>
</div>

> **Core question.** When a generated scene must leave and later return, which kind of memory helps an action world model preserve **identity**, **layout**, and **viewpoint** instead of drifting into a plausible but different world?

<div align="center">
<img src="assets/paper_cases/figure_1_abs_framework.png?v=fig1-crop" alt="Echo-Memory paper teaser and workflow" width="92%">
</div>

<p align="center">
<b>Paper teaser.</b> Echo-Memory studies how Context, Compression, Spatial, and State-Space memory carry historical observations across chunk-wise action-world generation and revisit trajectories.
</p>

**Echo-Memory** is the release code for the paper's controlled memory study. It keeps the shared **Wan video backbone**, memory modules, training recipes, data utilities, open-domain revisit assets, and public replay/static evaluation suites.

**What is included:** reproducible memory rows, paper-aligned ablation scripts, GT replay, in-domain revisit, open-domain revisit, visual evidence frames, and representative videos.

**What is intentionally removed:** private dynamic benchmarks, cluster submit files, logs, generated outputs, and machine-local paths.

## News

**[2026/06]** [Developer Guide](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/developer.html) — bilingual dev workflows + Cursor vibe coding ([doc/DEVELOPER.md](doc/DEVELOPER.md)). Project page **EN / 中文** toggle.

**[2026/06]** Paper baseline checkpoints on Hugging Face: [Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory) (Wan 2.1 1.3B, epoch-0, 30k steps). See [doc/checkpoints.md](doc/checkpoints.md).

**[2026/06]** Report on [ResearchGate](https://doi.org/10.13140/RG.2.2.19906.34248) (CC BY 4.0) and [project page](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/) released.

**[2026/06]** Public code: **Wan 2.1 1.3B** memory ablations, replay/revisit eval, `eval/v2/revisit_suite/`, and `assets/opendomain_revisit/`.

## Roadmap

**Models**
- [x] **Wan 2.1 1.3B** backbone and public training recipes
- [x] Four memory families — **Context**, **Compression**, **Spatial**, **State-Space**
- [x] **Dynamic training pool** — SpatialVID subset export & settings ([doc](doc/dynamic_dataset_preprocessing.md))
- [x] **Paper checkpoints** — [Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory) on Hugging Face ([doc](doc/checkpoints.md))
- [ ] **Wan 2.2** and multi-scale **5B / 14B** backbones

**Eval**
- [ ] Dynamic evaluation beyond static replay/revisit
- [ ] More revisit probes and scoring presets

## Authors and Release Statement

This repository is released by **Echo Team @ Joy Future Academy, JD**. The code and evaluation assets are intended to support reproducible memory-mechanism comparisons for action-conditioned video world models. If you use this repository, please cite the Echo-Memory paper or acknowledge the Echo Team release.

## Visual Assets Included

This release directly includes paper-facing visual assets. Each example is a small diagnostic:

> **First frame → leave the view → revisit tail.**  
> The first frame fixes the world state, the trajectory moves away, and the revisit tail shows whether memory brings the model back to the same object, pose, background, and camera geometry.

```text
assets/opendomain_revisit/  Held-out first-frame sources for the open-domain toy-bear revisit probe
assets/paper_cases/         Paper teaser and memory overview figures
assets/readme_previews/     Low-resolution animated GIF previews for direct README playback
```

### 🧩 Memory Context List

<div align="center">
<table>
  <tr>
    <th>Family</th>
    <th>Memory row</th>
    <th>What it tests</th>
  </tr>
  <tr>
    <td><b>Floor</b></td>
    <td><b>No memory / I2V floor</b></td>
    <td>Re-generate from the first frame only; a lower bound for revisit consistency.</td>
  </tr>
  <tr>
    <td><b>Raw context</b></td>
    <td><b>Context K=1 / K=5 / K=20</b></td>
    <td>Whether simply keeping more recent frames is enough to prevent long-horizon drift.</td>
  </tr>
  <tr>
    <td><b>Compression</b></td>
    <td><b>Compression r = 4</b></td>
    <td>Whether a compact temporal representation can retain useful history without raw-frame growth.</td>
  </tr>
  <tr>
    <td><b>Spatial</b></td>
    <td><b>Spatial Memory</b></td>
    <td>Whether explicit spatial read/write state improves scene-layout recall.</td>
  </tr>
  <tr>
    <td><b>State-space</b></td>
    <td><b>Legacy Hybrid / Block-wise SSM</b></td>
    <td>Whether recurrent state updates can stabilize revisits beyond short context windows.</td>
  </tr>
</table>
</div>

<div align="center">
<img src="assets/paper_cases/figure_2_mem_overview.png" alt="Overview of four memory approaches" width="88%">
</div>

<p align="center">
<b>Memory design matrix.</b> The paper groups concrete variants by what is stored and how it is read back: raw context, compressed history, spatial state, or recurrent state-space memory.
</p>

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

### 🎬 Replay Video List

**Representative replay videos** are shown as compressed README previews. These clips replay ground-truth trajectories with each memory mechanism, making it easier to compare **local fidelity**, **motion smoothness**, and whether the generated chunk stays anchored to earlier visual evidence.

> **GitHub note:** the animated previews below are low-resolution GIFs for direct playback in the README.

<div align="center">
<table>
  <tr>
    <td align="center">
      <b>Context K=1</b><br>
      <img src="assets/readme_previews/context_k1_replay_gt.gif" width="140">
    </td>
    <td align="center">
      <b>Context K=5</b><br>
      <img src="assets/readme_previews/context_k5_replay_gt.gif" width="140">
    </td>
    <td align="center">
      <b>Compression r = 4</b><br>
      <img src="assets/readme_previews/framepack_len_r4_replay_gt.gif" width="140">
    </td>
    <td align="center">
      <b>Spatial Memory</b><br>
      <img src="assets/readme_previews/spatial_memory_replay_gt.gif" width="140">
    </td>
    <td align="center">
      <b>Legacy Hybrid</b><br>
      <img src="assets/readme_previews/ssm_legacy_replay_gt.gif" width="140">
    </td>
    <td align="center">
      <b>Block-wise SSM</b><br>
      <img src="assets/readme_previews/ssm_blockwise_replay_gt.gif" width="140">
    </td>
  </tr>
</table>
</div>

## Layout

```text
doc/                        Data pool download & preprocessing guides
diffsynth/                  Core model, pipeline, trainer utilities
src/model_training/         Main training code and memory/context helpers
src/model_inference/        Stage-2 inference entrypoints
src/data/                   Dataset metadata construction utilities
train/                      Public training recipes
eval/v2/                    Static consistency and basic GT replay eval
eval/metrics/               Visual/basic capability metrics
scripts/                    Data construction and latent precompute scripts
assets/opendomain_revisit/  Held-out first frames for open-domain revisit
assets/paper_cases/         Paper teaser and memory overview figures
assets/readme_previews/     README-friendly animated previews
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

## Quick Start

Evaluate a released checkpoint end-to-end in three steps:

```bash
# 1. Download the Wan 2.1 base model
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B

# 2. Download the static in-domain eval pool (~340 GB)
huggingface-cli download KlingTeam/Context-as-Memory-Dataset --repo-type dataset --local-dir ./data/Context-as-Memory-Dataset
export DATASET_BASE_PATH=./data/Context-as-Memory-Dataset
bash scripts/run_generate_metadata.sh

# 3. Download a checkpoint and run evaluation
huggingface-cli download Echo-Team/Echo-Memory spatial_mem/epoch-0.safetensors --local-dir ./ckpts

export WAN_BASE_MODEL=./Wan2.1-T2V-1.3B
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CKPT=./ckpts/spatial_mem/epoch-0.safetensors

bash eval/v2/run_basic_replay_gt.sh                        # single-video smoke test (~5 min)
bash eval/v2/run_static_consistency_loop_and_revisit.sh     # full paper eval bundle
```

Outputs are saved under `${CKPT_DIR}/evals_v2/`. See [Evaluation](#evaluation) for interpreting results.

## Required Paths

Most scripts are path-portable and use environment variables:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset   # static in-domain pool (code default)
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

`DATASET_BASE_PATH` points at whichever training pool you use (see [Data](#data)):

- **Static in-domain pool** — default `data/Context-as-Memory-Dataset` if unset; [doc/dataset_preprocessing.md](doc/dataset_preprocessing.md)
- **Dynamic training pool** — e.g. `data/dynamic-memory-dataset`; [doc/dynamic_dataset_preprocessing.md](doc/dynamic_dataset_preprocessing.md)

`WAN_BASE_MODEL` should contain `diffusion_pytorch_model.safetensors`, `models_t5_umt5-xxl-enc-bf16.pth`, and `Wan2.1_VAE.pth`.

## Checkpoints

Paper-aligned **epoch-0** fine-tunes (Wan 2.1 1.3B, **30,000 steps**):

**[Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory)** · full table & usage → [doc/checkpoints.md](doc/checkpoints.md)

| Family | Paper row | HF path | Steps |
| --- | --- | --- | ---: |
| Raw context | Context K=1 | [`context_k1/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/context_k1) | 30,000 |
| Raw context | Context K=20 | [`context_k20/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/context_k20) | 30,000 |
| Spatial | Spatial Memory | [`spatial_mem/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/spatial_mem) | 30,000 |
| State-space | Block-wise SSM | [`block_wise_ssm_two_chunk/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/block_wise_ssm_two_chunk) | 30,000 |
| State-space | Legacy Hybrid | [`videossm_hybrid/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/videossm_hybrid) | 30,000 |

Extended spatial / SSM ablation rows (+6) are listed in [doc/checkpoints.md](doc/checkpoints.md).

**Download & eval:**

```bash
huggingface-cli download Echo-Team/Echo-Memory spatial_mem/epoch-0.safetensors --local-dir ./ckpts
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CKPT=./ckpts/spatial_mem/epoch-0.safetensors
bash eval/v2/run_static_consistency_loop_and_revisit.sh
```

Keep the row folder name in `CKPT` so `env/memory_baseline_runtime.py` can recover the matching memory profile.

## Inference

Use the unified inference script for single-chunk generation with any memory family:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export PYTHONPATH=$PWD:${PYTHONPATH:-}

python src/model_inference/unified_inference.py \
    --ckpt ./ckpts/spatial_mem/epoch-0.safetensors \
    --prompt "A toy bear on a table, the camera rotates around it" \
    --output_path output.mp4
```

Switch memory type via `--memory_type` (default: `auto` — detects from checkpoint path):

| `--memory_type` | Family | Description |
|---|---|---|
| `auto` | (detected) | Auto-detect from checkpoint path |
| `no_memory` | Floor | No memory, I2V baseline |
| `context_k1` / `context_k5` / `context_k20` | Raw context | 1 / 5 / 20 context frames |
| `framepack_weight` | Compression | FramePack temporal decay reweighting |
| `framepack_len_r2` / `framepack_len_r4` | Compression | FramePack length compression ratio 2 / 4 |
| `spatial_mem` | Spatial | Spatial grid memory (64 tokens) |
| `videossm_hybrid` | State-space | Legacy hybrid SSM |
| `block_wise_ssm` | State-space | Block-wise recurrent SSM |

Add `--context_image` for first-frame conditioning and `--action_path` for camera trajectory control:

```bash
python src/model_inference/unified_inference.py \
    --ckpt ./ckpts/spatial_mem/epoch-0.safetensors \
    --memory_type spatial_mem \
    --context_image assets/opendomain_revisit/1774363417.png \
    --action_path env/action_rotation_left_45.json \
    --prompt "A toy bear on a table" \
    --output_path output.mp4
```

Full argument reference: `python src/model_inference/unified_inference.py --help`. Additional scripts and details in [`src/model_inference/README.md`](src/model_inference/README.md).

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

### Two-Chunk Training Paradigm

All memory baselines are trained in a **two-chunk** setup that simulates the revisit scenario:

- **Chunk 1 (context):** A clean reference segment encoded by the VAE. Context frames are sampled from the same video preceding the target segment and concatenated with the target latents at the suffix position.
- **Chunk 2 (target):** The noisy segment that the model learns to denoise. The memory mechanism operates on context latents to retain historical information across chunks.
- **Training-time monitoring:** The `--sampling_atomic_left_right` flag generates a left-45-degree then right-45-degree rotation pair during training for visual quality checks, using the same loop-closure probe used in evaluation.

This two-chunk structure forces the model to rely on memory when generating chunk 2, directly training the memory pathway that evaluation later tests.

### Hyperparameters

All memory baselines share a common training configuration:

| Parameter | Value | Notes |
|---|---:|---|
| Learning rate | 5e-5 | All memory rows |
| Batch size | 1 | Per device |
| Gradient accumulation | 1 | |
| Training epochs | 1 | ~30,000 steps on static pool |
| Resolution | 640 x 352 | Width x Height |
| Frames per chunk | 81 | ~5.4 s at 15 fps |
| Timestep shift | 15 | Memory baselines; context learning uses 5 |
| Optimizer | AdamW | Via `accelerate` |
| Backbone | Wan 2.1 T2V 1.3B | Full DiT trainable (`--trainable_models dit --save_full_model`) |
| T2V / I2V conditioning ratio | 0.10 / 0.10 | Classifier-free guidance target-only (`--cfg_target_only`) |

### Memory-Specific Parameters

Each memory family introduces its own flags on top of the shared configuration:

| Family | Key parameter | Values | Training flag |
|---|---|---|---|
| Raw context | `context_memory_frames` | 1 / 5 / 20 | `--context_memory_frames {1,5,20}` |
| FramePack weight | `context_temporal_decay` | 0.9 | `--use_framepack_memory --context_temporal_decay 0.9` |
| FramePack length | `framepack_ratio` | 2 or 4 | `--use_framepack_length_compress --framepack_ratio {2,4}` |
| FramePack hybrid | decay + ratio | 0.95 + 2 or 4 | Both `--use_framepack_memory` and `--use_framepack_length_compress` |
| Spatial memory | `spatial_memory_tokens` | 64 | `--use_spatial_memory --spatial_memory_tokens 64` |
| Spatial inject mode | `inject_mode` | concat_text / cross_attn_readout / none | `--spatial_memory_inject_mode {mode}` |
| SSM (block-wise) | block-wise recurrent | — | `--use_block_wise_ssm` |
| SSM (legacy hybrid) | legacy recurrent | — | `--use_videossm_hybrid` |

All training scripts share `train/_shared/common_env_memory.sh` for path resolution, environment setup, and common defaults.

## Data

Echo-Memory training and in-domain evaluation use two **training pools** that share the same on-disk layout. Set `DATASET_BASE_PATH` to the root of the pool you are using.

```text
{DATASET_BASE_PATH}/
├── frames/
├── jsons/
├── overlap_labels/      # recommended
├── metadata_full.csv
└── latents/             # optional
```

### Static in-domain pool

- **Source:** [KlingTeam/Context-as-Memory-Dataset](https://huggingface.co/datasets/KlingTeam/Context-as-Memory-Dataset) (~340 GB)
- **Local root:** `data/Context-as-Memory-Dataset` (code default when `DATASET_BASE_PATH` is unset)
- **Guide:** [doc/dataset_preprocessing.md](doc/dataset_preprocessing.md) — download, merge zip parts, verify layout

**Metadata (required):**

```bash
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
bash scripts/run_generate_metadata.sh
```

**Latents (optional):**

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
NUM_PROCESSES=8 bash scripts/run_precompute_ctx_target_latents.sh
```

### Dynamic training pool

- **Source:** simplified [SpatialVID/SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID) subset
- **Local root:** `data/dynamic-memory-dataset`
- **Guide:** [doc/dynamic_dataset_preprocessing.md](doc/dynamic_dataset_preprocessing.md) — download subset, export to Echo layout, training settings
- **Note:** `metadata_full.csv` is written at export; latent precompute uses the same script as the static pool if needed

### Open-domain assets

Held-out first frames for the open-domain revisit suite are already in `assets/opendomain_revisit/`; no download or construction step is required.

## Evaluation

In-domain replay and revisit eval use the **static in-domain pool** (`DATASET_BASE_PATH`). Run the paper evaluation bundle for a checkpoint:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
export CKPT=./ckpts/spatial_mem/epoch-0.safetensors
bash eval/v2/run_static_consistency_loop_and_revisit.sh
bash eval/v2/run_basic_replay_gt.sh
```

Run the open-domain revisit suite with the released first frames:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
PHASE=stage1 OOD_DIR=assets/opendomain_revisit \
  bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```

If an OpenAI-compatible VLM endpoint is available, add `PHASE=vlm` or run the default `PHASE=all` with `VLM_API_BASE` and `VLM_MODEL`.

### Evaluation Types

The evaluation suite has three complementary tiers, from fast smoke test to full generalization check:

| Eval type | Script | What it tests | When to use |
|---|---|---|---|
| **Basic replay** | `run_basic_replay_gt.sh` | Single-video GT trajectory fidelity. Per-frame comparison against ground-truth. | Smoke test: does the model follow the ground-truth camera path? |
| **Static consistency** | `run_static_consistency_loop_and_revisit.sh` | Multi-chunk loop closure (leave and return to the same pose) and action-combo revisit. | Paper-level evaluation: memory mechanism comparison on revisit consistency. |
| **Open-domain revisit** | `revisit_suite/run_one_click_revisit_eval.sh` | Held-out first frames not in training data. Tests whether memory generalizes to unseen scenes. | Generalization check: does memory help on new images? |

Basic replay validates action control; static consistency isolates memory quality; open-domain revisit tests generalization.

### Metrics

| Metric | Full name | Measures | Range | Better |
|---|---|---|---|---|
| **MSE** | Mean Squared Error | Per-pixel difference between generated and GT frames | 0 -- inf | Lower |
| **PSNR** | Peak Signal-to-Noise Ratio | Signal reconstruction quality (log-scale of MSE) | 0 -- ~50 dB | Higher |
| **SSIM** | Structural Similarity Index | Structural similarity in luminance, contrast, and structure | -1 -- 1 | Higher |
| **LPIPS** | Learned Perceptual Image Patch Similarity | Perceptual distance using deep feature representations | 0 -- 1 | Lower |
| **FID** | Frechet Inception Distance | Distribution-level realism of generated images | 0 -- inf | Lower |
| **FVD** | Frechet Video Distance | Distribution-level temporal quality of generated video | 0 -- inf | Lower |

### Interpreting Results

- **Basic replay** outputs `replay_gt_metrics.json` with per-frame and aggregate MSE, PSNR, SSIM. PSNR above ~25 dB and SSIM above ~0.7 indicate reasonable single-chunk fidelity.
- **Static consistency** outputs per-sample revisit metrics under `evals_v2/static_consistency/`. Compare first-frame-vs-revisit-tail MSE across memory rows: lower MSE means the model better preserved the original scene on return.
- **Open-domain revisit** outputs frame pairs and optional VLM scores. Compare across memory families to assess which mechanism generalizes best to unseen scenes.

## Capability Metrics

```bash
python eval/metrics/run_all_metrics.py --help
python eval/metrics/run_visual_eval.py --help
```

Dynamic evaluation is not part of this release.

## Community

Project page supports **EN / 中文** — [echo-team-joy-future-academy-jd.github.io/Echo-Memory](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/)

Maintainers: [Developer Guide](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/developer.html) · [doc/DEVELOPER.md](doc/DEVELOPER.md) · Cursor skills in [`.cursor/skills/`](.cursor/skills/)

<div align="center">
<img src="assets/wechat_group_qrcode.jpg" alt="Echo-Memory WeChat group" width="1166" height="1640" style="width:240px;height:auto;max-width:100%;">
<p><b>Echo-Memory 交流群</b> — scan to join (QR refreshes periodically)</p>
</div>

## Citation

If you use this repository or the Echo-Memory report, please cite:

**ResearchGate (June 2026)** · DOI [10.13140/RG.2.2.19906.34248](https://doi.org/10.13140/RG.2.2.19906.34248) · Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

```bibtex
@article{king2026echomemory,
  title={Echo-Memory: A Controlled Study of Memory in Action World Models},
  author={King, Wayne and Xue, Zeyue and Bian, Yuxuan and Huang, Jie and Li, Haoran and Li, Yaowei and Su, Yaofeng and Li, Yuming and Wang, Haoyu and Zhang, Shiyi and Zhang, Songchun and Niu, Yuwei and Xu, Sihan and Zhuang, Junhao and Huang, Haoyang and Duan, Nan},
  journal={Echo-Memory technical report},
  publisher={ResearchGate},
  year={2026},
  month={jun},
  doi={10.13140/RG.2.2.19906.34248},
  url={https://doi.org/10.13140/RG.2.2.19906.34248},
  note={Licensed under CC BY 4.0}
}
```

arXiv preprint forthcoming — use the BibTeX above until posted.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Echo-Team-Joy-Future-Academy-JD/Echo-Memory&type=date&legend=bottom-right)](https://www.star-history.com/#Echo-Team-Joy-Future-Academy-JD/Echo-Memory&type=date&legend=bottom-right)
