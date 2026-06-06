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

**[2026/06]** Project page **EN / 中文** toggle + [Developer Guide](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/developer.html) ([doc/DEVELOPER.md](doc/DEVELOPER.md)).

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

## Capability Metrics

```bash
python eval/metrics/run_all_metrics.py --help
python eval/metrics/run_visual_eval.py --help
```

Dynamic evaluation is not part of this release.

## Community

Project page supports **EN / 中文** — [echo-team-joy-future-academy-jd.github.io/Echo-Memory](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/)

Maintainers: [Developer Guide (EN/中文)](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/developer.html) · [doc/DEVELOPER.md](doc/DEVELOPER.md)

<div align="center">
<img src="assets/wechat_group_qrcode.jpg" alt="Echo-Memory WeChat group" width="200">
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
