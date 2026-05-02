# Echo-Memory

Echo-Memory is a release-ready codebase for context-as-memory video generation experiments. This export keeps the core model code, memory baseline training recipes, data construction utilities, inference entrypoints, and public static/basic evaluation suites.

Dynamic evaluation, dynamic benchmark training scripts, cluster submit files, logs, outputs, and local machine paths are intentionally removed.

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

## Required Paths

Most scripts are path-portable and use environment variables:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

`WAN_BASE_MODEL` should contain `diffusion_pytorch_model.safetensors`, `models_t5_umt5-xxl-enc-bf16.pth`, and `Wan2.1_VAE.pth`.

## Training

Memory baseline recipes live in `train/memory_baselines_basic/`:

```bash
bash train/memory_baselines_basic/run_spatial_memory_baseline.sh
bash train/memory_baselines_basic/run_videossm_hybrid_baseline.sh
bash train/memory_baselines_basic/run_ablation_block_wise_ssm_two_chunk.sh
bash train/memory_baselines_basic/run_framepack_baseline.sh
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

Static consistency and GT replay:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
export CKPT=/path/to/epoch-0.safetensors
bash eval/v2/run_static_consistency_loop_and_revisit.sh
bash eval/v2/run_basic_replay_gt.sh
```

Capability metrics:

```bash
python eval/metrics/run_all_metrics.py --help
python eval/metrics/run_visual_eval.py --help
```

Dynamic evaluation and training are not part of this release.
