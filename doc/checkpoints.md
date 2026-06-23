# Checkpoints (Hugging Face)

**Repo:** [Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory)

Fine-tuned DiT weights on top of [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B). Released rows are saved as `{row_id}/epoch-0.safetensors` after **1 epoch / 30,000 steps** on the static in-domain pool (640×352, 81-frame chunks). Mechanism names follow [memory_mechanisms.md](memory_mechanisms.md).

## Checkpoint index

| Family | Paper row | HF path | Steps | Echo-Memory recipe |
| --- | --- | --- | ---: | --- |
| Raw context | Context K=1 | [`context_k1/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/context_k1) | 30,000 | `train/context_learning/run_pre_qkv_ctx1.sh` |
| Raw context | Context K=20 | [`context_k20/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/context_k20) | 30,000 | `train/context_learning/run_pre_qkv_ctx20.sh` |
| Spatial | Spatial Memory | [`spatial_mem/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/spatial_mem) | 30,000 | `train/memory_baselines_basic/run_spatial_memory_baseline.sh` |
| State-space | Block-wise SSM | TODO | TODO | `train/memory_baselines_basic/run_ablation_block_wise_ssm_two_chunk.sh` |
| State-space | Legacy Hybrid (VideoSSM) | TODO | TODO | `train/memory_baselines_basic/run_videossm_hybrid_baseline.sh` |
| Spatial | concat text (ablation) | [`spatial_concat_text_two_chunk/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/spatial_concat_text_two_chunk) | 30,000 | `train/memory_baselines_basic/run_ablation_spatial_concat_text_two_chunk.sh` |
| Spatial | inject none (ablation) | [`spatial_inject_none_two_chunk/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/spatial_inject_none_two_chunk) | 30,000 | `train/memory_baselines_basic/run_ablation_spatial_inject_none_two_chunk.sh` |
| Spatial | cross-attn t32 (ablation) | [`spatial_cross_attn_readout_t32_g4_two_chunk/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/spatial_cross_attn_readout_t32_g4_two_chunk) | 30,000 | `train/memory_baselines_basic/run_ablation_spatial_cross_attn_readout_two_chunk.sh` |
| State-space | SSM ctx1 / every4 / hint21 | TODO | TODO | SSM ablation |
| State-space | SSM ctx5 / every1 / hint21 | TODO | TODO | SSM ablation |
| State-space | SSM ctx5 / every4 / hint81 | TODO | TODO | SSM ablation |

Context K=5, FramePack compression, and State-space / SSM rows are TODO and not yet released as `epoch-0` weights.

## Download

```bash
pip install -U "huggingface_hub[cli]"

# one row (keeps HF folder layout under ./ckpts/)
huggingface-cli download Echo-Team/Echo-Memory context_k1/epoch-0.safetensors --local-dir ./ckpts

# all currently released rows
huggingface-cli download Echo-Team/Echo-Memory --local-dir ./ckpts
```

Keep the subdirectory name in the local path (e.g. `./ckpts/spatial_mem/epoch-0.safetensors`). Eval scripts use `env/memory_baseline_runtime.py` to infer memory flags from path substrings such as `spatial_mem`; SSM checkpoint rows remain TODO.

## Use with Echo-Memory

Set the Wan backbone, static in-domain data pool, and checkpoint path:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CKPT=./ckpts/spatial_mem/epoch-0.safetensors
```

**In-domain replay + revisit (paper bundle):**

```bash
bash eval/v2/run_static_consistency_loop_and_revisit.sh
bash eval/v2/run_basic_replay_gt.sh
```

**Open-domain revisit** (first frames already in `assets/opendomain_revisit/`):

```bash
PHASE=stage1 OOD_DIR=assets/opendomain_revisit \
  bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```

**Visual comparison** (fixed prompt + first frame):

```bash
python eval/metrics/run_visual_eval.py \
  --ckpt "$CKPT" \
  --output_root ./evals_visual
```

See [eval/v2/README.md](../eval/v2/README.md) and [eval/metrics/README.md](../eval/metrics/README.md) for full options.
