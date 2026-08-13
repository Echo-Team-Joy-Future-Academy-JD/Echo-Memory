# Checkpoints (Hugging Face)

**Repo:** [Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory)

Fine-tuned DiT weights on top of [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B). Released rows are saved as `{row_id}/epoch-0.safetensors` after **1 epoch / 30,000 steps** on the static in-domain pool (640×352, 81-frame chunks). Mechanism names follow [memory_mechanisms.md](memory_mechanisms.md).

## Checkpoint index

| Family | Paper row | HF path | Steps | Echo-Memory recipe |
| --- | --- | --- | ---: | --- |
| Raw context | Context K=1 | [`context_k1/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/context_k1) | 30,000 | `train/context_learning/run_pre_qkv_ctx1.sh` |
| Raw context | Context K=20 | TODO | TODO | `train/context_learning/run_pre_qkv_ctx20.sh` |
| Spatial | Spatial Memory | TODO | TODO | `train/memory_baselines_basic/run_spatial_memory_baseline.sh` |
| Compression | FramePack length r8 | [`framepack_len_r8/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/framepack_len_r8) | 30,000 | `train/memory_baselines_basic/run_ablation_framepack_len_r8_two_chunk.sh` |
| State-space | Block-wise SSM causal v2 | [`block_wise_ssm_causal_v2/epoch-0.safetensors`](https://huggingface.co/Echo-Team/Echo-Memory/tree/main/block_wise_ssm_causal_v2) | 30,000 | `train/memory_baselines_basic/run_ablation_block_wise_ssm_causal_v2_two_chunk.sh` |
| State-space | Legacy Hybrid (VideoSSM) | TODO | TODO | `train/memory_baselines_basic/run_videossm_hybrid_baseline.sh` |
| Spatial | concat text (ablation) | TODO | TODO | `train/memory_baselines_basic/run_ablation_spatial_concat_text_two_chunk.sh` |
| Spatial | inject none (ablation) | TODO | TODO | `train/memory_baselines_basic/run_ablation_spatial_inject_none_two_chunk.sh` |
| Spatial | cross-attn t32 (ablation) | TODO | TODO | `train/memory_baselines_basic/run_ablation_spatial_cross_attn_readout_two_chunk.sh` |
| State-space | SSM ctx1 / every4 / hint21 | TODO | TODO | SSM ablation |
| State-space | SSM ctx5 / every1 / hint21 | TODO | TODO | SSM ablation |
| State-space | SSM ctx5 / every4 / hint81 | TODO | TODO | SSM ablation |

Context K=5, Context K=20, Spatial memory, legacy SSM, MoC, and geometry
rows remain under validation. The two corrected rows above are released.

## Diffusers overlay

`context_k1` remapped to official Diffusers Wan 2.1 1.3B transformer names (825 / 825 matching keys): [`Wayne-King/echo-memory-diffusers`](https://huggingface.co/Wayne-King/echo-memory-diffusers) `context_k1-diffusers/diffusion_pytorch_model.safetensors`. Community pipeline: [huggingface/diffusers#14471](https://github.com/huggingface/diffusers/pull/14471). A copy of the converted row is also proposed on the paper repo ([discussion #4](https://huggingface.co/Echo-Team/Echo-Memory/discussions/4)).

## Validated artifact hashes

| HF path | SHA256 |
| --- | --- |
| `block_wise_ssm_causal_v2/epoch-0.safetensors` | `0dd90ea3f3423644f4d68c6d1185d7d717d328f5922ee21f745fc85abe9a01a9` |
| `framepack_len_r8/epoch-0.safetensors` | `dd57625506a2c68c402dc05de8f3c6fc5f5376fcac77d7fb6f26eb6ace1d74bf` |

## Download

```bash
pip install -U "huggingface_hub[cli]"

# one row (keeps HF folder layout under ./ckpts/)
huggingface-cli download Echo-Team/Echo-Memory context_k1/epoch-0.safetensors --local-dir ./ckpts

# all currently released rows
huggingface-cli download Echo-Team/Echo-Memory --local-dir ./ckpts
```

Keep the subdirectory name in the local path. The runtime registry detects
`block_wise_ssm_causal_v2` and `framepack_len_r8` and restores context layout,
compression strategy, and generated-history continuation automatically.

## Use with Echo-Memory

Set the Wan backbone, static in-domain data pool, and checkpoint path:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CKPT=./ckpts/context_k1/epoch-0.safetensors
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
