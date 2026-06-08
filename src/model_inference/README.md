# Inference

## Unified Inference (Recommended)

`unified_inference.py` supports **all memory families** through a single `--memory_type` argument:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export PYTHONPATH=$PWD:${PYTHONPATH:-}

# Auto-detect memory type from checkpoint path
python inference/unified_inference.py \
    --ckpt ./ckpts/spatial_mem/epoch-0.safetensors \
    --prompt "A toy bear on a table, the camera rotates around it" \
    --output_path output.mp4

# Explicit memory type
python inference/unified_inference.py \
    --ckpt ./ckpts/my_checkpoint.safetensors \
    --memory_type spatial_mem \
    --prompt "A scene" \
    --output_path output.mp4

# With context image + action control
python inference/unified_inference.py \
    --ckpt ./ckpts/spatial_mem/epoch-0.safetensors \
    --context_image assets/opendomain_revisit/1774363417.png \
    --action_path env/action_rotation_left_45.json \
    --prompt "A toy bear on a table" \
    --output_path output.mp4
```

### Available Memory Types

| `--memory_type` | Family | Description |
|---|---|---|
| `auto` | (detected) | Auto-detect from checkpoint path |
| `no_memory` | Floor | No memory, I2V baseline |
| `context_k1` | Raw context | 1 context frame |
| `context_k5` | Raw context | 5 context frames |
| `context_k20` | Raw context | 20 context frames |
| `framepack_weight` | Compression | FramePack temporal decay reweighting |
| `framepack_len_r2` / `framepack_len_r4` | Compression | FramePack length compression ratio 2 / 4 |
| `framepack_hybrid_r2` / `framepack_hybrid_r4` | Compression | Hybrid: length compression + token weighting |
| `spatial_mem` | Spatial | Spatial grid memory (64 tokens) |
| `spatial_concat_text` | Spatial | Spatial memory via text KV concatenation |
| `spatial_inject_none` | Spatial | Spatial memory with withheld read-out |
| `spatial_cross_attn_readout` | Spatial | Spatial memory via cross-attention |
| `videossm_hybrid` | State-space | Legacy hybrid SSM |
| `block_wise_ssm` | State-space | Block-wise recurrent SSM |

Run `python inference/unified_inference.py --help` for full argument reference.

## Legacy Scripts

Individual entrypoints kept for backward compatibility:

- `stage2-inference-1.3B.py` — Basic inference with action control (no memory)
- `stage2-inference-context-memory.py` — Context memory stub (experimental)
- `stage2-inference-non-generalization.py` — GT action replay
- `Wan2.1-T2V-1.3B.py` / `Wan2.1-Fun-1.3B-InP.py` / `Wan2.1-Fun-V1.1-1.3B-InP.py` — Base model variants

Set the common paths before running any script:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

Use `python src/model_inference/<script>.py --help` to inspect script-specific options.
