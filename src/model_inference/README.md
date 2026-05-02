# Inference

This folder keeps the public stage-2 inference entrypoints:

- `stage2-inference-context-memory.py`
- `stage2-inference-1.3B.py`
- `stage2-inference-non-generalization.py`
- `Wan2.1-T2V-1.3B.py`
- `Wan2.1-Fun-1.3B-InP.py`
- `Wan2.1-Fun-V1.1-1.3B-InP.py`

Set the common paths before running:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

Use `python src/model_inference/<script>.py --help` to inspect script-specific options.
