---
license: cc-by-4.0
language:
- en
tags:
- video-generation
- world-model
- memory
- action-conditioned
- wan
library_name: diffsynth
---

# Echo-Memory — Wan 2.1 1.3B memory baseline checkpoints

Paper-aligned **epoch-0** fine-tunes for the Echo-Memory controlled memory study ([GitHub](https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Memory) · [project page](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/)).

**Backbone:** [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)  
**Training:** static in-domain pool, 1 epoch, **30,000 steps**, 640×352, 81-frame chunks  
**File layout:** `{row_id}/epoch-0.safetensors` — see `checkpoints.json` for metadata.

## Download

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Echo-Team/Echo-Memory context_k1/epoch-0.safetensors --local-dir ./ckpts
```

## Usage with Echo-Memory eval

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
export CKPT=./ckpts/context_k1/epoch-0.safetensors
bash eval/v2/run_static_consistency_loop_and_revisit.sh
```

Runtime memory flags are inferred from the checkpoint path via `env/memory_baseline_runtime.py`.

## Citation

Echo-Memory: A Controlled Study of Memory in Action World Models — Echo Team @ Joy Future Academy, JD ([ResearchGate DOI](https://doi.org/10.13140/RG.2.2.19906.34248)).
