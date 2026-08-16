# ComfyUI nodes for Echo-Memory

Custom nodes that wrap the paper inference path: **Wan 2.1 1.3B** + an Echo-Memory row, first frame, text prompt, and an 81-frame camera-action JSON.

This is not a port of [Kijai's WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper). It calls `inference/unified_inference.py` so Context / Compression / Spatial / State-Space rows stay aligned with the released scripts.

## Install

**Option A — clone the repo into ComfyUI** (Manager “Install via Git URL” also works):

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Memory.git
```

The repo-root `__init__.py` registers the nodes when ComfyUI imports the folder.

**Option B — symlink only the node pack** (if Echo-Memory already lives elsewhere):

```bash
ln -s /path/to/Echo-Memory/comfyui /path/to/ComfyUI/custom_nodes/ComfyUI-EchoMemory
```

Restart ComfyUI. You need the same Python env as Echo-Memory (DiffSynth / Wan deps) **or** ComfyUI's env with those packages installed. For option B, set `PYTHONPATH` to the Echo-Memory root if the nodes cannot import `inference` / `env`.

ComfyUI-Manager search listing requires publishing to the [Comfy Registry](https://registry.comfy.org/) (`comfy node publish`). The metadata is in the repo-root `pyproject.toml`; create a publisher whose id is `echo-team` (or change `PublisherId` to match yours) before publishing.

## Graph

1. **Echo-Memory Loader** — `Wan2.1-T2V-1.3B` directory + `{row}/epoch-0.safetensors` + memory type  
2. **Echo-Memory Camera Action** — bundled `left_45` / `right_45`, or a custom 81-frame JSON  
3. **Load Image** — first frame  
4. **Echo-Memory Generate** — prompt, chunks, steps → `IMAGE` batch (one tensor per frame)

Connect the image batch to Preview Image, or to Video Helper Suite **Video Combine** (`frame_rate=15`) to write an mp4.

Released rows: `context_k1`, `framepack_len_r8`, `block_wise_ssm_causal_v2` on [Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory). Multi-chunk SSM / FramePack rows need a first frame; keep `num_frames=81` unless you know the row was trained otherwise.

## No local GPU

Use the browser demo instead: [hugging-apps/echo-memory](https://huggingface.co/spaces/hugging-apps/echo-memory).
