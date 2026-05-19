# Revisit Suite One-Click Eval

Purpose:

1. Discover `outputs/**/epoch-0.safetensors`.
2. Run basic static-memory revisit on training-set samples, sharded across GPUs.
3. Run OOD static-memory revisit from `assets/opendomain_revisit`, sharded across GPUs.
4. Save stage-1 evidence images: first frame and final revisit tail frames.
5. Save traditional first-vs-final metrics: MSE, PSNR, SSIM.
6. Optionally run Qwen/VLM scoring with high weight on the toy bear appearance and lower weight on background scene.

## Run

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```

Useful overrides:

```bash
NUM_GPUS=8 \
TRAIN_LIMIT=8 \
OOD_LIMIT=8 \
MODES=rot180_4chunk,rot360_8chunk,right45_return_2chunk \
VLM_API_BASE=http://127.0.0.1:8000/v1 \
VLM_MODEL=Qwen/Qwen2.5-VL-72B-Instruct \
bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```

If the VLM server is not up yet, run only stage 1:

```bash
PHASE=stage1 bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```

Then run VLM later:

```bash
PHASE=vlm EVAL_ROOT=/path/to/eval_root bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```

For a dry run of the VLM phase:

```bash
VLM_DRY_RUN=1 bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```

## Revisit Modes

- `rot180_4chunk`: four chunks, each 45 degrees, `+45,+45,-45,-45`.
- `rot360_8chunk`: eight chunks, each 45 degrees, full 360-degree loop.
- `right45_return_2chunk`: two chunks, right 45 degrees then return 45 degrees; focuses on second-chunk memory.
- `random_closed`: random yaw/translation path with approximate closed-loop return.

The success criterion is visual scene consistency between the first frame and the final revisit tail. For OOD samples with the toy bear, the VLM prompt weights bear appearance and presence most strongly.

## Outputs

Default root:

```text
eval_outputs/revisit_suite_<timestamp>/
```

Each case contains:

- `revisit_gen_only.mp4`
- `stage1_frames/first_00.png`
- `stage1_frames/revisit_tail_*.png`
- `stage1_metrics.json`
- `vlm_score.json` after VLM scoring

## Inspecting Videos and Images

The suite is designed to keep both metric files and visual evidence:

- `revisit_gen_only.mp4`: generated return trajectory for the case.
- `stage1_frames/first_00.png`: input first frame.
- `stage1_frames/revisit_tail_*.png`: final return frames shown to the VLM judge.
- `stage1_frames/first_last_chunk_changes/*.png`: side-by-side first chunk, last chunk, and absolute-difference panels when enabled.

To browse a run from a remote machine:

```bash
python -m http.server 8000 --directory eval_outputs
```

To collect material for paper figures:

```bash
python eval/v2/revisit_suite/export_revisit_materials.py \
  --eval-root eval_outputs/revisit_suite_<timestamp> \
  --out-dir paper_case_materials \
  --prefix echo_memory_revisit
```

Good qualitative cases usually combine `first_00.png`, the last two or four `revisit_tail_*.png` frames, and the corresponding `revisit_gen_only.mp4`. Use `right45_return_2chunk` for the open-domain paper probe and `rot180_4chunk` for in-domain loop closure examples.
