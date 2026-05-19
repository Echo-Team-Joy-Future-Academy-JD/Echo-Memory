# Visual Evaluation Design

This evaluation is for human inspection rather than a single scalar score. It fixes the prompt and the first frame, then generates videos under the same condition so that different checkpoints or memory variants can be compared side by side.

## 1. Design

- **Prompt groups**: prompts are grouped by what they stress, such as identity preservation, long-horizon consistency, loop revisit, object state, or generic scene stability.
- **First-frame presets**: cases can use a fixed image or a frame extracted from the dataset via `(video_name, start_frame)`.
- **Output layout**: outputs are grouped by `prompt_id` and `first_chunk_id`, for example `evals_visual/prompt_identity_single_person_first_fixed_face/`. Each folder contains short MP4 files for inspection.

## 2. Configuration

Use `visual_eval_config.yaml`:

- `prompts`: each item has `id`, `text`, `category`, and an optional `note`.
- `first_chunk_presets`: each preset is either a `fixed_image` path or a `dataset_frame` with `video_name` and `start_frame`.
- `recommended_pairs`: optional `[prompt_id, first_chunk_id]` pairs for a smaller curated run.

## 3. Recommended Workflow

1. Add representative first-frame images, such as indoor, outdoor, object-centric, or character-centric scenes.
2. Add dataset-frame presets if you want repeatable in-domain examples.
3. Run `run_visual_eval.py` with `--ckpt` and `--output_root`.
4. Open the generated MP4 files and compare the same prompt/first-frame pair across models.

## 4. Run Examples

```bash
# Run all configured prompt x first-frame pairs.
python eval/metrics/run_visual_eval.py \
  --ckpt /path/to/epoch-0.safetensors \
  --output_root /path/to/ckpt_dir/evals_visual \
  --config eval/metrics/visual_eval_config.yaml

# Run selected prompts and first frames.
python eval/metrics/run_visual_eval.py \
  --ckpt /path/to/epoch-0.safetensors \
  --output_root /path/to/ckpt_dir/evals_visual \
  --prompts identity_single_person scene_indoor_room \
  --first_chunks fixed_default fixed_face

# Use dataset frames as first frames.
python eval/metrics/run_visual_eval.py \
  --ckpt /path/to/epoch-0.safetensors \
  --dataset_base /path/to/Context-as-Memory-Dataset \
  --output_root /path/to/ckpt_dir/evals_visual
```

## 5. Relationship to Paper Cases

- **In-domain loop cases** come from dataset-backed first frames and prompts. Use them to inspect whether a model returns to a known scene.
- **Open-domain revisit cases** use `assets/opendomain_revisit` and are best generated with `eval/v2/revisit_suite`.
- **Paper qualitative panels** should usually combine `first_00.png`, several `revisit_tail_*.png` frames, and `revisit_gen_only.mp4` from the same case directory.

## 6. What to Inspect

- Whether the same object is still present after the camera returns.
- Whether object color, shape, and identity remain stable.
- Whether the final view is actually a revisit rather than a plausible but different scene.
- Whether background consistency is preserved without overpowering the object-identity judgment.
