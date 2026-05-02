#!/bin/bash
# Source after common_env.sh: shared train-side video sampling = 2-chunk monitor (eval-aligned).
# Replaces --sampling_atomic_left_right. Mutually exclusive with four_prompts / two_prompts in ModelLogger.
# shellcheck disable=SC2034
left45_action_path="$(dirname "${sampling_action_path}")/action_rotation_left_45.json"
right45_action_path="$(dirname "${sampling_action_path}")/action_rotation_right_45.json"
SAMPLING_TWO_CHUNK_FLAGS=(
  --enable_video_sampling
  --sampling_two_chunk_memory
  --sampling_interval_steps
  "1000"
  --sampling_two_chunk_action_path
  "${left45_action_path}"
  --sampling_num_inference_steps
  "50"
  --sampling_negative_prompt
  "oversaturated colors, overexposed, static, blurry details"
  --sampling_height
  "352"
  --sampling_width
  "640"
  --sampling_num_frames
  "81"
  --samples_per_epoch
  "0"
  --sampling_action_path
  "${right45_action_path}"
  --use_anchor_frame
)
