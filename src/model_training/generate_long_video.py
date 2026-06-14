#!/usr/bin/env python3
"""CLI compatibility entrypoint for long-video generation."""

from src.model_training.multichunk_sample_utils import (
    encode_frames_to_latents,
    generate_long_video,
    load_model,
    main,
    sample_prompts_from_dataset,
)

__all__ = [
    "encode_frames_to_latents",
    "generate_long_video",
    "load_model",
    "main",
    "sample_prompts_from_dataset",
]

if __name__ == "__main__":
    main()
