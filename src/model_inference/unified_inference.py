#!/usr/bin/env python3
"""
Unified single-chunk inference for all Echo-Memory memory families.

Supports: no_memory, context_k1/k5/k20, framepack_weight, framepack_len_r2/r4,
framepack_hybrid_r2/r4, spatial_mem, spatial_concat_text, spatial_inject_none,
spatial_cross_attn_readout, videossm_hybrid, block_wise_ssm.

Memory type can be specified explicitly via --memory_type or auto-detected
from the checkpoint path (--memory_type auto).

Examples:

    # Auto-detect memory type from checkpoint path
    python src/model_inference/unified_inference.py \
        --ckpt ./ckpts/spatial_mem/epoch-0.safetensors \
        --prompt "A toy bear on a table, the camera rotates around it" \
        --output_path output.mp4

    # Explicit memory type
    python src/model_inference/unified_inference.py \
        --ckpt ./ckpts/my_checkpoint.safetensors \
        --memory_type spatial_mem \
        --prompt "A scene" \
        --output_path output.mp4

    # With context image (first frame conditioning)
    python src/model_inference/unified_inference.py \
        --ckpt ./ckpts/spatial_mem/epoch-0.safetensors \
        --context_image assets/opendomain_revisit/1774363417.png \
        --action_path env/action_rotation_left_45.json \
        --prompt "A toy bear on a table" \
        --output_path output.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is in sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# memory_baseline_runtime has no heavy deps — safe to import at module level
from env.memory_baseline_runtime import (
    MemoryProfile,
    MEMORY_PROFILE_REGISTRY,
    infer_memory_profile_spec,
    apply_memory_baseline_pipe,
)


# ---------------------------------------------------------------------------
# Memory type → profile mapping
# ---------------------------------------------------------------------------

# Friendly name → registry profile_id
_REGISTRY_ALIAS = {
    "no_memory":                "no_memory_extra_two_chunk",
    "framepack_weight":         "framepack_weight_only",
    "framepack_len_r2":         "framepack_lencompress_r2",
    "framepack_len_r4":         "framepack_lencompress_r4",
    "framepack_hybrid_r2":      "framepack_hybrid_r2_weight_two_chunk",
    "framepack_hybrid_r4":      "framepack_hybrid_r4_weight_two_chunk",
    "spatial_mem":              "spatial_mem",
    "spatial_concat_text":      "spatial_concat_text_two_chunk",
    "spatial_inject_none":      "spatial_inject_none_two_chunk",
    "spatial_cross_attn_readout": "spatial_cross_attn_readout_two_chunk",
    "videossm_hybrid":          "videossm_hybrid_legacy",
    "block_wise_ssm":           "block_wise_ssm_two_chunk",
}

# context_k* are not in the registry; they use default pipe flags
# with only context_override differing
_CONTEXT_K_PROFILES = {
    "context_k1":  MemoryProfile(context_override=1),
    "context_k5":  MemoryProfile(context_override=5),
    "context_k20": MemoryProfile(context_override=20),
}

# Build profile_id → MemoryProfile lookup from the registry
_REGISTRY_PROFILES = {spec.profile_id: spec.profile for spec in MEMORY_PROFILE_REGISTRY}

ALL_MEMORY_TYPES = ["auto"] + sorted(
    set(_REGISTRY_ALIAS.keys()) | set(_CONTEXT_K_PROFILES.keys())
)


def resolve_memory_profile(memory_type: str, ckpt_path: str) -> MemoryProfile:
    """Resolve --memory_type to a MemoryProfile."""
    if memory_type == "auto":
        spec = infer_memory_profile_spec(ckpt_path)
        if spec is None:
            print(
                f"[unified_inference] WARNING: --memory_type=auto but checkpoint path "
                f"does not match any known memory profile. Running with no memory flags.\n"
                f"  ckpt: {ckpt_path}\n"
                f"  Hint: use --memory_type to specify explicitly.",
                file=sys.stderr,
                flush=True,
            )
            return MemoryProfile()
        print(f"[unified_inference] Auto-detected memory profile: {spec.profile_id}")
        return spec.profile

    if memory_type in _CONTEXT_K_PROFILES:
        print(f"[unified_inference] Using context learning profile: {memory_type}")
        return _CONTEXT_K_PROFILES[memory_type]

    if memory_type in _REGISTRY_ALIAS:
        profile_id = _REGISTRY_ALIAS[memory_type]
        profile = _REGISTRY_PROFILES[profile_id]
        print(f"[unified_inference] Using memory profile: {memory_type} ({profile_id})")
        return profile

    print(
        f"[unified_inference] ERROR: unknown --memory_type '{memory_type}'. "
        f"Available: {', '.join(ALL_MEMORY_TYPES)}",
        file=sys.stderr,
    )
    sys.exit(1)


def apply_profile_to_pipe(pipe, profile: MemoryProfile) -> None:
    """Apply a MemoryProfile directly to the pipeline object."""
    pipe.use_framepack_memory = bool(profile.use_framepack_memory)
    pipe.context_temporal_decay = float(profile.context_temporal_decay or 1.0)
    pipe.context_attention_weight = float(profile.context_attention_weight or 1.0)
    pipe.use_framepack_length_compress = bool(profile.use_framepack_length_compress)
    pipe.framepack_ratio = int(profile.framepack_ratio or 2)
    pipe.use_spatial_memory = bool(profile.use_spatial_memory)
    pipe.spatial_memory_tokens = int(profile.spatial_memory_tokens or 64)
    if profile.spatial_memory_inject_mode:
        pipe.spatial_memory_inject_mode = str(profile.spatial_memory_inject_mode)
    pipe.use_spatial_memory_legacy = bool(profile.use_spatial_memory_legacy)
    # Warn if spatial memory requested but module not loaded from checkpoint
    if (
        pipe.use_spatial_memory
        and not pipe.use_spatial_memory_legacy
        and getattr(pipe, "spatial_memory_module", None) is None
    ):
        pipe.use_spatial_memory_legacy = True
        print(
            "[unified_inference] WARN: spatial_mem profile but spatial_memory_module "
            "not found in checkpoint. Falling back to legacy adaptive pool.",
            file=sys.stderr,
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified single-chunk inference for all Echo-Memory memory families.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Memory types:
  auto                  Auto-detect from checkpoint path
  no_memory             No memory (I2V floor baseline)
  context_k1/k5/k20    Raw context with 1/5/20 frames
  framepack_weight      FramePack temporal decay reweighting
  framepack_len_r2/r4   FramePack length compression ratio 2/4
  framepack_hybrid_r2/r4  FramePack hybrid (length + weight)
  spatial_mem           Spatial grid memory (64 tokens)
  spatial_concat_text   Spatial memory via text KV concatenation
  spatial_inject_none   Spatial memory with withheld read-out
  spatial_cross_attn_readout  Spatial memory via cross-attention
  videossm_hybrid       Legacy hybrid SSM
  block_wise_ssm        Block-wise recurrent SSM
""",
    )

    # Required
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to fine-tuned .safetensors checkpoint")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt describing the scene")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output video path (.mp4)")

    # Memory selection
    parser.add_argument("--memory_type", type=str, default="auto",
                        choices=ALL_MEMORY_TYPES,
                        help="Memory type (default: auto-detect from checkpoint path)")

    # Model paths
    parser.add_argument("--base_model", type=str,
                        default=os.environ.get("WAN_BASE_MODEL", ""),
                        help="Wan2.1 base model directory (default: $WAN_BASE_MODEL)")

    # Context image
    parser.add_argument("--context_image", type=str, default=None,
                        help="Path to first-frame context image (enables context memory)")

    # Action control
    parser.add_argument("--action_path", type=str, default=None,
                        help="Path to action JSON file (81-frame camera trajectory)")

    # Generation parameters
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames per chunk (default: 81)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=15.0)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt (default: standard quality filter)")

    # Output
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS")

    return parser


def main():
    args = build_parser().parse_args()

    # ── Validate paths ──────────────────────────────────────────────────
    # Heavy imports deferred so --help works without GPU/conda environment
    import torch
    from PIL import Image
    from env.loop_utils import load_pipeline_and_ckpt, DEFAULT_NEGATIVE_PROMPT
    from env.run_replay_loop_two_chunk import run_one_chunk, encode_context_frames_per_frame
    from diffsynth import save_video

    neg_prompt = args.negative_prompt if args.negative_prompt else DEFAULT_NEGATIVE_PROMPT

    if not args.base_model:
        print("ERROR: --base_model or $WAN_BASE_MODEL must be set.", file=sys.stderr)
        sys.exit(1)

    dit_path = os.path.join(args.base_model, "diffusion_pytorch_model.safetensors")
    text_encoder_path = os.path.join(args.base_model, "models_t5_umt5-xxl-enc-bf16.pth")
    vae_path = os.path.join(args.base_model, "Wan2.1_VAE.pth")

    for p in [dit_path, text_encoder_path, vae_path]:
        if not os.path.isfile(p):
            print(f"ERROR: base model file not found: {p}", file=sys.stderr)
            sys.exit(1)

    if not os.path.isfile(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    # ── Resolve memory profile ──────────────────────────────────────────
    profile = resolve_memory_profile(args.memory_type, args.ckpt)

    # ── Load pipeline + checkpoint ──────────────────────────────────────
    print(f"[unified_inference] Loading pipeline from {args.base_model}")
    print(f"[unified_inference] Loading checkpoint from {args.ckpt}")
    pipe = load_pipeline_and_ckpt(
        ckpt_path=args.ckpt,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        vae_path=vae_path,
        device="cuda",
        add_action_attn=False,
        action_use_temporal_attention=True,
    )

    # ── Apply memory flags ──────────────────────────────────────────────
    if args.memory_type == "auto":
        apply_memory_baseline_pipe(pipe, args.ckpt)
    else:
        apply_profile_to_pipe(pipe, profile)

    # ── Encode context image (if provided) ──────────────────────────────
    context_latents = None
    context_actions_t = None
    num_context_frames = 0

    if args.context_image:
        if not os.path.isfile(args.context_image):
            print(f"ERROR: context image not found: {args.context_image}", file=sys.stderr)
            sys.exit(1)

        print(f"[unified_inference] Encoding context image: {args.context_image}")
        ctx_pil = Image.open(args.context_image).convert("RGB").resize(
            (args.width, args.height), Image.LANCZOS
        )
        pipe.load_models_to_device(["vae"])
        with torch.no_grad():
            context_latents = encode_context_frames_per_frame(
                pipe, [ctx_pil], pipe.device
            )
        num_context_frames = 1
        # Identity RT for context frame (no relative pose change)
        identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        context_actions_t = torch.tensor([identity_rt], dtype=torch.float32)

    # ── Generate ────────────────────────────────────────────────────────
    print(f"[unified_inference] Generating {args.num_frames} frames @ {args.width}x{args.height}")
    frames = run_one_chunk(
        pipe=pipe,
        prompt=args.prompt,
        use_negative_prompt=neg_prompt,
        action_path=args.action_path,
        context_latents=context_latents,
        num_context_frames=num_context_frames,
        context_actions_t=context_actions_t,
        chunk_frames=args.num_frames,
        h=args.height,
        w=args.width,
        seed=args.seed,
        sigma_shift=args.sigma_shift,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        log_prefix="[unified_inference]",
    )

    # ── Save video ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    save_video(frames, args.output_path, fps=args.fps, quality=5)
    print(f"[unified_inference] Video saved to {args.output_path}")


if __name__ == "__main__":
    main()
