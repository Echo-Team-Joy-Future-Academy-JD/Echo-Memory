#!/usr/bin/env python3
"""
Static consistency: multi-action chunks then revisit (fixed first frame).

This is the \"MultiActionRevisit\" task:
- chunk0: rotate_left_45 (or provided)
- chunk1: translate_forward
- chunk2: rotate_right_45
- chunk3: translate_backward

All actions are per-chunk relative to that chunk's first frame, matching training / existing eval conventions.
We generate a single concatenated mp4 and also save per-chunk gen-only mp4s for inspection.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import time
from typing import List

import numpy as np
import torch
from PIL import Image

_script_dir = os.path.dirname(os.path.abspath(__file__))
_eval_v2_dir = os.path.dirname(_script_dir)
_repo_root = os.path.dirname(os.path.dirname(_eval_v2_dir))
_env_dir = os.path.join(_repo_root, "env")

if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if _env_dir not in sys.path:
    sys.path.insert(0, _env_dir)

import loop_utils as irc
import memory_baseline_runtime as mbr
from diffsynth import save_video
from run_replay_loop_two_chunk import (
    encode_context_frames_per_frame,
    context_frames_for_next_chunk,
    replay_context_from_generated_frames,
    run_one_chunk,
    _frame_to_pil,
    load_sample_first_frame,
)


def _mse_rgb(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d ** 2))


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return 100.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _resize_to_sampling_size(pil_img, width, height):
    if pil_img.size == (width, height):
        return pil_img
    try:
        return pil_img.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    except AttributeError:
        return pil_img.convert("RGB").resize((width, height), Image.LANCZOS)


def main():
    p = argparse.ArgumentParser(description="Static consistency: composite action revisit (fixed first frame)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--first_frame_image", type=str, default=None, help="Open-domain first frame (optional if --dataset_base+video+start)")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Wan2.1 base model dir; default: $WAN_BASE_MODEL",
    )
    p.add_argument("--prompt", type=str, default="A scene.", help="Used only with --first_frame_image; dataset mode uses CSV prompt")
    p.add_argument("--dataset_base", type=str, default=None, help="In-domain: training set root (frames/, jsons/, metadata)")
    p.add_argument("--video_name", type=str, default=None)
    p.add_argument("--start_frame", type=int, default=None)
    p.add_argument("--action_combo_dir", required=True, help="Directory containing chunk0..chunk3 action jsons")
    p.add_argument("--chunk_frames", type=int, default=81)
    p.add_argument("--context_frames", type=int, default=1)
    # Memory baseline runtime flags (must align with ckpt training for multichunk consistency)
    p.add_argument("--use_framepack_memory", action="store_true", help="FramePack/FAR-style context reweighting")
    p.add_argument("--context_temporal_decay", type=float, default=1.0, help="FramePack/FAR per-frame decay")
    p.add_argument("--context_attention_weight", type=float, default=1.0, help="FramePack/FAR global scale for context tokens")
    p.add_argument("--use_framepack_length_compress", action="store_true", help="FramePack length compress context tokens K->K'")
    p.add_argument("--framepack_ratio", type=int, default=2, help="FramePack length compress ratio r")
    p.add_argument("--use_spatial_memory", action="store_true", help="Enable spatial memory baseline")
    p.add_argument("--use_spatial_memory_legacy", action="store_true", help="Legacy adaptive pool (no SpatialGridMemory in ckpt)")
    p.add_argument("--spatial_memory_tokens", type=int, default=64, help="Spatial memory token count")
    p.add_argument(
        "--spatial_memory_inject_mode",
        type=str,
        default=None,
        choices=("concat_text", "cross_attn_readout", "none"),
        help="Spatial memory inject mode; must match training",
    )
    p.add_argument("--height", type=int, default=352)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--camera_inject_mode", type=str, default=None)
    p.add_argument("--no_camera_encoder_separate_t_r", action="store_true")
    p.add_argument("--no_omit_context_actions", action="store_true")
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"CKPT not found: {args.ckpt}")
    action_paths_pre = [
        os.path.join(args.action_combo_dir, "chunk0_rotate_left_45.json"),
        os.path.join(args.action_combo_dir, "chunk1_translate_forward.json"),
        os.path.join(args.action_combo_dir, "chunk2_rotate_right_45.json"),
        os.path.join(args.action_combo_dir, "chunk3_translate_backward.json"),
    ]
    for apth in action_paths_pre:
        if not os.path.isfile(apth):
            raise FileNotFoundError(f"Missing action json (fail-fast before load_pipeline): {apth}")

    base_model = args.base_model or os.environ.get("WAN_BASE_MODEL")
    if not base_model:
        raise ValueError("Set --base_model or WAN_BASE_MODEL to the Wan2.1 base model directory.")
    for _name in (
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    ):
        _p = os.path.join(base_model, _name)
        if not os.path.isfile(_p):
            raise FileNotFoundError(f"Missing Wan2.1 base weight (fail-fast): {_p}")

    os.makedirs(args.output_dir, exist_ok=True)
    w, h = args.width, args.height

    in_domain = (
        args.dataset_base
        and args.video_name is not None
        and args.start_frame is not None
    )
    if in_domain:
        first_frame_pil = load_sample_first_frame(args.dataset_base, args.video_name, int(args.start_frame), w, h)
        if first_frame_pil is None:
            raise FileNotFoundError(
                f"Cannot load first frame for in-domain sample {(args.video_name, args.start_frame)} under {args.dataset_base}"
            )
        prompt = irc.load_prompt_for_video(args.dataset_base, args.video_name) or "A scene."
    else:
        if not args.first_frame_image or not os.path.isfile(args.first_frame_image):
            raise ValueError("Provide --dataset_base --video_name --start_frame OR a valid --first_frame_image")
        first_frame_pil = Image.open(args.first_frame_image).convert("RGB")
        first_frame_pil = _resize_to_sampling_size(first_frame_pil, w, h)
        prompt = args.prompt

    camera_inject_mode = (args.camera_inject_mode or "").strip() or None
    if not camera_inject_mode:
        env_cam = (os.environ.get("CAMERA_INJECT_MODE") or "").strip().lower()
        if env_cam in ("pre_qkv_post", "pre_qkv", "pre_norm", "post"):
            camera_inject_mode = env_cam
    if not camera_inject_mode:
        for mode in ("pre_qkv_post", "pre_qkv", "pre_norm", "post"):
            if mode.replace("_", "") in (args.ckpt or "").lower():
                camera_inject_mode = mode
                break
    if not camera_inject_mode:
        camera_inject_mode = "pre_qkv"

    load_kw = dict(
        action_inject_after_spatial_attn=True,
        add_action_attn=True,
        action_use_temporal_attention=True,
        camera_inject_mode=camera_inject_mode,
    )
    if args.no_camera_encoder_separate_t_r:
        load_kw["camera_encoder_separate_t_r"] = False

    pipe = irc.load_pipeline_and_ckpt(
        args.ckpt,
        f"{base_model}/diffusion_pytorch_model.safetensors",
        f"{base_model}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{base_model}/Wan2.1_VAE.pth",
        **load_kw,
    )

    # Runtime memory flags: CLI wins when any --use_* is set; else infer from ckpt path (memory_baselines_basic_*).
    cli_mem = bool(
        getattr(args, "use_framepack_memory", False)
        or getattr(args, "use_framepack_length_compress", False)
        or getattr(args, "use_spatial_memory", False)
    )
    if cli_mem:
        pipe.use_framepack_memory = bool(getattr(args, "use_framepack_memory", False))
        pipe.context_temporal_decay = float(getattr(args, "context_temporal_decay", 1.0) or 1.0)
        pipe.context_attention_weight = float(getattr(args, "context_attention_weight", 1.0) or 1.0)
        pipe.use_framepack_length_compress = bool(getattr(args, "use_framepack_length_compress", False))
        pipe.framepack_ratio = int(getattr(args, "framepack_ratio", 2) or 2)
        pipe.use_spatial_memory = bool(getattr(args, "use_spatial_memory", False))
        pipe.spatial_memory_tokens = int(getattr(args, "spatial_memory_tokens", 64) or 64)
        if getattr(args, "spatial_memory_inject_mode", None):
            pipe.spatial_memory_inject_mode = str(getattr(args, "spatial_memory_inject_mode"))
        pipe.use_spatial_memory_legacy = bool(getattr(args, "use_spatial_memory_legacy", False))
        if pipe.use_spatial_memory and not pipe.use_spatial_memory_legacy and getattr(pipe, "spatial_memory_module", None) is None:
            pipe.use_spatial_memory_legacy = True
    else:
        mbr.apply_memory_baseline_pipe(pipe, args.ckpt)
    if getattr(pipe, "use_spatial_memory", False) and not getattr(pipe, "use_spatial_memory_legacy", False) and getattr(pipe, "spatial_memory_module", None) is None:
        pipe.use_spatial_memory_legacy = True

    use_neg = getattr(irc, "DEFAULT_NEGATIVE_PROMPT", "oversaturated colors, overexposed, static, blurry details")
    omit = not args.no_omit_context_actions

    action_paths = action_paths_pre

    # chunk0 context = first frame
    identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        ctx_latents = encode_context_frames_per_frame(pipe, [first_frame_pil], pipe.device)
    ctx_actions_t = torch.tensor([identity_rt], dtype=torch.float32)

    chunks: List[List] = []
    times = []
    for ch, action_path in enumerate(action_paths):
        t0 = time.time()
        frames = run_one_chunk(
            pipe,
            prompt,
            use_neg,
            action_path,
            context_latents=ctx_latents,
            num_context_frames=ctx_latents.shape[2],
            context_actions_t=ctx_actions_t,
            chunk_frames=args.chunk_frames,
            h=h,
            w=w,
            seed=args.seed + ch,
            sigma_shift=args.sigma_shift,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            inference_noise_level=0.0,
            omit_context_actions=omit,
            log_prefix="[combo_revisit]",
        )
        t1 = time.time()
        times.append({"chunk": ch, "seconds": t1 - t0, "action": os.path.basename(action_path)})
        chunks.append(frames)

        # prepare context for next chunk (except last)
        if ch < len(action_paths) - 1:
            n_ctx = min(args.context_frames, len(frames))
            prev_frames = replay_context_from_generated_frames(frames, n_ctx)
            prev_pil = [_frame_to_pil(f, w, h) for f in prev_frames]
            pipe.load_models_to_device(["vae"])
            with torch.no_grad():
                ctx_latents = encode_context_frames_per_frame(pipe, prev_pil, pipe.device)
            num_ctx_tokens = ctx_latents.shape[2]
            ctx_actions_t = torch.tensor([identity_rt] * num_ctx_tokens, dtype=torch.float32)

    # save outputs
    all_frames = []
    for ch, frames in enumerate(chunks):
        save_video(frames, os.path.join(args.output_dir, f"combo_chunk{ch}_gen_only.mp4"), fps=15, quality=5)
        all_frames.extend(frames)
    save_video(all_frames, os.path.join(args.output_dir, "combo_revisit_4chunk_gen_only.mp4"), fps=15, quality=5)
    with open(os.path.join(args.output_dir, "combo_revisit_speed.json"), "w", encoding="utf-8") as f:
        json.dump({"chunks": times}, f, indent=2)

    first_np = np.array(first_frame_pil.convert("RGB"), dtype=np.uint8)
    last_pil = _frame_to_pil(all_frames[-1], w, h)
    last_np = np.array(last_pil.convert("RGB"), dtype=np.uint8)
    closure_mse = _mse_rgb(first_np, last_np)
    closure = {
        "closure_first_vs_last_mse": closure_mse,
        "closure_first_vs_last_psnr": _psnr_from_mse(closure_mse),
        "in_domain": bool(in_domain),
        "video_name": args.video_name,
        "start_frame": args.start_frame,
        "num_chunks": len(action_paths),
        "chunk_frames": args.chunk_frames,
    }
    with open(os.path.join(args.output_dir, "revisit_closure_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(closure, f, indent=2)

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()

