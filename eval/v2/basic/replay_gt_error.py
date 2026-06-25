#!/usr/bin/env python3
"""
Basic capability (v2): replay GT trajectory and compute per-frame error vs GT.
For reporting cross-chunk behavior, use num_chunks >= 2 (eval scripts default 3 via NUM_CHUNKS_LONG).

Output:
- gen video (gen_only mp4)
- per-frame mse, psnr, ssim (if scikit-image), lpips (if lpips pkg unless --no_lpips)
- speed profile (seconds, fps)
- replay_gt_metrics.json includes metric_definitions and quality_notes

This script uses existing run_one_chunk + build_gt_trajectory_actions for action, and
loads GT frames from dataset_base/frames/<video_name>/<frame>.png for comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

_script_dir = os.path.dirname(os.path.abspath(__file__))
_eval_v2_dir = os.path.dirname(_script_dir)
_repo_root = os.path.dirname(os.path.dirname(_eval_v2_dir))
_env_dir = os.path.join(_repo_root, "env")
_metrics_dir = os.path.join(_eval_v2_dir, "metrics")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if _env_dir not in sys.path:
    sys.path.insert(0, _env_dir)
if _metrics_dir not in sys.path:
    sys.path.insert(0, _metrics_dir)

try:
    from skimage.metrics import structural_similarity as _skimage_ssim

    _HAS_SKIMAGE_SSIM = True
except Exception:
    _skimage_ssim = None  # type: ignore[assignment,misc]
    _HAS_SKIMAGE_SSIM = False

try:
    import psnr_lpips as _pl

    _HAS_PSNR_LPIPS_MOD = True
except Exception:
    _HAS_PSNR_LPIPS_MOD = False
    _pl = None  # type: ignore

import loop_utils as irc
import memory_baseline_runtime as mbr
from diffsynth import save_video
from run_replay_loop_two_chunk import (
    build_gt_trajectory_actions,
    encode_context_frames_per_frame,
    context_frames_for_next_chunk,
    replay_context_from_generated_frames,
    run_one_chunk,
    _frame_to_pil,
    load_sample_first_frame,
)


def _read_gt_frame(dataset_base: str, video_name: str, idx: int, w: int, h: int) -> np.ndarray | None:
    base = os.path.join(dataset_base, "frames", str(video_name))
    for fmt in (f"{idx:04d}.png", f"{idx}.png"):
        p = os.path.join(base, fmt)
        if os.path.isfile(p):
            im = Image.open(p).convert("RGB").resize((w, h))
            return np.array(im, dtype=np.uint8)
    return None


def _label(img: Image.Image, text: str) -> Image.Image:
    from PIL import ImageDraw

    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, 8 + 8 * len(text), 18], fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 0))
    return out


def _build_sidebyside(paired: List[tuple], w: int, h: int) -> List[Image.Image]:
    """Left=GT, Right=Gen. Missing GT frames render as black."""
    frames: List[Image.Image] = []
    for gen_pil, gt in paired:
        gt_img = Image.fromarray(gt) if gt is not None else Image.new("RGB", (w, h), (0, 0, 0))
        canvas = Image.new("RGB", (w * 2, h), (0, 0, 0))
        canvas.paste(_label(gt_img, "GT"), (0, 0))
        canvas.paste(_label(gen_pil, "Gen"), (w, 0))
        frames.append(canvas)
    return frames


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d ** 2))


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return 100.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _compute_ssim(gen: np.ndarray, gt: np.ndarray) -> Optional[float]:
    if not _HAS_SKIMAGE_SSIM or _skimage_ssim is None:
        return None
    try:
        try:
            return float(_skimage_ssim(gt, gen, channel_axis=2, data_range=255))
        except TypeError:
            return float(_skimage_ssim(gt, gen, multichannel=True, data_range=255))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="Replay GT trajectory and compute per-frame MSE vs GT")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset_base", required=True)
    p.add_argument("--video_name", required=True)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    p.add_argument("--chunk_frames", type=int, default=81)
    p.add_argument("--context_frames", type=int, default=1)
    p.add_argument("--height", type=int, default=352)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Wan2.1 base model dir; default: $WAN_BASE_MODEL",
    )
    p.add_argument("--prompt", type=str, default=None, help="default: dataset prompt")
    p.add_argument("--camera_inject_mode", type=str, default=None)
    p.add_argument("--no_camera_encoder_separate_t_r", action="store_true")
    p.add_argument("--no_omit_context_actions", action="store_true")
    p.add_argument("--write_csv", action="store_true")
    p.add_argument("--no_lpips", action="store_true", help="skip LPIPS (faster, no extra deps on GPU)")
    p.add_argument("--lpips_device", type=str, default="cuda", help="device for LPIPS model")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    w, h = args.width, args.height

    prompt = args.prompt or (irc.load_prompt_for_video(args.dataset_base, args.video_name) or "A scene.")
    use_neg = getattr(irc, "DEFAULT_NEGATIVE_PROMPT", "oversaturated colors, overexposed, static, blurry details")
    omit = not args.no_omit_context_actions

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
        # 与 memory_baselines_basic / run_static_consistency 默认一致（非 post）
        camera_inject_mode = "pre_qkv"

    load_kw = dict(
        action_inject_after_spatial_attn=True,
        add_action_attn=True,
        action_use_temporal_attention=True,
        camera_inject_mode=camera_inject_mode,
    )
    if args.no_camera_encoder_separate_t_r:
        load_kw["camera_encoder_separate_t_r"] = False

    base_model = args.base_model or os.environ.get("WAN_BASE_MODEL")
    if not base_model:
        raise ValueError("Set --base_model or WAN_BASE_MODEL to the Wan2.1 base model directory.")
    for _name in ("diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
        _p = os.path.join(base_model, _name)
        if not os.path.isfile(_p):
            raise FileNotFoundError(
                f"Missing Wan2.1 base weight: {_p} (set --base_model or WAN_BASE_MODEL to override)"
            )
    pipe = irc.load_pipeline_and_ckpt(
        args.ckpt,
        f"{base_model}/diffusion_pytorch_model.safetensors",
        f"{base_model}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{base_model}/Wan2.1_VAE.pth",
        **load_kw,
    )
    mbr.apply_memory_baseline_pipe(pipe, args.ckpt)

    quality_notes: List[str] = []
    if not _HAS_SKIMAGE_SSIM:
        quality_notes.append("SSIM skipped: scikit-image not available or import failed.")
    lpips_model = None
    if not args.no_lpips and _HAS_PSNR_LPIPS_MOD and _pl is not None:
        lpips_model = _pl._lpips_model(device=args.lpips_device)
        if lpips_model is None:
            quality_notes.append("LPIPS unavailable: lpips/torch import failed.")
    elif args.no_lpips:
        quality_notes.append("LPIPS disabled (--no_lpips).")

    # Initial context = sample first GT frame for this trajectory
    first_pil = load_sample_first_frame(args.dataset_base, args.video_name, args.start_frame, w, h)
    if first_pil is None:
        raise FileNotFoundError("Cannot load first GT frame for (video_name,start_frame)")

    identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        ctx_latents = encode_context_frames_per_frame(pipe, [first_pil], pipe.device)
    ctx_actions_t = torch.tensor([identity_rt], dtype=torch.float32)

    all_gen_frames = []
    paired_for_video: List[tuple] = []
    per_frame = []
    timings = []

    for ch in range(args.num_chunks):
        seg_start = args.start_frame + ch * args.chunk_frames
        actions = build_gt_trajectory_actions(args.dataset_base, args.video_name, seg_start, args.chunk_frames)
        if actions is None:
            raise RuntimeError(f"No GT actions for {args.video_name} start={seg_start}")
        action_path = os.path.join(args.output_dir, f"_gt_actions_chunk{ch}.json")
        with open(action_path, "w", encoding="utf-8") as f:
            json.dump(actions, f, indent=2)

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
            log_prefix="[replay_gt]",
        )
        t1 = time.time()
        timings.append({"chunk": ch, "seconds": t1 - t0, "fps": (len(frames) / max(1e-6, (t1 - t0)))})

        # compute per-frame mse vs GT for this segment
        for i, fr in enumerate(frames):
            gen_pil = _frame_to_pil(fr, w, h)
            gen = np.array(gen_pil, dtype=np.uint8)
            gt = _read_gt_frame(args.dataset_base, args.video_name, seg_start + i, w, h)
            # keep GT/gen pair for the side-by-side video even when GT is missing
            paired_for_video.append((gen_pil, gt))
            if gt is None:
                continue
            mse = _mse(gen, gt)
            ssim_v = _compute_ssim(gen, gt)
            lpips_v = None
            if lpips_model is not None and _pl is not None:
                lpips_v = _pl.lpips_distance(gen, gt, lpips_model, device=args.lpips_device)
            row: Dict[str, Any] = {
                "chunk": ch,
                "frame_in_chunk": i,
                "abs_frame": seg_start + i,
                "mse": mse,
                "psnr": _psnr_from_mse(mse),
                "ssim": ssim_v,
                "lpips": lpips_v,
            }
            per_frame.append(row)

        all_gen_frames.extend(frames)

        # prepare context for next chunk using generated frames (same as other evals)
        if ch < args.num_chunks - 1:
            n_ctx = min(args.context_frames, len(frames))
            prev_frames = replay_context_from_generated_frames(frames, n_ctx)
            prev_pil = [_frame_to_pil(f, w, h) for f in prev_frames]
            pipe.load_models_to_device(["vae"])
            with torch.no_grad():
                ctx_latents = encode_context_frames_per_frame(pipe, prev_pil, pipe.device)
            num_ctx_tokens = ctx_latents.shape[2]
            ctx_actions_t = torch.tensor([identity_rt] * num_ctx_tokens, dtype=torch.float32)

    # write outputs
    mp4_path = os.path.join(args.output_dir, "replay_gt_gen_only.mp4")
    save_video(all_gen_frames, mp4_path, fps=15, quality=5)

    sbs_path = None
    if paired_for_video:
        sbs_path = os.path.join(args.output_dir, "replay_gt_vs_gen_sidebyside.mp4")
        save_video(_build_sidebyside(paired_for_video, w, h), sbs_path, fps=15, quality=5)

    def _mean_optional(key: str) -> Optional[float]:
        vals = [r[key] for r in per_frame if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    metrics = {
        "video_name": args.video_name,
        "start_frame": args.start_frame,
        "num_chunks": args.num_chunks,
        "chunk_frames": args.chunk_frames,
        "mean_mse": float(np.mean([r["mse"] for r in per_frame])) if per_frame else None,
        "mean_psnr": float(np.mean([r["psnr"] for r in per_frame])) if per_frame else None,
        "mean_ssim": _mean_optional("ssim"),
        "mean_lpips": _mean_optional("lpips"),
        "timings": timings,
        "output_video": mp4_path,
        "sidebyside_video": sbs_path,
    }
    metric_definitions = {
        "mean_mse": "Mean squared error vs dataset GT frames (lower is better).",
        "mean_psnr": "Mean PSNR vs GT (higher is better).",
        "mean_ssim": "Mean structural similarity vs GT, data_range=255 (higher is better). Requires scikit-image.",
        "mean_lpips": "Mean LPIPS (Alex) vs GT (lower is better). Requires pip package lpips; use --no_lpips to skip.",
        "fid_fvd": "Pooled FID/FVD over all long-horizon runs: see long_horizon_fid_fvd_summary.json from aggregate_long_horizon_fid_fvd.py.",
    }
    payload: Dict[str, Any] = {
        "metric_definitions": metric_definitions,
        "quality_notes": quality_notes,
        "metrics": metrics,
        "per_frame": per_frame,
    }
    with open(os.path.join(args.output_dir, "replay_gt_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if args.write_csv and per_frame:
        csv_path = os.path.join(args.output_dir, "per_frame_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=list(per_frame[0].keys()))
            wri.writeheader()
            wri.writerows(per_frame)

    print(f"Done. Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()

