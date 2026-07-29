#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFont
from safetensors import safe_open

EVAL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_DIR = REPO_ROOT / "env"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ENV_DIR))
sys.path.insert(0, str(EVAL_DIR))

import loop_utils as irc  # noqa: E402
import memory_baseline_runtime as mbr  # noqa: E402
from diffsynth import save_video  # noqa: E402
from run_replay_loop_two_chunk import (  # noqa: E402
    _frame_to_pil,
    encode_context_frames_per_frame,
    replay_context_from_generated_frames,
    run_one_chunk,
)

from src.model_training.fov_retrieval import convert_rt_to_relative, load_camera_poses_batch, pose_to_rt  # noqa: E402
from src.model_training.fov_retrieval import retrieve_context_frames_advanced  # noqa: E402
from src.model_training.multichunk_sample_utils import (  # noqa: E402
    build_causal_continuation_protocol,
    build_prev_tail_continuation_protocol,
    load_causal_prefix_context,
    load_prev_chunk_tail_rt_actions,
    replay_context_actions_from_segment_actions,
)


def infer_camera_inject_mode_label(ckpt: str, override: str | None = None) -> str:
    if override:
        return override
    lower = (ckpt or "").lower()
    for mode in ("pre_qkv_post", "pre_qkv", "pre_norm", "post"):
        if mode.replace("_", "") in lower or mode in lower:
            return mode
    return os.environ.get("CAMERA_INJECT_MODE", "pre_qkv")


def _rotation_action(yaw_deg: float, chunk_frames: int) -> dict[str, list[float]]:
    out = {}
    denom = max(1, chunk_frames - 1)
    for i in range(chunk_frames):
        yaw = (i / denom) * float(yaw_deg)
        rad = math.radians(yaw)
        c, s = math.cos(rad), math.sin(rad)
        out[str(i)] = [0.0, 0.0, 0.0, c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
    return out


def _translation_action(dx: float, dy: float, chunk_frames: int) -> dict[str, list[float]]:
    out = {}
    denom = max(1, chunk_frames - 1)
    rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    for i in range(chunk_frames):
        t = i / denom
        out[str(i)] = [dx * t, dy * t, 0.0] + rot
    return out


def build_actions(mode: str, out_dir: Path, chunk_frames: int, seed: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    if mode == "rot180_4chunk":
        yaws = [45.0, 45.0, -45.0, -45.0]
        actions = [_rotation_action(y, chunk_frames) for y in yaws]
    elif mode == "rot360_8chunk":
        yaws = [45.0] * 8
        actions = [_rotation_action(y, chunk_frames) for y in yaws]
    elif mode == "right45_return_2chunk":
        # Two-chunk memory probe: turn right 45 degrees, then turn back.
        # The second chunk is the memory-sensitive segment.
        actions = [_rotation_action(-45.0, chunk_frames), _rotation_action(45.0, chunk_frames)]
    elif mode == "random_closed":
        yaw = rng.uniform(20.0, 60.0)
        dx = rng.uniform(-0.12, 0.12)
        dy = rng.uniform(0.06, 0.18)
        actions = [
            _rotation_action(yaw, chunk_frames),
            _translation_action(dx, dy, chunk_frames),
            _rotation_action(-yaw, chunk_frames),
            _translation_action(-dx, -dy, chunk_frames),
        ]
    else:
        raise ValueError(f"unknown mode: {mode}")
    paths = []
    for i, action in enumerate(actions):
        p = out_dir / f"chunk{i:02d}_{mode}.json"
        p.write_text(json.dumps(action, indent=2), encoding="utf-8")
        paths.append(p)
    return paths


def _resize(img: Image.Image, width: int, height: int) -> Image.Image:
    return img.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def _psnr(mse: float) -> float:
    return 100.0 if mse <= 0 else float(10.0 * np.log10((255.0 * 255.0) / mse))


def _ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    except Exception:
        return None


def _save_frames(frames: list[Image.Image], out_dir: Path, prefix: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(frames):
        p = out_dir / f"{prefix}_{i:02d}.png"
        img.save(p)
        paths.append(str(p))
    return paths


def _load_font(size: int = 18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _save_first_last_chunk_changes(
    chunks: list[list[Any]],
    out_dir: Path,
    width: int,
    height: int,
    max_frames: int = 0,
) -> list[str]:
    if len(chunks) < 2:
        return []
    first = [_frame_to_pil(f, width, height) for f in chunks[0]]
    last = [_frame_to_pil(f, width, height) for f in chunks[-1]]
    n = min(len(first), len(last))
    if max_frames and max_frames > 0:
        n = min(n, int(max_frames))
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(18)
    paths = []
    for i in range(n):
        a = first[i].convert("RGB").resize((width, height))
        b = last[i].convert("RGB").resize((width, height))
        diff = ImageChops.difference(a, b)
        canvas = Image.new("RGB", (width * 3, height), "black")
        canvas.paste(a, (0, 0))
        canvas.paste(b, (width, 0))
        canvas.paste(diff, (width * 2, 0))
        draw = ImageDraw.Draw(canvas)
        labels = ("first chunk", "last chunk", "abs diff")
        for j, label in enumerate(labels):
            x0 = j * width
            draw.rectangle([(x0, 0), (x0 + 180, 26)], fill=(0, 0, 0))
            draw.text((x0 + 6, 3), f"{label} f{i:02d}", fill=(255, 255, 255), font=font)
        p = out_dir / f"first_vs_last_{i:03d}.png"
        canvas.save(p)
        paths.append(str(p))
    return paths


def inspect_ckpt_keys(ckpt: str) -> dict[str, Any]:
    tokens = {
        "camera_encoder": "camera_encoder",
        "separate": "separate",
        "seperate": "seperate",
        "action_mlp": "action_mlp",
        "self_attn_with_action": "self_attn_with_action",
        "spatial_memory_module": "spatial_memory_module",
        "block_wise_ssm": "block_wise_ssm",
        "videossm_hybrid": "videossm_hybrid",
        "ssm": "ssm",
    }
    counts = {name: 0 for name in tokens}
    try:
        with safe_open(str(ckpt), framework="pt", device="cpu") as f:
            for key in f.keys():
                for name, token in tokens.items():
                    if token in key:
                        counts[name] += 1
    except Exception as exc:
        return {"inspect_error": f"{type(exc).__name__}: {exc}", **counts}
    return {
        **counts,
        "has_camera_encoder_keys": counts["camera_encoder"] > 0,
        "has_separate_rt_keys": counts["separate"] > 0 or counts["seperate"] > 0,
        "has_action_mlp_keys": counts["action_mlp"] > 0,
        "has_action_attention_keys": counts["self_attn_with_action"] > 0,
        "has_spatial_memory_module_keys": counts["spatial_memory_module"] > 0,
        "has_block_wise_ssm_keys": counts["block_wise_ssm"] > 0,
        "has_videossm_hybrid_keys": counts["videossm_hybrid"] > 0,
        "has_ssm_keys": counts["ssm"] > 0,
    }


def infer_training_memory_source(profile_id: str, ckpt: str) -> str:
    lower = (ckpt or "").lower()
    if (
        "block_wise_ssm_causal_v2" in lower
        or profile_id == "block_wise_ssm_causal_v2"
    ):
        return "causal_prev_prefix"
    if profile_id.startswith("ctx"):
        return "fov"
    if "framepack" in lower:
        return "prev_chunk_tail"
    if "spatial_mem" in lower:
        return "prev_chunk_tail"
    if "block_wise_ssm" in lower or "two_chunk" in lower:
        return "replay_synthetic"
    return "first_frame"


def _identity_rt() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _load_segment_frames(sample: dict[str, Any], width: int, height: int, num_frames: int) -> list[Image.Image]:
    dataset_base = sample.get("dataset_base")
    video_name = sample.get("video_name")
    start_frame = int(sample.get("start_frame") or 0)
    if not dataset_base or not video_name:
        return []
    frames = []
    frames_dir = Path(dataset_base) / "frames" / str(video_name)
    for i in range(num_frames):
        p = frames_dir / f"{start_frame + i:04d}.png"
        if not p.is_file():
            break
        frames.append(_resize(Image.open(p), width, height))
    return frames


def _load_segment_actions(sample: dict[str, Any], num_frames: int) -> list[list[float]]:
    dataset_base = sample.get("dataset_base")
    video_name = sample.get("video_name")
    start_frame = int(sample.get("start_frame") or 0)
    if not dataset_base or not video_name:
        return []
    json_path = Path(dataset_base) / "jsons" / f"{video_name}.json"
    if not json_path.is_file():
        return []
    poses = load_camera_poses_batch(str(json_path), [start_frame + i for i in range(num_frames)])
    rts = [pose_to_rt(p) if p else None for p in poses]
    if not rts or rts[0] is None:
        return []
    valid = [rt if rt is not None else rts[0] for rt in rts]
    return convert_rt_to_relative(valid, valid[0])


def build_initial_context(
    pipe,
    sample: dict[str, Any],
    profile_id: str,
    ckpt: str,
    context_frames: int,
    first_frame: Image.Image,
    width: int,
    height: int,
    chunk_frames: int,
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    source = infer_training_memory_source(profile_id, ckpt)
    ctx_pil: list[Image.Image] = [first_frame]
    ctx_actions: list[list[float]] = [_identity_rt()]
    ctx_indices: list[int] = [int(sample.get("start_frame") or 0)]
    ctx_source_detail = "first_frame_fallback"

    if sample.get("domain") == "train" and sample.get("dataset_base") and sample.get("video_name"):
        data = {
            "video": _load_segment_frames(sample, width, height, chunk_frames),
            "video_name": sample.get("video_name"),
            "start_frame": int(sample.get("start_frame") or 0),
            "end_frame": int(sample.get("end_frame") or (int(sample.get("start_frame") or 0) + chunk_frames - 1)),
        }
        try:
            if source == "fov":
                # Training used FOV retrieval with top_k=context_frames-1 and 10% random drop.
                # Eval disables random drop for deterministic full-memory in-domain measurement.
                cf, ca, ci, _cur, _vn, detail = retrieve_context_frames_advanced(
                    data=data,
                    dataset_base_path=str(sample["dataset_base"]),
                    top_k=max(0, int(context_frames) - 1),
                    drop_overlap_probability=0.0,
                    use_rt_relative=True,
                    retrieval_method="fov",
                )
                if cf:
                    ctx_pil = [_resize(x, width, height) for x in cf[:context_frames]]
                    ctx_actions = [list(x) for x in (ca or [])[: len(ctx_pil)]]
                    ctx_indices = [int(x) for x in (ci or [])[: len(ctx_pil)]]
                    ctx_source_detail = detail
            elif source == "prev_chunk_tail":
                from src.model_training.multichunk_sample_utils import load_prev_chunk_tail_from_disk

                cf, ci = load_prev_chunk_tail_from_disk(
                    str(sample["dataset_base"]),
                    str(sample["video_name"]),
                    int(sample.get("start_frame") or 0),
                    int(context_frames),
                    nearest_first=True,
                )
                ca, _ = load_prev_chunk_tail_rt_actions(
                    str(sample["dataset_base"]),
                    str(sample["video_name"]),
                    int(sample.get("start_frame") or 0),
                    int(context_frames),
                    use_rt_relative=True,
                    nearest_first=True,
                )
                if cf and ca:
                    ctx_pil = [_resize(x, width, height) for x in cf]
                    ctx_actions = [list(x) for x in ca]
                    ctx_indices = [int(x) for x in (ci or [])]
                    ctx_source_detail = "prev_chunk_tail_from_dataset"
            elif source == "causal_prev_prefix":
                cf, ca, _, protocol = load_causal_prefix_context(
                    str(sample["dataset_base"]),
                    str(sample["video_name"]),
                    int(sample.get("start_frame") or 0),
                    int(context_frames),
                    target_num_frames=int(chunk_frames),
                    use_rt_relative=True,
                )
                if cf and ca:
                    ctx_pil = [_resize(x, width, height) for x in cf]
                    ctx_actions = [list(x) for x in ca]
                    ctx_indices = [
                        int(x)
                        for x in protocol.get("context_frame_indices", [])
                    ]
                    ctx_source_detail = "causal_prev_prefix_from_dataset"
            elif source == "replay_synthetic":
                seg_frames = data["video"]
                seg_actions = _load_segment_actions(sample, chunk_frames)
                if seg_frames:
                    replay = replay_context_from_generated_frames(seg_frames, int(context_frames))
                    acts = replay_context_actions_from_segment_actions(seg_actions, chunk_frames, int(context_frames)) if seg_actions else None
                    ctx_pil = [_resize(x, width, height) for x in replay]
                    ctx_actions = [list(x) for x in (acts or [_identity_rt()] * len(ctx_pil))]
                    ctx_indices = []
                    ctx_source_detail = "replay_synthetic_from_dataset_gt_segment"
        except Exception as exc:
            ctx_source_detail = f"{source}_failed_fallback_first_frame:{type(exc).__name__}:{exc}"

    if len(ctx_actions) < len(ctx_pil):
        ctx_actions = ctx_actions + [_identity_rt()] * (len(ctx_pil) - len(ctx_actions))
    if len(ctx_pil) > context_frames:
        ctx_pil = ctx_pil[:context_frames]
        ctx_actions = ctx_actions[:context_frames]
        ctx_indices = ctx_indices[:context_frames]

    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        ctx_latents = encode_context_frames_per_frame(pipe, ctx_pil, pipe.device)
    ctx_actions_t = torch.tensor(ctx_actions[: ctx_latents.shape[2]], dtype=torch.float32)
    meta = {
        "training_memory_source": source,
        "context_source_detail": ctx_source_detail,
        "context_frame_count": len(ctx_pil),
        "context_action_count": len(ctx_actions),
        "context_indices": ctx_indices,
    }
    return ctx_latents, ctx_actions_t, meta


def _action_rows(path: Path) -> list[list[float]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload = payload.get("actions", payload)
    if isinstance(payload, dict):
        values = [
            value
            for _, value in sorted(
                (
                    (int(key), value)
                    for key, value in payload.items()
                    if str(key).isdigit()
                ),
                key=lambda item: item[0],
            )
        ]
    else:
        values = list(payload)
    return [list(value)[:12] for value in values]


def run_case(args: argparse.Namespace, ckpt_row: dict[str, Any], sample: dict[str, Any], mode: str, rank: int) -> None:
    ckpt = ckpt_row["ckpt"]
    run_id = ckpt_row.get("run_id") or Path(ckpt).parent.name
    sample_id = sample["sample_id"]
    out_dir = Path(args.output_root) / run_id / sample["domain"] / sample_id / mode
    if (out_dir / "stage1_metrics.json").is_file() and not args.force:
        print(f"[stage1] skip existing {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = args.base_model or os.environ.get("WAN_BASE_MODEL") or ""
    if not base_model:
        raise SystemExit("Set WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B or pass --base-model")
    load_kw = dict(
        add_action_attn=True,
        action_use_temporal_attention=True,
    )
    pipe = irc.load_pipeline_and_ckpt(
        ckpt,
        f"{base_model}/diffusion_pytorch_model.safetensors",
        f"{base_model}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{base_model}/Wan2.1_VAE.pth",
        **load_kw,
    )
    mbr.apply_memory_baseline_pipe(pipe, ckpt)
    context_frames = args.context_frames or mbr.effective_context_frames(ckpt, None)
    profile = mbr.infer_memory_profile_spec(ckpt)
    camera_inject_mode_label = infer_camera_inject_mode_label(ckpt, args.camera_inject_mode or None)
    ckpt_key_evidence = inspect_ckpt_keys(ckpt)
    spatial_module_loaded = getattr(pipe, "spatial_memory_module", None) is not None

    first_frame = _resize(Image.open(sample["first_frame_image"]), args.width, args.height)
    prompt = sample.get("prompt") or "A scene."
    action_paths = build_actions(mode, out_dir / "actions", args.chunk_frames, args.seed + rank)
    ctx_latents, ctx_actions_t, context_meta = build_initial_context(
        pipe,
        sample,
        profile.profile_id if profile else "default_context",
        ckpt,
        context_frames,
        first_frame,
        args.width,
        args.height,
        args.chunk_frames,
    )
    chunks = []
    times = []
    neg = getattr(irc, "DEFAULT_NEGATIVE_PROMPT", "oversaturated colors, overexposed, static, blurry details")
    for ch, action_path in enumerate(action_paths):
        t0 = time.time()
        frames = run_one_chunk(
            pipe,
            prompt,
            neg,
            str(action_path),
            context_latents=ctx_latents,
            num_context_frames=ctx_latents.shape[2],
            context_actions_t=ctx_actions_t,
            chunk_frames=args.chunk_frames,
            h=args.height,
            w=args.width,
            seed=args.seed + ch + rank * 1000,
            sigma_shift=args.sigma_shift,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            inference_noise_level=0.0,
            omit_context_actions=False,
            log_prefix=f"[revisit_suite][{mode}][rank{rank}]",
        )
        times.append(time.time() - t0)
        chunks.append(frames)
        if ch < len(action_paths) - 1:
            n_ctx = min(context_frames, len(frames))
            frame_pil = [
                _frame_to_pil(f, args.width, args.height) for f in frames
            ]
            source = getattr(pipe, "training_context_source", None)
            if source == "causal_prev_prefix":
                prev, next_actions, _ = build_causal_continuation_protocol(
                    frame_pil,
                    _action_rows(action_path),
                    n_ctx,
                )
            elif source == "prev_chunk_tail":
                prev, next_actions, _ = build_prev_tail_continuation_protocol(
                    frame_pil,
                    _action_rows(action_path),
                    n_ctx,
                    nearest_first=True,
                )
            else:
                prev = replay_context_from_generated_frames(frame_pil, n_ctx)
                next_actions = [_identity_rt()] * len(prev)
            pipe.load_models_to_device(["vae"])
            with torch.no_grad():
                ctx_latents = encode_context_frames_per_frame(pipe, prev, pipe.device)
            ctx_actions_t = torch.tensor(
                next_actions[: ctx_latents.shape[2]], dtype=torch.float32
            )

    all_frames = [f for chunk in chunks for f in chunk]
    save_video(all_frames, str(out_dir / "revisit_gen_only.mp4"), fps=15, quality=5)
    first_np = np.array(first_frame, dtype=np.uint8)
    last_pil = _frame_to_pil(all_frames[-1], args.width, args.height)
    last_np = np.array(last_pil, dtype=np.uint8)
    mse = _mse(first_np, last_np)
    ssim = _ssim(first_np, last_np)

    stage1_dir = out_dir / "stage1_frames"
    first_paths = _save_frames([first_frame], stage1_dir, "first")
    tail_imgs = [_frame_to_pil(f, args.width, args.height) for f in all_frames[-args.save_tail_frames :]]
    tail_paths = _save_frames(tail_imgs, stage1_dir, "revisit_tail")
    chunk_change_paths = []
    if args.save_chunk_change_frames:
        chunk_change_paths = _save_first_last_chunk_changes(
            chunks,
            stage1_dir / "first_last_chunk_changes",
            args.width,
            args.height,
            max_frames=args.max_chunk_change_frames,
        )
    metrics = {
        "ckpt": ckpt,
        "run_id": run_id,
        "sample": sample,
        "mode": mode,
        "rank": rank,
        "num_chunks": len(action_paths),
        "context_frames": context_frames,
        "alignment": {
            "action_injection_impl": "DiTBlock_w_Action + MLP_CamPose(Linear(12,D)); action embedding is added before norm1/self-attn",
            "camera_inject_mode_label_from_path": camera_inject_mode_label,
            "camera_inject_mode_effective": "ignored_by_current_vwm_loader",
            "camera_encoder_separate_t_r_effective": False,
            "camera_encoder_effective": False,
            "rt_encoding_effective": "single_linear_12d_rt",
            "ckpt_key_evidence": ckpt_key_evidence,
            "memory_profile": profile.profile_id if profile else "default_context",
            "memory_profile_matched": bool(profile),
            "use_framepack_memory": bool(getattr(pipe, "use_framepack_memory", False)),
            "use_framepack_length_compress": bool(getattr(pipe, "use_framepack_length_compress", False)),
            "framepack_ratio": int(getattr(pipe, "framepack_ratio", 2) or 2),
            "use_spatial_memory": bool(getattr(pipe, "use_spatial_memory", False)),
            "use_spatial_memory_legacy": bool(getattr(pipe, "use_spatial_memory_legacy", False)),
            "spatial_memory_module_loaded": spatial_module_loaded,
            "spatial_memory_effective_impl": "SpatialGridMemory" if spatial_module_loaded else ("legacy_adaptive_pool" if getattr(pipe, "use_spatial_memory", False) else "disabled"),
            "spatial_memory_tokens": int(getattr(pipe, "spatial_memory_tokens", 64) or 64),
            "spatial_memory_inject_mode": getattr(pipe, "spatial_memory_inject_mode", None),
            "initial_context": context_meta,
        },
        "closure_mse": mse,
        "closure_psnr": _psnr(mse),
        "closure_ssim": ssim,
        "first_frame_paths": first_paths,
        "revisit_tail_paths": tail_paths,
        "first_last_chunk_change_paths": chunk_change_paths,
        "video_path": str(out_dir / "revisit_gen_only.mp4"),
        "chunk_seconds": times,
    }
    (out_dir / "stage1_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage1] wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    ap.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    ap.add_argument("--modes", default="rot180_4chunk,rot360_8chunk,right45_return_2chunk")
    ap.add_argument("--base-model", default=os.environ.get("WAN_BASE_MODEL", ""))
    ap.add_argument("--camera-inject-mode", default="")
    ap.add_argument("--context-frames", type=int, default=0)
    ap.add_argument("--chunk-frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--sigma-shift", type=float, default=15.0)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-tail-frames", type=int, default=4)
    ap.add_argument("--save-chunk-change-frames", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-chunk-change-frames", type=int, default=0, help="0 = save every frame; otherwise cap first-vs-last chunk visualizations")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    ckpts = manifest.get("ckpts", [])
    samples = manifest.get("samples", [])
    jobs = [(c, s, m) for c in ckpts for s in samples for m in args.modes.split(",") if m]
    shard = [job for i, job in enumerate(jobs) if i % max(1, args.world_size) == args.rank]
    print(f"[stage1] rank={args.rank}/{args.world_size} jobs={len(shard)}/{len(jobs)}")
    for ckpt_row, sample, mode in shard:
        run_case(args, ckpt_row, sample, mode, args.rank)


if __name__ == "__main__":
    main()
