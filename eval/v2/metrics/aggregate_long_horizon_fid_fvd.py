#!/usr/bin/env python3
"""Aggregate long-horizon visual quality with FID/FVD (when available)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

_metrics_dir = os.path.dirname(os.path.abspath(__file__))
if _metrics_dir not in sys.path:
    sys.path.insert(0, _metrics_dir)
try:
    import psnr_lpips as _pl

    _HAS_PSNR_LPIPS = True
except Exception:
    _pl = None  # type: ignore
    _HAS_PSNR_LPIPS = False

try:
    from skimage.metrics import structural_similarity as _skimage_ssim

    _HAS_SKIMAGE = True
except Exception:
    _skimage_ssim = None  # type: ignore
    _HAS_SKIMAGE = False


@dataclass
class RunItem:
    run_dir: str
    video_name: str
    start_frame: int
    num_chunks: int
    chunk_frames: int
    gen_mp4: str


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_definitions() -> Dict[str, str]:
    return {
        "fid": "Pooled Frechet Inception Distance over all aligned frames from all runs (lower is better).",
        "fvd": "Pooled Frechet Video Distance over runs (lower is better). Requires multiple clips.",
        "per_run.mean_ssim": "Mean SSIM vs GT for that run (higher is better).",
        "per_run.mean_lpips": "Mean LPIPS (Alex) vs GT for that run (lower is better).",
        "per_run.fid_frame_divergence": (
            "FID computed only on that run's aligned frames (treats frames as samples). "
            "Not comparable to standard multi-video dataset FID; diagnostic only."
        ),
    }


def _discover_runs(root: str) -> List[RunItem]:
    out: List[RunItem] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "replay_gt_metrics.json" not in filenames:
            continue
        metrics_path = os.path.join(dirpath, "replay_gt_metrics.json")
        data = _read_json(metrics_path)
        m = data.get("metrics") or {}
        gen_mp4 = m.get("output_video") or os.path.join(dirpath, "replay_gt_gen_only.mp4")
        if not os.path.isfile(gen_mp4):
            continue
        try:
            out.append(
                RunItem(
                    run_dir=dirpath,
                    video_name=str(m["video_name"]),
                    start_frame=int(m["start_frame"]),
                    num_chunks=int(m["num_chunks"]),
                    chunk_frames=int(m["chunk_frames"]),
                    gen_mp4=gen_mp4,
                )
            )
        except Exception:
            continue
    return sorted(out, key=lambda x: x.run_dir)


def _read_video_rgb(video_path: str, max_frames: int = 0) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def _load_gt_frames(
    dataset_base: str,
    video_name: str,
    start_frame: int,
    total_frames: int,
    resize_wh: Tuple[int, int],
) -> List[np.ndarray]:
    w, h = resize_wh
    base = os.path.join(dataset_base, "frames", video_name)
    out: List[np.ndarray] = []
    for i in range(total_frames):
        idx = start_frame + i
        p1 = os.path.join(base, f"{idx:04d}.png")
        p2 = os.path.join(base, f"{idx}.png")
        p = p1 if os.path.isfile(p1) else p2
        if not os.path.isfile(p):
            break
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            break
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return out


def _frame_tensor_uint8(frames: List[np.ndarray]) -> torch.Tensor:
    # [N,H,W,C] -> [N,C,H,W] uint8
    arr = np.stack(frames, axis=0).astype(np.uint8)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def _try_fid(real_imgs: torch.Tensor, fake_imgs: torch.Tensor, device: str) -> Tuple[Optional[float], str]:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as e:
        return None, f"torchmetrics FID unavailable: {e}"
    try:
        metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        metric.update(real_imgs.to(device), real=True)
        metric.update(fake_imgs.to(device), real=False)
        val = metric.compute().item()
        return float(val), ""
    except Exception as e:
        return None, f"FID compute failed: {e}"


def _videos_to_uint8_tensor(videos: List[List[np.ndarray]], t_max: int) -> Optional[torch.Tensor]:
    # -> [N,T,C,H,W] uint8, truncated to min length and t_max
    if not videos:
        return None
    min_t = min(len(v) for v in videos if v)
    if min_t <= 0:
        return None
    if t_max > 0:
        min_t = min(min_t, t_max)
    clips = []
    for v in videos:
        clip = np.stack(v[:min_t], axis=0).astype(np.uint8)  # [T,H,W,C]
        clips.append(torch.from_numpy(clip).permute(0, 3, 1, 2))  # [T,C,H,W]
    return torch.stack(clips, dim=0).contiguous()  # [N,T,C,H,W]


def _ssim_rgb(fake: np.ndarray, real: np.ndarray) -> Optional[float]:
    if not _HAS_SKIMAGE or _skimage_ssim is None:
        return None
    try:
        try:
            return float(_skimage_ssim(real, fake, channel_axis=2, data_range=255))
        except TypeError:
            return float(_skimage_ssim(real, fake, multichannel=True, data_range=255))
    except Exception:
        return None


def _try_fvd(real_videos: torch.Tensor, fake_videos: torch.Tensor, device: str) -> Tuple[Optional[float], str]:
    try:
        from torchmetrics.video.fvd import FrechetVideoDistance
    except Exception as e:
        return None, f"torchmetrics FVD unavailable: {e}"
    try:
        metric = FrechetVideoDistance(feature=400).to(device)
        metric.update(real_videos.to(device), real=True)
        metric.update(fake_videos.to(device), real=False)
        val = metric.compute().item()
        return float(val), ""
    except Exception as e:
        return None, f"FVD compute failed: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate long-horizon FID/FVD from replay_gt outputs")
    ap.add_argument("--root", required=True, help=".../static_consistency/in_domain/long_horizon_gt_replay")
    ap.add_argument("--dataset_base", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_frames_per_video", type=int, default=243)
    ap.add_argument("--max_fvd_frames", type=int, default=81)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    runs = _discover_runs(root)
    if not runs:
        out = {
            "root": root,
            "num_runs": 0,
            "fid": None,
            "fvd": None,
            "per_run": [],
            "metric_definitions": _metric_definitions(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return 0

    device = args.device
    lpips_model = _pl._lpips_model(device=device) if _HAS_PSNR_LPIPS and _pl is not None else None

    per_run: List[Dict[str, Any]] = []
    all_real_frames: List[np.ndarray] = []
    all_fake_frames: List[np.ndarray] = []
    real_videos: List[List[np.ndarray]] = []
    fake_videos: List[List[np.ndarray]] = []

    for r in runs:
        fake = _read_video_rgb(r.gen_mp4, max_frames=args.max_frames_per_video)
        if not fake:
            per_run.append({"run_dir": r.run_dir, "error": f"cannot read generated video {r.gen_mp4}"})
            continue
        h, w = fake[0].shape[0], fake[0].shape[1]
        total = min(len(fake), r.num_chunks * r.chunk_frames)
        real = _load_gt_frames(args.dataset_base, r.video_name, r.start_frame, total, (w, h))
        n = min(len(real), len(fake))
        if n <= 0:
            per_run.append({"run_dir": r.run_dir, "error": "no aligned real/fake frames"})
            continue
        real = real[:n]
        fake = fake[:n]
        all_real_frames.extend(real)
        all_fake_frames.extend(fake)
        real_videos.append(real)
        fake_videos.append(fake)

        ssims: List[float] = []
        lpips_vals: List[float] = []
        for fr, gt in zip(fake, real):
            sv = _ssim_rgb(fr, gt)
            if sv is not None:
                ssims.append(sv)
            if lpips_model is not None and _pl is not None:
                lv = _pl.lpips_distance(fr, gt, lpips_model, device=device)
                if lv is not None:
                    lpips_vals.append(lv)

        real_t = _frame_tensor_uint8(real)
        fake_t = _frame_tensor_uint8(fake)
        fid_run, _fid_note = _try_fid(real_t, fake_t, device=device)

        per_run.append(
            {
                "run_dir": r.run_dir,
                "video_name": r.video_name,
                "start_frame": r.start_frame,
                "num_frames_used": n,
                "mean_ssim": float(np.mean(ssims)) if ssims else None,
                "mean_lpips": float(np.mean(lpips_vals)) if lpips_vals else None,
                "fid_frame_divergence": fid_run,
            }
        )

    fid_val: Optional[float] = None
    fvd_val: Optional[float] = None
    notes: List[str] = []

    if all_real_frames and all_fake_frames:
        real_img_t = _frame_tensor_uint8(all_real_frames)
        fake_img_t = _frame_tensor_uint8(all_fake_frames)
        fid_val, fid_note = _try_fid(real_img_t, fake_img_t, device=device)
        if fid_note:
            notes.append(fid_note)
    else:
        notes.append("No valid aligned frames for FID.")

    rv = _videos_to_uint8_tensor(real_videos, t_max=args.max_fvd_frames)
    fv = _videos_to_uint8_tensor(fake_videos, t_max=args.max_fvd_frames)
    if rv is not None and fv is not None:
        fvd_val, fvd_note = _try_fvd(rv, fv, device=device)
        if fvd_note:
            notes.append(fvd_note)
    else:
        notes.append("No valid aligned videos for FVD.")

    out = {
        "root": root,
        "dataset_base": os.path.abspath(args.dataset_base),
        "num_runs": len(per_run),
        "fid": fid_val,
        "fvd": fvd_val,
        "notes": notes,
        "per_run": per_run,
        "metric_definitions": _metric_definitions(),
    }
    outp = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[aggregate_long_horizon_fid_fvd] runs={len(per_run)} fid={fid_val} fvd={fvd_val} -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
