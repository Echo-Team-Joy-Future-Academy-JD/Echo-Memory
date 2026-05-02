#!/usr/bin/env python3
"""Adjacent-frame consistency for generated video (sequence self-consistency proxy).

Interpreting metrics:
- mean_adjacent_mse: mean RGB MSE between consecutive frames (lower = smoother change; near-zero may indicate collapse).
- mean_adjacent_ssim: structural similarity between t and t+1 (higher = less motion / very smooth).

This does NOT replace GT-aligned metrics (see replay_gt_metrics.json). Use together with long_horizon PSNR/SSIM/LPIPS vs GT.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as _skimage_ssim

    _HAS_SKIMAGE = True
except Exception:
    _skimage_ssim = None  # type: ignore
    _HAS_SKIMAGE = False


def _read_video_rgb(video_path: str, max_frames: int = 0) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    out: List[np.ndarray] = []
    if not cap.isOpened():
        return out
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames > 0 and len(out) >= max_frames:
            break
    cap.release()
    return out


def _ssim_pair(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not _HAS_SKIMAGE or _skimage_ssim is None:
        return None
    try:
        try:
            return float(_skimage_ssim(a, b, channel_axis=2, data_range=255))
        except TypeError:
            return float(_skimage_ssim(a, b, multichannel=True, data_range=255))
    except Exception:
        return None


def metrics_for_video(video_path: str, max_frames: int = 0) -> Dict[str, Any]:
    frames = _read_video_rgb(os.path.abspath(video_path), max_frames=max_frames)
    if len(frames) < 2:
        return {
            "video": os.path.abspath(video_path),
            "num_frames": len(frames),
            "mean_adjacent_mse": None,
            "mean_adjacent_ssim": None,
            "notes": ["need at least 2 frames"],
        }
    mses: List[float] = []
    ssims: List[float] = []
    for i in range(len(frames) - 1):
        a = frames[i].astype(np.float64)
        b = frames[i + 1].astype(np.float64)
        mses.append(float(np.mean((a - b) ** 2)))
        sv = _ssim_pair(frames[i], frames[i + 1])
        if sv is not None:
            ssims.append(sv)
    return {
        "video": os.path.abspath(video_path),
        "num_frames": len(frames),
        "num_adjacent_pairs": len(frames) - 1,
        "mean_adjacent_mse": float(np.mean(mses)),
        "mean_adjacent_ssim": float(np.mean(ssims)) if ssims else None,
        "metric_definitions": {
            "mean_adjacent_mse": "Mean RGB MSE between consecutive generated frames.",
            "mean_adjacent_ssim": "Mean SSIM between consecutive frames (optional; needs scikit-image).",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Adjacent-frame consistency for one mp4")
    ap.add_argument("--video", required=True)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()
    out = metrics_for_video(args.video, max_frames=args.max_frames)
    s = json.dumps(out, indent=2)
    print(s)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
