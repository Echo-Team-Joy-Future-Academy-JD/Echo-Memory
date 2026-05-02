"""
Loop Closure / Revisit metrics:
- View Recall PSNR: last frame vs first frame (revisit same view)
- Revisit SSIM: same
- Camera trajectory reference error (optional): expected pose vs GT pose when dataset provided
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from .common import discover_loop_closure_videos, load_video_frames

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
    from skimage.metrics import structural_similarity as ssim_skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _ensure_same_size(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if img1.shape == img2.shape:
        return img1, img2
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)
    return img1, img2


def _psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0) -> float:
    img1, img2 = _ensure_same_size(img1, img2)
    if HAS_SKIMAGE:
        return float(psnr_skimage(img1, img2, data_range=data_range))
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return float(10 * np.log10((data_range ** 2) / mse)) if mse > 0 else 100.0


def _ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0) -> float:
    img1, img2 = _ensure_same_size(img1, img2)
    if not HAS_SKIMAGE:
        return 0.0
    if img1.ndim == 3:
        return float(ssim_skimage(img1, img2, data_range=data_range, channel_axis=2))
    return float(ssim_skimage(img1, img2, data_range=data_range))


def view_recall_psnr_ssim(video_path: str) -> dict[str, Any] | None:
    """
    For a loop video: first frame = start view, last frame = revisit. Compute PSNR/SSIM(last, first).
    Returns dict with view_recall_psnr, view_recall_ssim, num_frames; or None if <2 frames.
    """
    frames = load_video_frames(video_path)
    n = frames.shape[0]
    if n < 2:
        return None
    first = frames[0]
    last = frames[-1]
    return {
        "view_recall_psnr": _psnr(first, last),
        "view_recall_ssim": _ssim(first, last),
        "num_frames": n,
    }


def run_loop_closure(
    evals_root: str,
    dataset_base: str | None = None,
    video_paths: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Compute loop closure metrics on 1_loop_4chunk and 3_multi_ctx_4chunk gen_only MP4s.
    Optionally compute trajectory reference error when dataset_base is set (stub: aggregate empty).
    """
    if video_paths is None:
        video_paths = discover_loop_closure_videos(evals_root)

    per_video = []
    psnrs = []
    ssims = []

    for rel, absp in video_paths:
        if not os.path.isfile(absp):
            continue
        res = view_recall_psnr_ssim(absp)
        if res is None:
            per_video.append({"rel": rel, "view_recall_psnr": None, "view_recall_ssim": None, "num_frames": 0})
            continue
        psnrs.append(res["view_recall_psnr"])
        ssims.append(res["view_recall_ssim"])
        per_video.append({"rel": rel, **res})

    aggregate = {}
    if psnrs:
        aggregate["mean_view_recall_psnr"] = float(np.mean(psnrs))
        aggregate["min_view_recall_psnr"] = float(np.min(psnrs))
        aggregate["mean_view_recall_ssim"] = float(np.mean(ssims))
        aggregate["min_view_recall_ssim"] = float(np.min(ssims))
    if dataset_base:
        aggregate["trajectory_ref_error_note"] = "Optional: set dataset_base and implement pose vs GT; currently not computed."

    return {
        "dimension": "loop_closure",
        "params": {"dataset_base": dataset_base},
        "per_video": per_video,
        "aggregate": aggregate,
        "num_videos": len(per_video),
    }


def main():
    p = argparse.ArgumentParser(description="Loop Closure / Revisit metrics")
    p.add_argument("--evals_root", type=str, required=True, help="evals_ep0 root")
    p.add_argument("--dataset_base", type=str, default=None, help="Dataset root for optional trajectory ref error")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    result = run_loop_closure(args.evals_root, dataset_base=args.dataset_base)
    out = json.dumps(result, indent=2)
    print(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
