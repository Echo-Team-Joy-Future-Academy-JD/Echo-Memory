"""
Temporal Coherence metrics:
- Frame-to-frame PSNR (mean/médian over consecutive pairs).
- Optional: optical flow consistency (warp frame t by flow t->t+1, compare to frame t+1); requires flow model.
- Optional: FVD (Fréchet Video Distance); requires I3D.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from .common import discover_evals_videos, load_video_frames

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _psnr_two(f1: np.ndarray, f2: np.ndarray) -> float:
    if f1.shape != f2.shape and HAS_CV2:
        f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]), interpolation=cv2.INTER_LINEAR)
    if HAS_SKIMAGE:
        return float(psnr_skimage(f1, f2, data_range=255))
    mse = np.mean((f1.astype(np.float64) - f2.astype(np.float64)) ** 2)
    return float(10 * np.log10((255 ** 2) / mse)) if mse > 0 else 100.0


def run_temporal_coherence(
    evals_root: str,
    enable_flow: bool = False,
    enable_fvd: bool = False,
    video_paths: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Compute temporal coherence: frame-to-frame PSNR; optionally flow consistency and FVD.
    """
    if video_paths is None:
        video_paths = discover_evals_videos(evals_root)

    per_video = []
    all_mean_psnr = []
    all_median_psnr = []

    for rel, absp in video_paths:
        if not os.path.isfile(absp):
            continue
        frames = load_video_frames(absp)
        if frames.shape[0] < 2:
            per_video.append({"rel": rel, "mean_frame_psnr": None, "median_frame_psnr": None})
            continue
        psnrs = [_psnr_two(frames[i], frames[i + 1]) for i in range(frames.shape[0] - 1)]
        mean_p = float(np.mean(psnrs))
        med_p = float(np.median(psnrs))
        all_mean_psnr.append(mean_p)
        all_median_psnr.append(med_p)
        row = {"rel": rel, "mean_frame_psnr": mean_p, "median_frame_psnr": med_p}
        if enable_flow:
            row["flow_consistency"] = None  # placeholder: would run flow model
        if enable_fvd:
            row["fvd"] = None  # placeholder
        per_video.append(row)

    aggregate = {}
    if all_mean_psnr:
        aggregate["mean_frame_psnr"] = float(np.mean(all_mean_psnr))
        aggregate["median_frame_psnr"] = float(np.median(all_median_psnr))
    if enable_flow:
        aggregate["flow_consistency_note"] = "Optional: enable with --enable_flow when RAFT/torchvision flow available."
    if enable_fvd:
        aggregate["fvd_note"] = "Optional: enable with --enable_fvd when pytorch-fvd/I3D available."

    return {
        "dimension": "temporal_coherence",
        "params": {"enable_flow": enable_flow, "enable_fvd": enable_fvd},
        "per_video": per_video,
        "aggregate": aggregate,
        "num_videos": len(per_video),
    }


def main():
    p = argparse.ArgumentParser(description="Temporal Coherence metrics")
    p.add_argument("--evals_root", type=str, required=True)
    p.add_argument("--enable_flow", action="store_true")
    p.add_argument("--enable_fvd", action="store_true")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    result = run_temporal_coherence(
        args.evals_root,
        enable_flow=args.enable_flow,
        enable_fvd=args.enable_fvd,
    )
    out = json.dumps(result, indent=2)
    print(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
