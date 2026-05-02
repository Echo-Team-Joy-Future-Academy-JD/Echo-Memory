"""
Long-Horizon Consistency metrics:
- Stable sequence length (frames until quality collapse)
- Frame-to-frame drift rate (fraction of consecutive pairs below similarity threshold)
Optional: CLIP-based drift when available.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from .common import discover_evals_videos, load_video_frames

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
    from skimage.metrics import structural_similarity as ssim_skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def _psnr_simple(img1: "np.ndarray", img2: "np.ndarray", data_range: float = 255.0) -> float:
    """PSNR from MSE (same size, float images)."""
    import numpy as np
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse <= 0:
        return 100.0
    return float(10 * np.log10((data_range ** 2) / mse))


def _frame_similarity_psnr(f1: "np.ndarray", f2: "np.ndarray") -> float:
    """Consecutive frame PSNR; resize f2 to f1 if shape mismatch."""
    import numpy as np
    if f1.shape != f2.shape:
        from PIL import Image
        import cv2
        h, w = f1.shape[:2]
        f2 = cv2.resize(f2, (w, h), interpolation=cv2.INTER_LINEAR)
    if HAS_SKIMAGE:
        return float(psnr_skimage(f1, f2, data_range=255))
    return _psnr_simple(f1, f2)


def compute_stable_length_and_drift(
    frames: "np.ndarray",
    collapse_psnr_threshold: float = 15.0,
    drift_psnr_threshold: float = 18.0,
) -> tuple[int, float, list[float]]:
    """
    Compute stable sequence length (number of frames until first collapse) and drift rate.
    - collapse: first frame index i where PSNR(frames[i], frames[i-1]) < collapse_psnr_threshold; length = that i (or len(frames) if never).
    - drift_rate: fraction of consecutive pairs with PSNR < drift_psnr_threshold.
    Returns (stable_length, drift_rate, list of consecutive PSNRs).
    """
    import numpy as np
    n = frames.shape[0]
    if n <= 1:
        return n, 0.0, []

    psnrs = []
    stable_length = n
    for i in range(1, n):
        p = _frame_similarity_psnr(frames[i - 1], frames[i])
        psnrs.append(p)
        if p < collapse_psnr_threshold and stable_length == n:
            stable_length = i  # collapse at frame i (0-indexed: frame i is first "bad")
    pairs = max(1, n - 1)
    below = sum(1 for p in psnrs if p < drift_psnr_threshold)
    drift_rate = below / pairs
    return stable_length, drift_rate, psnrs


def run_long_horizon_consistency(
    evals_root: str,
    collapse_psnr_threshold: float = 15.0,
    drift_psnr_threshold: float = 18.0,
    video_paths: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Run long-horizon consistency metrics on evals_ep0 outputs.
    Returns dict with per_video results and aggregate.
    """
    if video_paths is None:
        video_paths = discover_evals_videos(evals_root)

    per_video = []
    all_stable_lengths = []
    all_drift_rates = []

    for rel, absp in video_paths:
        if not os.path.isfile(absp):
            continue
        frames = load_video_frames(absp)
        if frames.size == 0:
            per_video.append({"rel": rel, "stable_length": 0, "drift_rate": 0.0, "num_frames": 0})
            continue
        n = frames.shape[0]
        stable_length, drift_rate, psnrs = compute_stable_length_and_drift(
            frames, collapse_psnr_threshold, drift_psnr_threshold
        )
        all_stable_lengths.append(stable_length)
        all_drift_rates.append(drift_rate)
        per_video.append({
            "rel": rel,
            "stable_length": stable_length,
            "drift_rate": drift_rate,
            "num_frames": n,
            "mean_consecutive_psnr": float(sum(psnrs) / len(psnrs)) if psnrs else 0.0,
        })

    agg = {}
    if all_stable_lengths:
        agg["mean_stable_length"] = float(sum(all_stable_lengths) / len(all_stable_lengths))
        agg["min_stable_length"] = int(min(all_stable_lengths))
        agg["max_stable_length"] = int(max(all_stable_lengths))
    if all_drift_rates:
        agg["mean_drift_rate"] = float(sum(all_drift_rates) / len(all_drift_rates))
        agg["max_drift_rate"] = float(max(all_drift_rates))

    return {
        "dimension": "long_horizon_consistency",
        "params": {
            "collapse_psnr_threshold": collapse_psnr_threshold,
            "drift_psnr_threshold": drift_psnr_threshold,
        },
        "per_video": per_video,
        "aggregate": agg,
        "num_videos": len(per_video),
    }


def main():
    p = argparse.ArgumentParser(description="Long-Horizon Consistency metrics")
    p.add_argument("--evals_root", type=str, required=True, help="evals_ep0 root (e.g. ckpt_dir/evals_ep0)")
    p.add_argument("--collapse_threshold", type=float, default=15.0, help="PSNR below this = collapse")
    p.add_argument("--drift_threshold", type=float, default=18.0, help="PSNR below this = drift pair")
    p.add_argument("--output", type=str, default=None, help="Write JSON here")
    args = p.parse_args()

    result = run_long_horizon_consistency(
        args.evals_root,
        collapse_psnr_threshold=args.collapse_threshold,
        drift_psnr_threshold=args.drift_threshold,
    )
    out = json.dumps(result, indent=2)
    print(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
