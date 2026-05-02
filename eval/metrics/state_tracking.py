"""
State Tracking metrics (light rules, no GT boxes):
- Frame-to-frame displacement proxy: mean/max L2 difference between consecutive frames (downsampled), as smoothness proxy.
- "Physics" proxy: fraction of consecutive pairs with abnormally large change (potential瞬移).
- Placeholder: object position error / state change accuracy (requires detection+tracking or VLM).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from .common import discover_evals_videos, load_video_frames

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _frame_diff_norm(f1: np.ndarray, f2: np.ndarray, scale: int = 4) -> float:
    """Mean L2 pixel difference between two frames (optionally downsampled)."""
    if not HAS_CV2 or f1.size == 0:
        return 0.0
    if scale > 1:
        h, w = f1.shape[:2]
        f1 = cv2.resize(f1, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(f2, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    d = f1.astype(np.float64) - f2.astype(np.float64)
    return float(np.sqrt(np.mean(d ** 2)))


def run_state_tracking_light(
    frames: np.ndarray,
    displacement_scale: int = 4,
    large_jump_quantile: float = 0.95,
) -> dict[str, Any]:
    """
    Light rules on frame sequence:
    - mean_consecutive_displacement: mean L2 diff between consecutive frames (downsampled).
    - max_consecutive_displacement: max such diff.
    - large_jump_fraction: fraction of consecutive pairs with diff > quantile(large_jump_quantile) of all diffs.
    """
    n = frames.shape[0]
    if n < 2:
        return {
            "mean_consecutive_displacement": 0.0,
            "max_consecutive_displacement": 0.0,
            "large_jump_fraction": 0.0,
        }
    diffs = []
    for i in range(n - 1):
        d = _frame_diff_norm(frames[i], frames[i + 1], scale=displacement_scale)
        diffs.append(d)
    diffs = np.array(diffs)
    thresh = float(np.quantile(diffs, large_jump_quantile)) if len(diffs) else 0.0
    large = np.sum(diffs >= thresh) / max(1, len(diffs))
    return {
        "mean_consecutive_displacement": float(np.mean(diffs)),
        "max_consecutive_displacement": float(np.max(diffs)),
        "large_jump_fraction": float(large),
        "large_jump_threshold": thresh,
    }


def run_state_tracking(
    evals_root: str,
    displacement_scale: int = 4,
    large_jump_quantile: float = 0.95,
    video_paths: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Run light state-tracking metrics on all gen_only videos.
    Object position error / state change accuracy require detection+tracking (placeholder).
    """
    if video_paths is None:
        video_paths = discover_evals_videos(evals_root)

    per_video = []
    all_mean_disp = []
    all_large_frac = []

    for rel, absp in video_paths:
        if not os.path.isfile(absp):
            continue
        frames = load_video_frames(absp)
        if frames.size == 0:
            per_video.append({"rel": rel, "mean_consecutive_displacement": None, "large_jump_fraction": None})
            continue
        res = run_state_tracking_light(frames, displacement_scale, large_jump_quantile)
        all_mean_disp.append(res["mean_consecutive_displacement"])
        all_large_frac.append(res["large_jump_fraction"])
        per_video.append({"rel": rel, **res})

    aggregate = {}
    if all_mean_disp:
        aggregate["mean_consecutive_displacement"] = float(np.mean(all_mean_disp))
        aggregate["mean_large_jump_fraction"] = float(np.mean(all_large_frac))
    aggregate["object_position_note"] = "Optional: add detection+tracking (e.g. ByteTrack, GroundingDINO) for object position error; state change accuracy can use VLM (see semantic_consistency)."

    return {
        "dimension": "state_tracking",
        "params": {"displacement_scale": displacement_scale, "large_jump_quantile": large_jump_quantile},
        "per_video": per_video,
        "aggregate": aggregate,
        "num_videos": len(per_video),
    }


def main():
    p = argparse.ArgumentParser(description="State Tracking (light rules)")
    p.add_argument("--evals_root", type=str, required=True)
    p.add_argument("--displacement_scale", type=int, default=4)
    p.add_argument("--large_jump_quantile", type=float, default=0.95)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    result = run_state_tracking(
        args.evals_root,
        displacement_scale=args.displacement_scale,
        large_jump_quantile=args.large_jump_quantile,
    )
    out = json.dumps(result, indent=2)
    print(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
