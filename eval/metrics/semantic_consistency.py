"""
Semantic / Logic Consistency metrics:
- Rule-based physics violation rate: e.g. fraction of consecutive frame pairs with abnormally large change (proxy for teleport).
- Optional: VLM-based common-sense / physics plausibility (placeholder).
- WorldModelBench: documented as external; not implemented here.
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


def _frame_diff_mean_l2(f1: np.ndarray, f2: np.ndarray, scale: int = 4) -> float:
    if not HAS_CV2 or f1.size == 0:
        return 0.0
    if scale > 1:
        h, w = f1.shape[:2]
        f1 = cv2.resize(f1, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(f2, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    d = f1.astype(np.float64) - f2.astype(np.float64)
    return float(np.sqrt(np.mean(d ** 2)))


def run_semantic_consistency(
    evals_root: str,
    teleport_threshold_quantile: float = 0.98,
    displacement_scale: int = 4,
    video_paths: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Rule-based physics: treat consecutive pairs with displacement above threshold as potential violation.
    violation_rate = fraction of pairs with diff > quantile(teleport_threshold_quantile) of all diffs (per-video).
    """
    if video_paths is None:
        video_paths = discover_evals_videos(evals_root)

    per_video = []
    all_rates = []

    for rel, absp in video_paths:
        if not os.path.isfile(absp):
            continue
        frames = load_video_frames(absp)
        if frames.shape[0] < 2:
            per_video.append({"rel": rel, "physics_violation_rate": 0.0})
            continue
        diffs = [_frame_diff_mean_l2(frames[i], frames[i + 1], scale=displacement_scale) for i in range(frames.shape[0] - 1)]
        thresh = float(np.quantile(diffs, teleport_threshold_quantile))
        n = len(diffs)
        violations = sum(1 for d in diffs if d >= thresh)
        rate = violations / n if n else 0.0
        all_rates.append(rate)
        per_video.append({
            "rel": rel,
            "physics_violation_rate": rate,
            "threshold_used": thresh,
            "num_pairs": n,
        })

    aggregate = {}
    if all_rates:
        aggregate["mean_physics_violation_rate"] = float(np.mean(all_rates))
        aggregate["max_physics_violation_rate"] = float(np.max(all_rates))
    aggregate["vlm_note"] = "Optional: use VLM to score physics/commonsense per clip; see README."
    aggregate["world_model_bench_note"] = "WorldModelBench: use official protocol and data; export evals to their format if needed (see README)."

    return {
        "dimension": "semantic_consistency",
        "params": {
            "teleport_threshold_quantile": teleport_threshold_quantile,
            "displacement_scale": displacement_scale,
        },
        "per_video": per_video,
        "aggregate": aggregate,
        "num_videos": len(per_video),
    }


def main():
    p = argparse.ArgumentParser(description="Semantic/Logic Consistency (rule-based physics)")
    p.add_argument("--evals_root", type=str, required=True)
    p.add_argument("--teleport_quantile", type=float, default=0.98)
    p.add_argument("--displacement_scale", type=int, default=4)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    result = run_semantic_consistency(
        args.evals_root,
        teleport_threshold_quantile=args.teleport_quantile,
        displacement_scale=args.displacement_scale,
    )
    out = json.dumps(result, indent=2)
    print(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
