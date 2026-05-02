#!/usr/bin/env python3
"""
Randomly pick N revisit candidates (gen-only MP4) and compute average PSNR(+LPIPS if available).

Intended for eval_v2/static_consistency outputs:
- <evals_root>/in_domain/loop_closure/**/*_gen_only.mp4
- <evals_root>/in_domain/combo_revisit_fixed_first/**/*_gen_only.mp4

Outputs:
- per_video metrics JSON
- aggregate summary JSON
- optional side-by-side visualization PNGs (first vs last frame)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple


def _discover_candidates(evals_root: str) -> List[str]:
    cands: List[str] = []
    for root, _dirs, files in os.walk(evals_root):
        for f in files:
            if f.endswith("_gen_only.mp4"):
                cands.append(os.path.join(root, f))
    return sorted(cands)


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate revisit metrics by random sampling")
    p.add_argument("--evals_root", type=str, required=True, help="Path to static_consistency/in_domain (or similar)")
    p.add_argument("--num_samples", type=int, default=5, help="How many videos to sample (min(len, num_samples))")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--write_viz", action="store_true", help="Also export side-by-side PNG for sampled videos")
    p.add_argument("--viz_dir", type=str, default=None, help="Override visualization output directory")
    args = p.parse_args()

    evals_root = os.path.abspath(args.evals_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    candidates = _discover_candidates(evals_root)
    if not candidates:
        summary = {
            "num_candidates": 0,
            "num_samples": 0,
            "aggregate": {},
            "per_video": [],
            "note": "No *_gen_only.mp4 found under evals_root",
        }
        with open(os.path.join(output_dir, "revisit_metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[aggregate_revisit_metrics] No candidates under: {evals_root}", file=sys.stderr)
        return

    rng = random.Random(args.seed)
    num_samples = min(len(candidates), max(1, int(args.num_samples)))
    selected = rng.sample(candidates, k=num_samples) if num_samples < len(candidates) else candidates

    # Import metrics with local sys.path insertion to avoid needing package __init__.py
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _script_dir)
    import psnr_lpips  # type: ignore

    # Optional visualization
    if args.write_viz:
        viz_dir = args.viz_dir or os.path.join(output_dir, "viz_first_vs_last")
        os.makedirs(viz_dir, exist_ok=True)
    else:
        viz_dir = None

    per_video: List[Dict[str, Any]] = []
    psnr_list: List[float] = []
    lpips_list: List[float] = []

    for vp in selected:
        rel = os.path.relpath(vp, evals_root)
        try:
            res = psnr_lpips.compute_revisit_metrics(vp, device=args.device)
            per_video.append({"rel": rel, **res})
            if res.get("psnr") is not None:
                psnr_list.append(float(res["psnr"]))
            if res.get("lpips") is not None:
                lpips_list.append(float(res["lpips"]))
        except Exception as e:
            per_video.append({"rel": rel, "error": str(e)})

        if args.write_viz and viz_dir is not None:
            # Export side-by-side png
            try:
                import subprocess
                _viz_script = os.path.join(os.path.dirname(_script_dir), "visualize", "revisit_pairs_viz.py")
                out_png = os.path.join(viz_dir, rel.replace(os.sep, "__") + ".png")
                os.makedirs(os.path.dirname(out_png), exist_ok=True)
                subprocess.run(
                    ["python3", _viz_script, "--video", vp, "--output", out_png],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    aggregate = {
        "num_candidates": len(candidates),
        "num_samples": len(selected),
        "mean_psnr": _safe_mean(psnr_list),
        "mean_lpips": _safe_mean(lpips_list),
        "note": "LPIPS is None for each video when lpips dependency is missing.",
    }

    summary = {
        "evals_root": evals_root,
        "seed": args.seed,
        "aggregate": aggregate,
        "per_video": per_video,
    }
    with open(os.path.join(output_dir, "revisit_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "revisit_metrics_per_video.json"), "w", encoding="utf-8") as f:
        json.dump(per_video, f, indent=2)

    print(f"[aggregate_revisit_metrics] wrote: {output_dir}/revisit_metrics_summary.json")


if __name__ == "__main__":
    main()

