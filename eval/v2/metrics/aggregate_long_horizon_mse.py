#!/usr/bin/env python3
"""Average mean_mse, mean_psnr, mean_ssim, mean_lpips from replay_gt_error.py outputs under a root."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. .../static_consistency/in_domain/long_horizon_gt_replay")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    rows: List[Dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "replay_gt_metrics.json" not in filenames:
            continue
        p = os.path.join(dirpath, "replay_gt_metrics.json")
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            m = (data.get("metrics") or {})
            rows.append(
                {
                    "rel": os.path.relpath(p, root),
                    "video_name": m.get("video_name"),
                    "start_frame": m.get("start_frame"),
                    "num_chunks": m.get("num_chunks"),
                    "mean_mse": m.get("mean_mse"),
                    "mean_psnr": m.get("mean_psnr"),
                    "mean_ssim": m.get("mean_ssim"),
                    "mean_lpips": m.get("mean_lpips"),
                }
            )
        except Exception as e:
            rows.append({"rel": os.path.relpath(p, root), "error": str(e)})

    mse_vals = [float(r["mean_mse"]) for r in rows if r.get("mean_mse") is not None]
    psnr_vals = [float(r["mean_psnr"]) for r in rows if r.get("mean_psnr") is not None]
    ssim_vals = [float(r["mean_ssim"]) for r in rows if r.get("mean_ssim") is not None]
    lpips_vals = [float(r["mean_lpips"]) for r in rows if r.get("mean_lpips") is not None]

    def _mean(xs: List[float]) -> Optional[float]:
        return float(sum(xs) / len(xs)) if xs else None

    summary = {
        "root": root,
        "num_runs": len(rows),
        "aggregate_mean_mse": _mean(mse_vals),
        "aggregate_mean_psnr": _mean(psnr_vals),
        "aggregate_mean_ssim": _mean(ssim_vals),
        "aggregate_mean_lpips": _mean(lpips_vals),
        "per_run": rows,
    }
    outp = os.path.abspath(args.output_json)
    _par = os.path.dirname(outp)
    if _par:
        os.makedirs(_par, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate_long_horizon_mse] runs={len(rows)} mean_mse={summary['aggregate_mean_mse']} -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
