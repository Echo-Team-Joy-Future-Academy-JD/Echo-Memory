#!/usr/bin/env python3
"""Walk long_horizon_gt_replay (or any root) and add temporal_adjacency_metrics.json per run."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_metrics_dir = os.path.dirname(os.path.abspath(__file__))
if _metrics_dir not in sys.path:
    sys.path.insert(0, _metrics_dir)
from temporal_adjacency_metrics import metrics_for_video  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. .../long_horizon_gt_replay")
    ap.add_argument("--output_json", required=True, help="summary over all runs")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    rows: List[Dict[str, Any]] = []
    mse_vals: List[float] = []

    for dirpath, _dirnames, filenames in os.walk(root):
        if "replay_gt_gen_only.mp4" not in filenames:
            continue
        mp4 = os.path.join(dirpath, "replay_gt_gen_only.mp4")
        try:
            m = metrics_for_video(mp4, max_frames=args.max_frames)
            m["rel_dir"] = os.path.relpath(dirpath, root)
            outp = os.path.join(dirpath, "temporal_adjacency_metrics.json")
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
            rows.append({"rel_dir": m["rel_dir"], **{k: v for k, v in m.items() if k != "rel_dir"}})
            if m.get("mean_adjacent_mse") is not None:
                mse_vals.append(float(m["mean_adjacent_mse"]))
        except Exception as e:
            rows.append({"rel_dir": os.path.relpath(dirpath, root), "error": str(e)})

    summary = {
        "root": root,
        "num_runs": len(rows),
        "aggregate_mean_adjacent_mse": float(sum(mse_vals) / len(mse_vals)) if mse_vals else None,
        "per_run": rows,
    }
    outp = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate_long_horizon_temporal_adjacency] runs={len(rows)} -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
