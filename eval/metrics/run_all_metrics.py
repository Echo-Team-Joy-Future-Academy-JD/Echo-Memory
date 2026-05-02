#!/usr/bin/env python3
"""
Entry script for all memory eval dimensions. Run after evals_ep0 has produced videos.
Usage:
  python run_all_metrics.py --evals_root /path/to/ckpt_dir/evals_ep0 [--dataset /path/to/dataset] [--dims 1 2 3 4 5 6] [--output_dir ...]
Output: JSON (and optional CSV summary) under output_dir or evals_root/metrics/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure package-relative imports work when run as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from . import long_horizon_consistency
from . import loop_closure
from . import identity_preservation
from . import state_tracking
from . import temporal_coherence
from . import semantic_consistency

DIMENSIONS = {
    "1": ("long_horizon_consistency", long_horizon_consistency.run_long_horizon_consistency),
    "2": ("loop_closure", loop_closure.run_loop_closure),
    "3": ("identity_preservation", identity_preservation.run_identity_preservation),
    "4": ("state_tracking", state_tracking.run_state_tracking),
    "5": ("temporal_coherence", temporal_coherence.run_temporal_coherence),
    "6": ("semantic_consistency", semantic_consistency.run_semantic_consistency),
}


def main():
    p = argparse.ArgumentParser(description="Run all memory eval metrics on evals_ep0 output")
    p.add_argument("--evals_root", type=str, required=True, help="Path to evals_ep0 root (e.g. ckpt_dir/evals_ep0)")
    p.add_argument("--dataset", type=str, default=None, help="Optional dataset base for loop_closure trajectory ref")
    p.add_argument("--dims", type=str, nargs="*", default=list(DIMENSIONS.keys()), help="Which dimensions to run (default: all 1-6)")
    p.add_argument("--output_dir", type=str, default=None, help="Write results here; default: evals_root/metrics")
    p.add_argument("--write_csv", action="store_true", help="Write aggregate CSV summary")
    p.add_argument("--use_clip", action="store_true", help="Use CLIP in identity_preservation when available")
    args = p.parse_args()

    evals_root = os.path.abspath(args.evals_root)
    if not os.path.isdir(evals_root):
        print(f"[run_all_metrics] evals_root not found: {evals_root}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(evals_root, "metrics")
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for dim in args.dims:
        if dim not in DIMENSIONS:
            print(f"[run_all_metrics] Unknown dim {dim}, skip.", file=sys.stderr)
            continue
        name, fn = DIMENSIONS[dim]
        kwargs = {"evals_root": evals_root}
        if name == "loop_closure":
            kwargs["dataset_base"] = args.dataset
        if name == "identity_preservation":
            kwargs["use_clip"] = args.use_clip
        print(f"[run_all_metrics] Running {name} ...", file=sys.stderr)
        try:
            out = fn(**kwargs)
            results[name] = out
            with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            print(f"[run_all_metrics] {name} failed: {e}", file=sys.stderr)
            results[name] = {"error": str(e)}

    summary_path = os.path.join(output_dir, "all_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[run_all_metrics] Summary written to {summary_path}", file=sys.stderr)

    if args.write_csv:
        import csv
        rows = []
        for name, data in results.items():
            if "aggregate" not in data or isinstance(data.get("aggregate"), str):
                continue
            row = {"dimension": name}
            for k, v in data["aggregate"].items():
                if isinstance(v, (int, float)) and "note" not in k.lower():
                    row[k] = v
            rows.append(row)
        if rows:
            keys = list(rows[0].keys())
            csv_path = os.path.join(output_dir, "aggregate_summary.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(f"[run_all_metrics] CSV written to {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
