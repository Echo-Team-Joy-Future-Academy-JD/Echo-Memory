#!/usr/bin/env python3
"""Aggregate closure_first_vs_last_mse from combo revisit outputs (revisit_closure_metrics.json)."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. .../combo_revisit_in_domain")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    rows: List[Dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "revisit_closure_metrics.json" not in filenames:
            continue
        p = os.path.join(dirpath, "revisit_closure_metrics.json")
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows.append({"rel": os.path.relpath(dirpath, root), **data})
        except Exception as e:
            rows.append({"rel": os.path.relpath(dirpath, root), "error": str(e)})

    mses = [float(r["closure_first_vs_last_mse"]) for r in rows if r.get("closure_first_vs_last_mse") is not None]
    mean_m = (float(sum(mses) / len(mses)) if mses else None)
    summary = {
        "root": root,
        "num_runs": len(rows),
        "mean_closure_mse": mean_m,
        "per_run": rows,
    }
    outp = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate_combo_closure_metrics] runs={len(rows)} mean_closure_mse={summary['mean_closure_mse']} -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
