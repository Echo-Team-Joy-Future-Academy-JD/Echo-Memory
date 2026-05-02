#!/usr/bin/env python3
"""Aggregate geometry diagnostics from geometry_diagnostics.json files."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root directory containing geometry_diagnostics.json")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    rows: List[Dict[str, Any]] = []
    vals: List[float] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "geometry_diagnostics.json" not in filenames:
            continue
        p = os.path.join(dirpath, "geometry_diagnostics.json")
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            d["rel_dir"] = os.path.relpath(dirpath, root)
            rows.append(d)
            v = d.get("ghosting_proxy_color_var")
            if v is not None:
                vals.append(float(v))
        except Exception as e:
            rows.append({"rel_dir": os.path.relpath(dirpath, root), "error": str(e)})

    out = {"root": root, "num_runs": len(rows), "mean_ghosting_proxy_color_var": _mean(vals), "per_run": rows}
    outp = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[aggregate_geometry_consistency] runs={len(rows)} mean={out['mean_ghosting_proxy_color_var']} -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
