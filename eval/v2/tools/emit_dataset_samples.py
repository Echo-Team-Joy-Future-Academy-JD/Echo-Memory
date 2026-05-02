#!/usr/bin/env python3
"""Emit (video_name, start_frame) lines for eval batching. Stdlib only; no torch."""
from __future__ import annotations

import argparse
import os
import random
import sys


def _collect_starts(dataset_base: str, video_name: str, min_span: int) -> list[int]:
    vd = os.path.join(dataset_base, "frames", video_name)
    if not os.path.isdir(vd):
        return []
    names = [f for f in os.listdir(vd) if f.endswith(".png")]
    indices = []
    for n in names:
        try:
            indices.append(int(os.path.splitext(n)[0]))
        except ValueError:
            continue
    if not indices:
        return []
    indices = sorted(set(indices))
    max_idx = max(indices)
    out = []
    for start in indices:
        if start + min_span - 1 <= max_idx:
            out.append(start)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Print video_name<TAB>start_frame for trajectories with enough frames")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num_samples", type=int, default=6)
    ap.add_argument("--min_frames", type=int, default=243, help="Need start + min_frames - 1 <= last index")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ds = os.path.abspath(args.dataset)
    frames_root = os.path.join(ds, "frames")
    if not os.path.isdir(frames_root):
        print(f"[emit_dataset_samples] no frames dir: {frames_root}", file=sys.stderr)
        return 1
    candidates: list[tuple[str, int]] = []
    for vn in sorted(os.listdir(frames_root)):
        vd = os.path.join(frames_root, vn)
        if not os.path.isdir(vd):
            continue
        for st in _collect_starts(ds, vn, args.min_frames):
            candidates.append((vn, st))
    if not candidates:
        print("[emit_dataset_samples] no candidates", file=sys.stderr)
        return 1
    rng = random.Random(int(args.seed))
    rng.shuffle(candidates)
    n = min(len(candidates), max(1, int(args.num_samples)))
    for vn, st in candidates[:n]:
        print(f"{vn}\t{st}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
