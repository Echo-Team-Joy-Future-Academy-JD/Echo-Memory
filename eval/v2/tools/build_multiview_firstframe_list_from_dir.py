#!/usr/bin/env python3
"""
Build a txt list for run_multiview_revisit_from_firstframes.py from a directory of images.

Each line: absolute_path[\toptional_prompt]
If --prompt is set, every line gets the same prompt after a tab (overrides per-file default).
"""
from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True, help="e.g. .../opendomain2_revisit")
    ap.add_argument("--output", required=True, help="txt path written for MULTIVIEW_FIRSTFRAME_LIST")
    ap.add_argument(
        "--prompt",
        default="",
        help="If non-empty, append same tab-prompt to every line; else path-only (runner defaults to 'A scene.')",
    )
    ap.add_argument(
        "--extensions",
        default=".png,.jpg,.jpeg,.webp",
        help="Comma-separated suffixes (lowercase)",
    )
    args = ap.parse_args()
    root = os.path.abspath(args.image_dir)
    if not os.path.isdir(root):
        print(f"[build_multiview_list] not a directory: {root}", flush=True)
        return 1
    exts = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}
    names = []
    for n in sorted(os.listdir(root)):
        low = n.lower()
        if any(low.endswith(e) for e in exts):
            p = os.path.join(root, n)
            if os.path.isfile(p):
                names.append(p)
    if not names:
        print(f"[build_multiview_list] no images under {root}", flush=True)
        return 1
    outp = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    prompt = (args.prompt or "").strip()
    with open(outp, "w", encoding="utf-8") as f:
        for p in names:
            if prompt:
                f.write(f"{p}\t{prompt}\n")
            else:
                f.write(f"{p}\n")
    print(f"[build_multiview_list] wrote {len(names)} lines -> {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
