#!/usr/bin/env python3
"""Fail-fast checks for eval_v2 static_consistency (avoid hours of GPU work then argv errors)."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import List

# Required chunk json names (must match run_combo_revisit_fixed_first.py)
COMBO_CHUNK_FILES = (
    "chunk0_rotate_left_45.json",
    "chunk1_translate_forward.json",
    "chunk2_rotate_right_45.json",
    "chunk3_translate_backward.json",
)

WAN_FILES = (
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
)


def _parse_remainder_like_multiview(mem_args: List[str]) -> List[str]:
    """Mirror run_multiview_revisit_from_firstframes.py --extra_args REMAINDER handling."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--firstframe_list", required=True)
    ap.add_argument("--action_combo_dir", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[])
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("# preflight\n")
        list_path = f.name
    out_root = tempfile.mkdtemp(prefix="multiview_preflight_")
    try:
        argv = [
            "_",
            "--ckpt",
            os.path.abspath(__file__),
            "--firstframe_list",
            list_path,
            "--action_combo_dir",
            out_root,
            "--output_root",
            out_root,
            "--extra_args",
        ] + list(mem_args)
        ns = ap.parse_args(argv[1:])
        extra = list(ns.extra_args or [])
        if extra and extra[0] == "--":
            extra = extra[1:]
        return extra
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


def cmd_remainder_mem_args(mem_args: List[str]) -> int:
    try:
        got = _parse_remainder_like_multiview(mem_args)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[verify_static_eval_prereqs] FATAL: multiview-style REMAINDER parse failed: {e}", file=sys.stderr)
        return 1
    if got != mem_args:
        print(
            f"[verify_static_eval_prereqs] FATAL: MEM_ARGS round-trip mismatch:\n  expect={mem_args!r}\n  got={got!r}",
            file=sys.stderr,
        )
        return 1
    print(f"[verify_static_eval_prereqs] OK remainder-mem-args ({len(mem_args)} tokens)")
    return 0


def cmd_wan_base(_: List[str]) -> int:
    wan_base = os.environ.get("WAN_BASE_MODEL", "")
    if not wan_base:
        print("[verify_static_eval_prereqs] FATAL: WAN_BASE_MODEL is not set", file=sys.stderr)
        return 1
    missing = []
    for name in WAN_FILES:
        p = os.path.join(wan_base, name)
        if not os.path.isfile(p):
            missing.append(p)
    if missing:
        print("[verify_static_eval_prereqs] FATAL: missing Wan2.1 base files:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print("[verify_static_eval_prereqs] hint: set WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B", file=sys.stderr)
        return 1
    print(f"[verify_static_eval_prereqs] OK wan-base ({wan_base})")
    return 0


def _load_firstframe_paths(list_path: str) -> List[str]:
    ext = os.path.splitext(list_path)[1].lower()
    paths: List[str] = []
    if ext == ".jsonl":
        import json

        with open(list_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                d = json.loads(ln)
                paths.append(str(d.get("first_frame_image") or d.get("image") or ""))
        return [p for p in paths if p]
    if ext == ".csv":
        import csv

        with open(list_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                paths.append(str(row.get("first_frame_image") or row.get("image") or ""))
        return [p for p in paths if p]
    with open(list_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            paths.append(ln.split("\t", 1)[0].strip())
    return paths


def cmd_multiview_preflight(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Validate paths before multiview GPU run")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--firstframe-list", required=True, dest="firstframe_list")
    ap.add_argument("--action-combo-dir", required=True, dest="action_combo_dir")
    ap.add_argument("--runner", default="", help="default: run_combo_revisit_fixed_first.py next to multiview script")
    ap.add_argument("mem_args", nargs=argparse.REMAINDER, default=[])
    ns = ap.parse_args(argv)
    mem = list(ns.mem_args or [])
    if mem and mem[0] == "--":
        mem = mem[1:]

    if not os.path.isfile(ns.ckpt):
        print(f"[verify_static_eval_prereqs] FATAL: CKPT not a file: {ns.ckpt}", file=sys.stderr)
        return 1
    if not os.path.isfile(ns.firstframe_list):
        print(f"[verify_static_eval_prereqs] FATAL: firstframe list missing: {ns.firstframe_list}", file=sys.stderr)
        return 1
    if not os.path.isdir(ns.action_combo_dir):
        print(f"[verify_static_eval_prereqs] FATAL: action combo dir missing: {ns.action_combo_dir}", file=sys.stderr)
        return 1
    for fn in COMBO_CHUNK_FILES:
        p = os.path.join(ns.action_combo_dir, fn)
        if not os.path.isfile(p):
            print(f"[verify_static_eval_prereqs] FATAL: missing combo action: {p}", file=sys.stderr)
            return 1

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runner = ns.runner or os.path.join(here, "static", "run_combo_revisit_fixed_first.py")
    if not os.path.isfile(runner):
        print(f"[verify_static_eval_prereqs] FATAL: runner missing: {runner}", file=sys.stderr)
        return 1

    try:
        paths = _load_firstframe_paths(ns.firstframe_list)
    except Exception as e:
        print(f"[verify_static_eval_prereqs] FATAL: cannot read firstframe list: {e}", file=sys.stderr)
        return 1
    if not paths:
        print("[verify_static_eval_prereqs] FATAL: firstframe list has no image paths", file=sys.stderr)
        return 1
    missing_img = [p for p in paths if not os.path.isfile(p)]
    if missing_img:
        print(f"[verify_static_eval_prereqs] FATAL: missing first-frame image(s), e.g.: {missing_img[:3]}", file=sys.stderr)
        return 1

    if cmd_remainder_mem_args(mem) != 0:
        return 1

    print(
        f"[verify_static_eval_prereqs] OK multiview-preflight "
        f"(views={len(paths)}, mem_tokens={len(mem)}, combo={ns.action_combo_dir})"
    )
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_static_eval_prereqs.py {remainder-mem-args|wan-base|multiview-preflight} ...", file=sys.stderr)
        return 2
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd == "remainder-mem-args":
        return cmd_remainder_mem_args(rest)
    if cmd == "wan-base":
        return cmd_wan_base(rest)
    if cmd == "multiview-preflight":
        return cmd_multiview_preflight(rest)
    print(f"[verify_static_eval_prereqs] unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
