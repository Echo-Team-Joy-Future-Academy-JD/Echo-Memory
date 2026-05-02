#!/usr/bin/env python3
"""Batch multiview revisit from a first-frame list."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List

COMBO_CHUNK_FILES = (
    "chunk0_rotate_left_45.json",
    "chunk1_translate_forward.json",
    "chunk2_rotate_right_45.json",
    "chunk3_translate_backward.json",
)


def _validate_prereqs(args: argparse.Namespace, runner: str) -> None:
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"[run_multiview_revisit] CKPT 不是文件: {args.ckpt}")
    if not os.path.isfile(args.firstframe_list):
        raise FileNotFoundError(f"[run_multiview_revisit] firstframe_list 不存在: {args.firstframe_list}")
    if not os.path.isdir(args.action_combo_dir):
        raise FileNotFoundError(f"[run_multiview_revisit] action_combo_dir 不是目录: {args.action_combo_dir}")
    for fn in COMBO_CHUNK_FILES:
        p = os.path.join(args.action_combo_dir, fn)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[run_multiview_revisit] 缺少 combo 动作文件（请先 build_action_combo）: {p}")
    if not os.path.isfile(runner):
        raise FileNotFoundError(f"[run_multiview_revisit] runner 不存在: {runner}")


def _load_items(path: str) -> List[Dict[str, str]]:
    ext = os.path.splitext(path)[1].lower()
    items: List[Dict[str, str]] = []
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                d = json.loads(ln)
                items.append(
                    {
                        "view_id": str(d.get("view_id") or len(items)),
                        "first_frame_image": str(d.get("first_frame_image") or d.get("image") or ""),
                        "prompt": str(d.get("prompt") or "A scene."),
                    }
                )
        return items
    if ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                items.append(
                    {
                        "view_id": str(row.get("view_id") or len(items)),
                        "first_frame_image": str(row.get("first_frame_image") or row.get("image") or ""),
                        "prompt": str(row.get("prompt") or "A scene."),
                    }
                )
        return items
    # txt: each line -> image_path[tab prompt]
    with open(path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t", 1)
            items.append(
                {
                    "view_id": str(i),
                    "first_frame_image": parts[0],
                    "prompt": parts[1] if len(parts) > 1 else "A scene.",
                }
            )
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="Run combo revisit for a list of edited first frames")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--firstframe_list", required=True, help="txt/csv/jsonl")
    ap.add_argument("--action_combo_dir", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--runner", default=None, help="default: eval_v2/static/run_combo_revisit_fixed_first.py")
    ap.add_argument("--chunk_frames", type=int, default=81)
    ap.add_argument("--context_frames", type=int, default=1)
    ap.add_argument("--sigma_shift", type=float, default=5.0)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--camera_inject_mode",
        type=str,
        default=None,
        help="与 evals_ep0 一致；默认不传则由子进程读环境 CAMERA_INJECT_MODE",
    )
    # REMAINDER: 子进程参数若以 - 开头，nargs='*' 会被 argparse 误当作本脚本的选项而报错
    ap.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="传给 run_combo_revisit_fixed_first.py 的额外参数；须放在命令行最后（如 MEM_ARGS）",
    )
    args = ap.parse_args()
    # 允许用户写「占位」-- 与 shell 的 -- 一致
    extra = list(args.extra_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    args.extra_args = extra

    out_root = os.path.abspath(args.output_root)
    if args.runner:
        runner = os.path.abspath(args.runner)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        runner = os.path.join(here, "run_combo_revisit_fixed_first.py")

    _validate_prereqs(args, runner)

    os.makedirs(out_root, exist_ok=True)
    items = _load_items(os.path.abspath(args.firstframe_list))
    if not items:
        print("[run_multiview_revisit] no entries found")
        return 0

    missing_ff = [
        str(it.get("first_frame_image") or "")
        for it in items
        if not os.path.isfile(str(it.get("first_frame_image") or ""))
    ]
    if missing_ff:
        raise FileNotFoundError(
            f"[run_multiview_revisit] 首帧图不存在: {missing_ff[:5]}{'...' if len(missing_ff) > 5 else ''}"
        )

    summary: List[Dict[str, object]] = []
    for i, it in enumerate(items):
        view_id = str(it["view_id"])
        ff = str(it["first_frame_image"])
        prompt = str(it["prompt"])
        out_dir = os.path.join(out_root, f"view_{view_id}")
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            sys.executable,
            runner,
            "--ckpt",
            args.ckpt,
        ]
        if (args.camera_inject_mode or "").strip():
            cmd.extend(["--camera_inject_mode", str(args.camera_inject_mode).strip()])
        cmd.extend(
            [
            "--first_frame_image",
            ff,
            "--output_dir",
            out_dir,
            "--prompt",
            prompt,
            "--action_combo_dir",
            args.action_combo_dir,
            "--context_frames",
            str(args.context_frames),
            "--chunk_frames",
            str(args.chunk_frames),
            "--sigma_shift",
            str(args.sigma_shift),
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--cfg_scale",
            str(args.cfg_scale),
            "--seed",
            str(args.seed + i),
            "--no_camera_encoder_separate_t_r",
            ]
        )
        cmd.extend(list(args.extra_args))
        rc = subprocess.run(cmd, check=False).returncode
        ok = rc == 0
        summary.append(
            {
                "view_id": view_id,
                "first_frame_image": ff,
                "prompt": prompt,
                "ok": ok,
                "output_dir": out_dir,
            }
        )

    out = {
        "ckpt": args.ckpt,
        "firstframe_list": os.path.abspath(args.firstframe_list),
        "num_items": len(items),
        "summary": summary,
    }
    with open(os.path.join(out_root, "multiview_revisit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[run_multiview_revisit] wrote {out_root}/multiview_revisit_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
