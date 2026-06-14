#!/usr/bin/env python3
"""
Lightweight DATASET sanity check before heavy eval (no torch / no train.py).
Verifies jsons + frames dirs and that pose keys exist for the replay frame range.
Exit 0 if OK, 1 otherwise; prints absolute paths and first failure reason.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _poses_dict(data: dict) -> dict:
    if "CineCameraActor" in data:
        return data["CineCameraActor"]
    return data if isinstance(data, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_frames", type=int, default=81)
    args = ap.parse_args()

    ds = _abs(args.dataset)
    jsons = os.path.join(ds, "jsons")
    frames_root = os.path.join(ds, "frames")
    vn = str(args.video).replace(".mp4", "").replace(".avi", "").strip()
    jpath = os.path.join(jsons, f"{vn}.json")
    fdir = os.path.join(frames_root, vn)

    print(f"[check_dataset_gt] DATASET={ds}")
    print(f"[check_dataset_gt] json={jpath}")
    print(f"[check_dataset_gt] frames_dir={fdir}")

    if not os.path.isdir(ds):
        print("[check_dataset_gt] FAIL: DATASET is not a directory", file=sys.stderr)
        return 1
    if not os.path.isdir(jsons):
        print("[check_dataset_gt] FAIL: missing jsons/", file=sys.stderr)
        return 1
    if not os.path.isfile(jpath):
        print("[check_dataset_gt] FAIL: missing video json", file=sys.stderr)
        return 1
    if not os.path.isdir(fdir):
        print("[check_dataset_gt] WARN: frames subdir missing (PNG compare may fail)", file=sys.stderr)

    try:
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[check_dataset_gt] FAIL: cannot read json: {e}", file=sys.stderr)
        return 1

    poses = _poses_dict(data)
    need = []
    for ch in range(args.num_chunks):
        seg = args.start_frame + ch * args.chunk_frames
        for i in range(args.chunk_frames):
            need.append(seg + i)

    missing_pose = []
    for fi in need:
        k = str(fi)
        if k not in poses:
            missing_pose.append(fi)
            if len(missing_pose) >= 5:
                break

    if missing_pose:
        print(
            f"[check_dataset_gt] FAIL: missing pose keys for frames (showing up to 5): {missing_pose}",
            file=sys.stderr,
        )
        def _knum(x):
            try:
                return int(x)
            except (TypeError, ValueError):
                return 0

        sample_keys = sorted(poses.keys(), key=_knum)[:8]
        print(f"[check_dataset_gt] sample pose keys: {sample_keys}", file=sys.stderr)
        return 1

    # Optional: same 12-dim path as replay (numpy via fov_retrieval; no torch)
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        # .../eval/v2/basic -> repo root
        _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_here))))
        if _repo not in sys.path:
            sys.path.insert(0, _repo)
        from src.model_training.fov_retrieval import load_camera_pose, pose_to_rt

        for fi in need[: min(3, len(need))]:
            pose = load_camera_pose(jpath, fi)
            rt = pose_to_rt(pose, constrain_to_xy=True) if pose else None
            if rt is None or len(rt) < 12:
                print(f"[check_dataset_gt] FAIL: pose_to_rt None for frame {fi}", file=sys.stderr)
                return 1
    except Exception as e:
        print(f"[check_dataset_gt] WARN: RT parse spot-check skipped: {e}", file=sys.stderr)

    print("[check_dataset_gt] OK: paths and pose keys cover the replay range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
