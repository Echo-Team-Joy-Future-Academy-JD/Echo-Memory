#!/usr/bin/env python3
"""
Build per-chunk action JSONs for a composite action sequence, for static-consistency revisit tests.

First version supports the requested pattern:
  rotate_left_45 -> translate_forward -> rotate_right_45 -> translate_backward

Notes:
- Each JSON is a dict: frame_index(str) -> RT list length 12: [tx,ty,tz,R11..R33] (row-major 3x3).
- Actions are *relative to each chunk's first frame* (matching training / existing eval conventions).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List, Tuple


def _load_json(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_rotation_yaw_chunk(yaw_total_deg: float, clockwise: bool, chunk_frames: int) -> Dict[str, List[float]]:
    """Linear yaw 0→±yaw_total_deg in chunk (Z-only RT), same convention as run_replay_loop_two_chunk.build_action_chunk."""
    denom = max(1, chunk_frames - 1)
    sign = -1.0 if clockwise else 1.0
    out: Dict[str, List[float]] = {}
    for i in range(chunk_frames):
        yaw = sign * (i / denom) * float(yaw_total_deg)
        rad = math.radians(yaw)
        c, s = math.cos(rad), math.sin(rad)
        r_flat = [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
        out[str(i)] = [0.0, 0.0, 0.0] + r_flat
    return out


def build_translation_only(direction: str, translation_delta: float, chunk_frames: int) -> Dict[str, List[float]]:
    """Mirror run_replay_loop_two_chunk.build_action_translation_only without importing heavy deps."""
    identity_rot = [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    denom = max(1, chunk_frames - 1)
    out: Dict[str, List[float]] = {}
    for i in range(chunk_frames):
        t = (i / denom) * translation_delta
        if direction == "forward":
            tx, ty, tz = 0.0, t, 0.0
        elif direction == "backward":
            tx, ty, tz = 0.0, -t, 0.0
        elif direction == "left":
            tx, ty, tz = -t, 0.0, 0.0
        elif direction == "right":
            tx, ty, tz = t, 0.0, 0.0
        else:
            tx, ty, tz = 0.0, 0.0, 0.0
        out[str(i)] = [tx, ty, tz] + identity_rot
    return out


def save_action_json(actions: Dict[str, List[float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(actions, f, indent=2)


def build_combo(
    exp_dir: str,
    out_dir: str,
    chunk_frames: int = 81,
    translation_delta: float = 0.1,
) -> Tuple[str, str, str, str]:
    """
    Create 4 chunk action jsons under out_dir:
      chunk0_rotate_left_45.json
      chunk1_translate_forward.json
      chunk2_rotate_right_45.json
      chunk3_translate_backward.json
    """
    left_path = os.path.join(exp_dir, "action_rotation_left_45.json")
    right_path = os.path.join(exp_dir, "action_rotation_right_45.json")
    if not (os.path.isfile(left_path) and os.path.isfile(right_path)):
        raise FileNotFoundError(f"Missing rotation jsons under exp_dir: {left_path} / {right_path}")

    rot_left = _load_json(left_path)
    rot_right = _load_json(right_path)

    # Sanity: ensure expected frame keys exist; if not, allow but warn via truncation
    def _trim(d: Dict[str, List[float]]) -> Dict[str, List[float]]:
        return {str(i): d[str(i)] for i in range(chunk_frames) if str(i) in d}

    rot_left = _trim(rot_left)
    rot_right = _trim(rot_right)

    trans_fwd = build_translation_only("forward", translation_delta, chunk_frames)
    trans_bwd = build_translation_only("backward", translation_delta, chunk_frames)

    p0 = os.path.join(out_dir, "chunk0_rotate_left_45.json")
    p1 = os.path.join(out_dir, "chunk1_translate_forward.json")
    p2 = os.path.join(out_dir, "chunk2_rotate_right_45.json")
    p3 = os.path.join(out_dir, "chunk3_translate_backward.json")
    save_action_json(rot_left, p0)
    save_action_json(trans_fwd, p1)
    save_action_json(rot_right, p2)
    save_action_json(trans_bwd, p3)
    return p0, p1, p2, p3


def build_random_symmetric_closed_loop(
    out_dir: str,
    chunk_frames: int,
    rng: random.Random,
    yaw_min: float = 20.0,
    yaw_max: float = 55.0,
    translation_min: float = 0.05,
    translation_max: float = 0.18,
) -> Tuple[List[str], Dict]:
    """
    Symmetric motion that composes to ~identity in the training RT convention:
      chunk0: CCW yaw (left) 0→+Y
      chunk1: forward +d along Y
      chunk2: CW yaw (right) 0→-Y  (cancels chunk0 in world yaw if chunk frames align)
      chunk3: backward -d along Y (cancels chunk1 translation)

    Same filenames as fixed 45° combo for drop-in use with run_combo_revisit_fixed_first.py.
    """
    yaw = rng.uniform(float(yaw_min), float(yaw_max))
    d = rng.uniform(float(translation_min), float(translation_max))
    rot_left = build_rotation_yaw_chunk(yaw, clockwise=False, chunk_frames=chunk_frames)
    rot_right = build_rotation_yaw_chunk(yaw, clockwise=True, chunk_frames=chunk_frames)
    trans_fwd = build_translation_only("forward", d, chunk_frames)
    trans_bwd = build_translation_only("backward", d, chunk_frames)
    p0 = os.path.join(out_dir, "chunk0_rotate_left_45.json")
    p1 = os.path.join(out_dir, "chunk1_translate_forward.json")
    p2 = os.path.join(out_dir, "chunk2_rotate_right_45.json")
    p3 = os.path.join(out_dir, "chunk3_translate_backward.json")
    save_action_json(rot_left, p0)
    save_action_json(trans_fwd, p1)
    save_action_json(rot_right, p2)
    save_action_json(trans_bwd, p3)
    meta = {
        "pattern": "symmetric_closed_loop_random",
        "yaw_deg": yaw,
        "translation_delta": d,
        "chunk_frames": chunk_frames,
        "chunks": [
            {"file": os.path.basename(p0), "desc": "ccw_yaw_0_to_+yaw"},
            {"file": os.path.basename(p1), "desc": "forward_d"},
            {"file": os.path.basename(p2), "desc": "cw_yaw_0_to_-yaw"},
            {"file": os.path.basename(p3), "desc": "backward_d"},
        ],
    }
    return [p0, p1, p2, p3], meta


def main() -> None:
    p = argparse.ArgumentParser(description="Build composite action JSONs for revisit tests")
    p.add_argument("--exp_dir", type=str, default="", help="exp dir with action_rotation_*.json (fixed 45° mode)")
    p.add_argument("--out_dir", type=str, required=True, help="output directory to write per-chunk action jsons")
    p.add_argument("--chunk_frames", type=int, default=81)
    p.add_argument("--translation_delta", type=float, default=0.1)
    p.add_argument(
        "--random_symmetric",
        action="store_true",
        help="Random yaw/translation magnitudes with symmetric closed-loop (left→fwd→right→back); ignores exp_dir rotations",
    )
    p.add_argument("--combo_seed", type=int, default=42, help="RNG seed for --random_symmetric")
    p.add_argument("--yaw_min", type=float, default=20.0)
    p.add_argument("--yaw_max", type=float, default=55.0)
    p.add_argument("--translation_min", type=float, default=0.05)
    p.add_argument("--translation_max", type=float, default=0.18)
    args = p.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.random_symmetric:
        rng = random.Random(int(args.combo_seed))
        paths, meta = build_random_symmetric_closed_loop(
            out_dir,
            chunk_frames=args.chunk_frames,
            rng=rng,
            yaw_min=args.yaw_min,
            yaw_max=args.yaw_max,
            translation_min=args.translation_min,
            translation_max=args.translation_max,
        )
        with open(os.path.join(out_dir, "combo_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("\n".join(paths))
        return

    if not args.exp_dir:
        raise SystemExit("build_action_combo: need --exp_dir unless --random_symmetric")
    paths = build_combo(
        exp_dir=os.path.abspath(args.exp_dir),
        out_dir=out_dir,
        chunk_frames=args.chunk_frames,
        translation_delta=args.translation_delta,
    )
    print("\n".join(paths))


if __name__ == "__main__":
    main()

