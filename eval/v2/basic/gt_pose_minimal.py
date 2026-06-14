#!/usr/bin/env python3
"""
GT trajectory helpers without importing torch / diffsynth / run_replay_loop_two_chunk.
Used by run_basic_replay_gt.sh to resolve VIDEO_NAME=AUTO without pulling train.py.
Logic must match run_replay_loop_two_chunk.build_gt_trajectory_actions.
"""
from __future__ import annotations

import os
import sys


def _repo_root_from_here() -> str:
    _here = os.path.dirname(os.path.abspath(__file__))
    # .../eval/v2/basic -> repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_here))))


def _ensure_repo_path() -> None:
    r = _repo_root_from_here()
    if r not in sys.path:
        sys.path.insert(0, r)


def load_pose_rt(json_file: str, frame_idx: int):
    _ensure_repo_path()
    from src.model_training.fov_retrieval import load_camera_pose, pose_to_rt
    pose = load_camera_pose(json_file, int(frame_idx))
    if pose is None:
        return None
    return pose_to_rt(pose, constrain_to_xy=True)


def get_relative_rt(rt, ref_rt):
    _ensure_repo_path()
    from src.model_training.fov_retrieval import convert_rt_to_relative
    if rt is None or ref_rt is None or len(rt) < 12 or len(ref_rt) < 12:
        return None
    out = convert_rt_to_relative([rt], ref_rt)
    return out[0] if out else None


def build_gt_trajectory_actions(dataset_base, video_name, start_frame, chunk_frames, json_file=None):
    if json_file is None:
        json_file = os.path.join(dataset_base, "jsons", f"{video_name}.json")
    if not os.path.isfile(json_file):
        return None
    try:
        rt_list = [load_pose_rt(json_file, start_frame + i) for i in range(chunk_frames)]
        if not rt_list or any(r is None or len(r) < 12 for r in rt_list):
            return None
        ref_rt = rt_list[0]
        rel_actions = {str(i): get_relative_rt(rt_list[i], ref_rt) for i in range(chunk_frames)}
        if any(v is None for v in rel_actions.values()):
            return None
        return rel_actions
    except Exception:
        return None
