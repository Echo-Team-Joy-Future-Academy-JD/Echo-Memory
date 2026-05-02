"""
Shared utilities for eval_metrics: discover evals_ep0 outputs and load video frames.
"""
from __future__ import annotations

import os
from typing import List, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import numpy as np


def discover_evals_videos(evals_root: str, pattern: str = "*_gen_only.mp4") -> List[Tuple[str, str]]:
    """
    Discover all generated-only MP4s under evals_ep0 structure.
    Returns list of (relative_path, absolute_path) for each video.
    """
    out: List[Tuple[str, str]] = []
    evals_root = os.path.abspath(evals_root)
    for root, _dirs, files in os.walk(evals_root):
        for f in files:
            if f.endswith("_gen_only.mp4") or (pattern != "*_gen_only.mp4" and f.endswith(".mp4")):
                absp = os.path.join(root, f)
                rel = os.path.relpath(absp, evals_root)
                out.append((rel, absp))
    return sorted(out, key=lambda x: x[0])


def discover_loop_closure_videos(evals_root: str) -> List[Tuple[str, str]]:
    """Discover MP4s under 1_loop_4chunk and 3_multi_ctx_4chunk for loop closure (prefer gen_only)."""
    out: List[Tuple[str, str]] = []
    for sub in ("1_loop_4chunk", "3_multi_ctx_4chunk"):
        d = os.path.join(evals_root, sub)
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.endswith("_gen_only.mp4"):
                    absp = os.path.join(root, f)
                    rel = os.path.relpath(absp, evals_root)
                    out.append((rel, absp))
    return sorted(out, key=lambda x: x[0])


def load_video_frames(path: str, max_frames: int | None = None) -> np.ndarray:
    """
    Load video as array of frames (RGB, uint8).
    Returns (N, H, W, 3). If max_frames set, stop after that many frames.
    """
    if not HAS_CV2:
        raise RuntimeError("opencv-python is required for video loading (pip install opencv-python)")
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def load_video_frames_pil(path: str, max_frames: int | None = None):
    """Load video as list of PIL Images (for CLIP etc.)."""
    from PIL import Image
    arr = load_video_frames(path, max_frames=max_frames)
    if arr.size == 0:
        return []
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
