"""
Shared context selection for multi-chunk training/inference alignment.

- context_frames_for_next_chunk: same sampling as legacy run_replay_loop_two_chunk (replay mode).
- prev_chunk_tail: strict consecutive frames before start_frame from disk (frames/{video}/{idx:04d}.png).
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

FrameType = Any


def context_frames_for_next_chunk(frames_list: Sequence[FrameType], K: int) -> List[FrameType]:
    """Select K context frames from a finished chunk for the next chunk (replay-style).

    Order is [last_frame, ...]: last frame first (adjacent to target), then K-1 uniformly
    spaced frames from indices [0 .. n-2].

    - K==1: [last]
    - K>1: [last] + (K-1) uniform samples from [0, n-2]
    """
    n = len(frames_list)
    if n <= 0 or K <= 0:
        return []
    if K == 1:
        return [frames_list[-1]]
    n_ctx = min(K, n)
    if n_ctx == 1:
        return [frames_list[-1]]
    last = frames_list[-1]
    num_rest = n_ctx - 1
    if num_rest <= 0:
        return [last]
    if num_rest == 1:
        return [last, frames_list[0]]
    indices = [int(round(i * (n - 2) / (num_rest - 1))) for i in range(num_rest)]
    rest = [frames_list[i] for i in indices]
    return [last] + rest


def replay_context_global_indices(n_frames: int, K: int) -> List[int]:
    """Indices into frames_list matching context_frames_for_next_chunk order (for tests/debug)."""
    if n_frames <= 0 or K <= 0:
        return []
    if K == 1:
        return [n_frames - 1]
    n_ctx = min(K, n_frames)
    if n_ctx == 1:
        return [n_frames - 1]
    num_rest = n_ctx - 1
    if num_rest == 1:
        return [n_frames - 1, 0]
    indices = [int(round(i * (n_frames - 2) / (num_rest - 1))) for i in range(num_rest)]
    return [n_frames - 1] + indices


def replay_context_from_generated_frames(
    frames_list: Sequence[FrameType],
    n_ctx: int,
) -> List[FrameType]:
    """Single replay-style context selection entrypoint used by callsites.

    Keep legacy semantics:
    - n_ctx > 0: replay sampling rule (last + uniform historical)
    - n_ctx <= 0: fallback to last frame only
    """
    n_ctx = int(n_ctx)
    if n_ctx > 0:
        return context_frames_for_next_chunk(frames_list, n_ctx)
    return [frames_list[-1]]


def prev_chunk_tail_global_indices(start_frame: int, N: int, *, nearest_first: bool = False) -> Optional[List[int]]:
    """Strict consecutive globals with configurable order.

    - nearest_first=False: [start_frame - N, ..., start_frame - 1] (oldest -> newest)
    - nearest_first=True:  [start_frame - 1, ..., start_frame - N] (newest -> oldest)
    None if start_frame < N.
    """
    if N <= 0:
        return []
    if start_frame < N:
        return None
    if nearest_first:
        return list(range(int(start_frame) - 1, int(start_frame) - N - 1, -1))
    return list(range(int(start_frame) - N, int(start_frame)))


def load_prev_chunk_tail_from_disk(
    dataset_base_path: str,
    video_name: str,
    start_frame: int,
    N: int,
    *,
    nearest_first: bool = False,
) -> Tuple[Optional[List[Any]], Optional[List[int]]]:
    """Load N frames before start_frame in configured order."""
    from PIL import Image

    idxs = prev_chunk_tail_global_indices(int(start_frame), int(N), nearest_first=nearest_first)
    if idxs is None:
        return None, None
    if not idxs:
        return [], []
    vn = str(video_name)
    if vn.endswith((".mp4", ".avi")):
        vn = os.path.splitext(vn)[0]
    frames_root = os.path.join(dataset_base_path, "frames", vn)
    out: List[Any] = []
    for idx in idxs:
        path = os.path.join(frames_root, f"{int(idx):04d}.png")
        if not os.path.isfile(path):
            return None, None
        try:
            out.append(Image.open(path).convert("RGB"))
        except Exception:
            return None, None
    return out, idxs


def synthetic_replay_context_from_segment(
    video_frames: Sequence[FrameType],
    chunk_frames: int,
    K: int,
) -> Optional[List[FrameType]]:
    """Use first `chunk_frames` of video_frames as virtual chunk1; context for 'chunk2' via replay rule.

    Requires len(video_frames) >= chunk_frames. Returns None otherwise.
    """
    if len(video_frames) < chunk_frames or K <= 0:
        return None
    chunk1 = list(video_frames[:chunk_frames])
    return context_frames_for_next_chunk(chunk1, K)


def replay_context_actions_from_segment_actions(
    actions: Sequence[Sequence[float]],
    n_frames: int,
    K: int,
) -> Optional[List[List[float]]]:
    """Align RT/action rows with context_frames_for_next_chunk order (same indices as replay_context_global_indices)."""
    idxs = replay_context_global_indices(int(n_frames), int(K))
    if not idxs:
        return []
    need_max = max(idxs)
    if need_max >= len(actions):
        return None
    return [list(actions[i]) for i in idxs]


def load_prev_chunk_tail_rt_actions(
    dataset_base_path: str,
    video_name: str,
    start_frame: int,
    N: int,
    *,
    use_rt_relative: bool = True,
    nearest_first: bool = False,
) -> Tuple[Optional[List[List[float]]], Optional[List[int]]]:
    """Load RT poses in configured order, relative to first context frame."""
    idxs = prev_chunk_tail_global_indices(int(start_frame), int(N), nearest_first=nearest_first)
    if idxs is None:
        return None, None
    if not idxs:
        return [], []
    try:
        from .fov_retrieval import load_camera_poses_batch
        from .rt_utils import pose_to_rt, convert_rt_to_relative
    except ImportError:
        from fov_retrieval import load_camera_poses_batch
        from rt_utils import pose_to_rt, convert_rt_to_relative

    vn = str(video_name)
    if vn.endswith((".mp4", ".avi")):
        vn = os.path.splitext(vn)[0]
    json_file = os.path.join(dataset_base_path, "jsons", f"{vn}.json")
    if not os.path.isfile(json_file):
        return None, None
    poses = load_camera_poses_batch(json_file, idxs)
    rt_list = [pose_to_rt(p) if p else None for p in poses]
    if not rt_list or any(r is None for r in rt_list):
        return None, None
    ref_rt = rt_list[0]
    if use_rt_relative:
        out = convert_rt_to_relative(rt_list, ref_rt)
    else:
        out = [list(r) for r in rt_list]
    return out, idxs
