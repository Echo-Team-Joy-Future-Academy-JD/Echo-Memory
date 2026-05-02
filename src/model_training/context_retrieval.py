"""
Advanced context retrieval for Context-as-Memory (CAM).
Provides retrieval methods beyond FOV: latent similarity, optional learned/diversity.
Interface compatible with retrieve_simple_context_frames / retrieve_fov_context_frames.
"""

import os
import random
import sys
from typing import Dict, List, Optional, Tuple, Callable

try:
    from .fov_training_integration import retrieve_simple_context_frames
    from .fov_retrieval import load_overlap_frames, load_poses_dict
    from .rt_utils import pose_to_rt, rt_to_pose, convert_rt_to_relative
except ImportError:
    from fov_training_integration import retrieve_simple_context_frames
    from fov_retrieval import load_overlap_frames, load_poses_dict
    try:
        from rt_utils import pose_to_rt, rt_to_pose, convert_rt_to_relative
    except ImportError:
        pose_to_rt = rt_to_pose = convert_rt_to_relative = None  # type: ignore


_WARN_ONCE_KEYS = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _WARN_ONCE_KEYS:
        return
    _WARN_ONCE_KEYS.add(key)
    print(f"[context_retrieval] WARN: {msg}", file=sys.stderr, flush=True)


def _load_latent(latent_dir: str, video_name: str, frame_idx: int):
    """Load a single-frame latent. Expects latent_dir/video_name/{frame_idx:04d}.pt or .pt with key 'latent' or raw tensor."""
    base = os.path.join(latent_dir, video_name)
    for fmt in (f"{frame_idx:04d}.pt", f"{frame_idx}.pt"):
        path = os.path.join(base, fmt)
        if os.path.isfile(path):
            try:
                import torch
                try:
                    x = torch.load(path, map_location="cpu", weights_only=True)
                except TypeError:
                    x = torch.load(path, map_location="cpu")
                if isinstance(x, dict) and "latent" in x:
                    z = x["latent"]
                else:
                    z = x
                if hasattr(z, "shape"):
                    # (C, 1, H, W) or (C, H, W) -> flatten for similarity
                    return z.flatten()
                return None
            except Exception:
                pass
    return None


def latent_sim_rank(
    video_name: str,
    first_frame_idx: int,
    overlapping_indices: List[int],
    dataset_base_path: str,
    top_k: int,
    latent_dir: Optional[str] = None,
    use_cosine: bool = True,
) -> List[int]:
    """
    Rank overlapping frame indices by latent similarity to the first (reference) frame.
    If latent_dir is None or latents are missing, falls back to random sample.
    Expects per-frame latents under latent_dir/video_name/{frame_idx:04d}.pt (or {frame_idx}.pt).
    """
    if not overlapping_indices or top_k <= 0:
        return []
    if latent_dir is None or not os.path.isdir(latent_dir):
        return random.sample(overlapping_indices, min(top_k, len(overlapping_indices)))

    ref = _load_latent(latent_dir, video_name, first_frame_idx)
    if ref is None:
        return random.sample(overlapping_indices, min(top_k, len(overlapping_indices)))

    import torch
    ref = ref.float().unsqueeze(0)
    scores = []
    for idx in overlapping_indices:
        cand = _load_latent(latent_dir, video_name, idx)
        if cand is None:
            continue
        cand = cand.float().unsqueeze(0)
        if use_cosine:
            sim = torch.nn.functional.cosine_similarity(ref, cand, dim=1).item()
        else:
            sim = -((ref - cand) ** 2).sum().item()
        scores.append((idx, sim))
    if not scores:
        return random.sample(overlapping_indices, min(top_k, len(overlapping_indices)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scores[:top_k]]


def retrieve_context_frames_advanced(
    data: Dict,
    dataset_base_path: str,
    top_k: int = 4,
    drop_overlap_probability: float = 0.1,
    use_rt_relative: bool = False,
    retrieval_method: str = "fov",
    latent_retrieval_dir: Optional[str] = None,
    strict_overlap_labels: bool = False,
) -> Tuple[List, List, List[int], int, str, str]:
    """
    Retrieve context frames with pluggable retrieval method.
    Interface matches retrieve_fov_context_frames return:
      (context_frames, context_actions, context_indices, cur_idx, video_name, source).

    retrieval_method:
      - "fov": use existing FOV/overlap_labels logic (random or RT-scored from overlap).
      - "latent_sim": rank overlap candidates by latent similarity to first frame (requires latent_retrieval_dir).
    When latent_sim is used but latent_retrieval_dir is missing or latents absent, falls back to FOV behavior.
    """
    if retrieval_method == "latent_sim" and not latent_retrieval_dir:
        _warn_once(
            "latent_sim_missing_dir",
            "retrieval_method=latent_sim but latent_retrieval_dir is not set; fallback to FOV retrieval.",
        )
    if retrieval_method == "fov" or (retrieval_method == "latent_sim" and not latent_retrieval_dir):
        return retrieve_simple_context_frames(
            data=data,
            dataset_base_path=dataset_base_path,
            top_k=top_k,
            drop_overlap_probability=drop_overlap_probability,
            use_rt_relative=use_rt_relative,
        )
    if retrieval_method == "latent_sim" and latent_retrieval_dir and not os.path.isdir(latent_retrieval_dir):
        _warn_once(
            "latent_sim_bad_dir",
            f"latent_retrieval_dir not found: {latent_retrieval_dir}; latent_sim will degrade to random-overlap selection.",
        )

    # latent_sim: we need to inject a custom ranking into the flow. We do a minimal duplicate of the
    # overlap selection step then reuse the rest via a wrapper around retrieve_simple_context_frames
    # by passing a custom rank function. Since retrieve_simple_context_frames doesn't support that yet,
    # we implement a full path here that mirrors it but uses latent_sim_rank for overlap selection.
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor, as_completed

    video_frames = data.get("video", [])
    start_frame = data.get("start_frame", 0)
    end_frame = data.get("end_frame", None)
    if "frame_idx" in data:
        current_frame_idx = data.get("frame_idx")
    else:
        current_frame_idx = (start_frame + end_frame) // 2 if end_frame is not None else (len(video_frames) // 2 if video_frames else 0)
    first_frame_idx = start_frame
    video_name = data.get("video_name", "")
    if not video_name and "video_path" in data:
        video_name = os.path.basename(data["video_path"]).replace(".mp4", "").replace(".avi", "")
    elif not video_name and "file_path" in data:
        video_name = os.path.basename(data["file_path"]).replace(".mp4", "").replace(".avi", "")

    frames_dir = os.path.join(dataset_base_path, "frames", video_name)
    json_file = os.path.join(dataset_base_path, "jsons", f"{video_name}.json")
    poses_dict = load_poses_dict(json_file) if os.path.isfile(json_file) else {}
    first_frame_pose = poses_dict.get(str(first_frame_idx))
    first_frame_pose_rt = pose_to_rt(first_frame_pose) if first_frame_pose is not None and pose_to_rt else None

    context_frames: List[Image.Image] = []
    context_actions: List = []
    context_indices: List[int] = []
    source = "none"

    def _append_pose(frame_idx: int):
        pose = poses_dict.get(str(frame_idx))
        if pose is not None and pose_to_rt and use_rt_relative and first_frame_pose_rt is not None:
            rt = pose_to_rt(pose)
            if rt is not None and convert_rt_to_relative:
                rel = convert_rt_to_relative([rt], first_frame_pose_rt)
                context_actions.append(rel[0] if rel else [0.0] * 12)
            else:
                context_actions.append([0.0] * 12)
        elif pose is not None and pose_to_rt:
            rt = pose_to_rt(pose)
            context_actions.append(rt if rt is not None else [0.0] * 12)
        else:
            context_actions.append([0.0] * 12)

    if first_frame_pose_rt is not None and use_rt_relative:
        context_actions.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    else:
        context_actions.append(first_frame_pose_rt if first_frame_pose_rt is not None else [0.0] * 12)

    if os.path.isdir(frames_dir):
        first_path = os.path.join(frames_dir, f"{first_frame_idx:04d}.png")
        if os.path.isfile(first_path):
            try:
                context_frames.append(Image.open(first_path).convert("RGB"))
                context_indices.append(first_frame_idx)
            except Exception:
                pass
    if not context_frames and video_frames:
        if isinstance(video_frames[0], Image.Image):
            context_frames.append(video_frames[0])
            context_indices.append(start_frame)
        elif isinstance(video_frames[0], str) and os.path.isfile(video_frames[0]):
            try:
                context_frames.append(Image.open(video_frames[0]).convert("RGB"))
                context_indices.append(start_frame)
            except Exception:
                pass

    drop_overlap = random.random() < drop_overlap_probability
    if drop_overlap:
        target_total = top_k + 1
        if strict_overlap_labels and len(context_frames) < target_total:
            return [], [], [], current_frame_idx, video_name, "overlap_labels_insufficient"
        return context_frames, context_actions, context_indices, current_frame_idx, video_name, "first_frame_only_dropped"

    overlap_labels_dir = os.path.join(dataset_base_path, "overlap_labels")
    overlapping_indices = []
    if os.path.isdir(overlap_labels_dir):
        overlapping_indices = load_overlap_frames(overlap_labels_dir, video_name, current_frame_idx)
    overlapping_indices = [i for i in overlapping_indices if i != first_frame_idx]

    if not overlapping_indices:
        source = "first_frame_only"
        if strict_overlap_labels and len(context_frames) < top_k + 1:
            return [], [], [], current_frame_idx, video_name, "overlap_labels_insufficient"
        return context_frames, context_actions, context_indices, current_frame_idx, video_name, source

    sampled_overlap_indices = latent_sim_rank(
        video_name, first_frame_idx, overlapping_indices, dataset_base_path, top_k,
        latent_dir=latent_retrieval_dir, use_cosine=True,
    )

    def _load_frame(path: str):
        if os.path.isfile(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                pass
        return None

    to_load = [(idx, os.path.join(frames_dir, f"{idx:04d}.png")) for idx in sampled_overlap_indices[:top_k]]
    with ThreadPoolExecutor(max_workers=max(1, min(5, len(to_load)))) as ex:
        futures = {ex.submit(_load_frame, path): idx for idx, path in to_load}
        for fut in futures:
            idx = futures[fut]
            frame = fut.result()
            if frame is not None:
                context_frames.append(frame)
                context_indices.append(idx)
                _append_pose(idx)

    source = "latent_sim"
    if strict_overlap_labels and len(context_frames) < top_k + 1:
        return [], [], [], current_frame_idx, video_name, "overlap_labels_insufficient"
    return context_frames, context_actions, context_indices, current_frame_idx, video_name, source
