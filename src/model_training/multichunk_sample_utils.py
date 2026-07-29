"""
Shared multichunk sampling for training monitor and replay scripts.

Two-chunk path matches run_replay_loop_two_chunk: chunk1 with 1-frame context,
chunk2 with context_frames_for_next_chunk from chunk1 output.
No hidden state is carried across chunks (each pipe() is independent diffusion), only PIL frames + latents.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
from src.model_training.transformers_compat import patch_transformers_hybrid_cache

patch_transformers_hybrid_cache()

from diffsynth.trainers.utils import VideoDataset
from safetensors.torch import load_file as safe_load_file

from src.model_training.fov_retrieval import load_camera_poses_batch
from src.model_training.fov_retrieval import convert_rt_to_relative, pose_to_rt


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


def load_causal_prefix_context(
    dataset_base_path: str,
    video_name: str,
    start_frame: int,
    context_frames: int,
    *,
    target_num_frames: int = 81,
    target_action_stride: int = 4,
    use_rt_relative: bool = True,
) -> Tuple[
    Optional[List[Any]],
    Optional[List[List[float]]],
    Optional[List[List[float]]],
    Dict[str, Any],
]:
    """Build leak-free prefix history with one target-first RT reference."""
    start_frame = int(start_frame)
    context_frames = int(context_frames)
    meta: Dict[str, Any] = {
        "protocol": "causal_prefix_v2",
        "context_position": "prefix",
        "start_frame": start_frame,
        "context_frames": context_frames,
    }
    if context_frames <= 0 or start_frame < context_frames:
        meta["error"] = "insufficient_history"
        return None, None, None, meta

    frames, indices = load_prev_chunk_tail_from_disk(
        dataset_base_path,
        video_name,
        start_frame,
        context_frames,
        nearest_first=False,
    )
    if not frames or not indices:
        meta["error"] = "missing_context_frames"
        return None, None, None, meta
    if any(int(index) >= start_frame for index in indices):
        raise AssertionError("causal context leaked into the target")

    vn = os.path.splitext(str(video_name))[0]
    json_file = os.path.join(dataset_base_path, "jsons", f"{vn}.json")
    target_indices = list(
        range(start_frame, start_frame + int(target_num_frames), int(target_action_stride))
    )
    all_indices = list(indices) + target_indices
    if not os.path.isfile(json_file):
        meta["error"] = "missing_pose_json"
        return None, None, None, meta
    poses = load_camera_poses_batch(json_file, all_indices)
    absolute = [pose_to_rt(pose) if pose else None for pose in poses]
    if len(absolute) != len(all_indices) or any(rt is None for rt in absolute):
        meta["error"] = "missing_pose"
        return None, None, None, meta
    target_reference = absolute[len(indices)]
    relative = (
        convert_rt_to_relative(absolute, target_reference)
        if use_rt_relative
        else [list(rt) for rt in absolute]
    )
    context_actions = relative[: len(indices)]
    target_actions = relative[len(indices) :]
    meta.update(
        {
            "context_frame_indices": list(indices),
            "target_action_indices": target_indices,
            "target_reference_index": start_frame,
        }
    )
    return frames, context_actions, target_actions, meta


def build_causal_continuation_protocol(
    previous_frames: Sequence[Any],
    previous_chunk_actions: Sequence[Sequence[float]],
    context_frames: int = 5,
) -> Tuple[List[Any], List[List[float]], Dict[str, Any]]:
    """Use K latent-aligned poses strictly before the next target start."""
    frames = list(previous_frames or [])
    actions = [list(row) for row in (previous_chunk_actions or [])]
    k = int(context_frames)
    if k <= 0 or not frames or len(actions) < k + 1:
        return [], [], {"protocol": "causal_continuation_v3", "error": "empty_input"}
    history = actions[:-1]
    target_start = actions[-1]
    span = len(history)
    frame_indices = [
        int(round(i * (len(frames) - 1) / max(span, 1))) for i in range(span)
    ]
    action_indices = list(range(len(history) - k, len(history)))
    selected_frame_indices = [frame_indices[i] for i in action_indices]
    selected_frames = [frames[i] for i in selected_frame_indices]
    selected_actions = [history[i] for i in action_indices]
    rebased = convert_rt_to_relative(selected_actions, target_start)
    return selected_frames, rebased, {
        "protocol": "causal_continuation_v3",
        "context_position": "prefix",
        "context_frame_indices": selected_frame_indices,
        "context_action_indices": action_indices,
    }


def build_prev_tail_continuation_protocol(
    previous_frames: Sequence[Any],
    previous_chunk_actions: Sequence[Sequence[float]],
    context_frames: int,
    *,
    nearest_first: bool = True,
) -> Tuple[List[Any], List[List[float]], Dict[str, Any]]:
    """Reproduce suffix prev_chunk_tail ordering and target-relative RT rows."""
    frames = list(previous_frames or [])
    actions = [list(row) for row in (previous_chunk_actions or [])]
    k = min(int(context_frames), len(frames))
    if k <= 0 or not actions:
        return [], [], {"protocol": "prev_tail_continuation_v2", "error": "empty_input"}
    frame_indices = list(range(len(frames) - k, len(frames)))
    if nearest_first:
        frame_indices.reverse()
    if len(frames) == 1 or len(actions) == 1:
        action_indices = [len(actions) - 1] * k
    else:
        action_indices = [
            int(round(i * (len(actions) - 1) / (len(frames) - 1)))
            for i in frame_indices
        ]
    selected_actions = [actions[i] for i in action_indices]
    rebased = convert_rt_to_relative(selected_actions, actions[-1])
    return [frames[i] for i in frame_indices], rebased, {
        "protocol": "prev_tail_continuation_v2",
        "context_position": "suffix" if nearest_first else "prefix",
        "context_frame_indices": frame_indices,
        "context_action_indices": action_indices,
    }


def encode_context_frames(pipe, pil_list, device, dtype=torch.bfloat16, per_frame: bool = False):
    """Encode context frames to latents aligned with training behavior.

    per_frame=False: encode the whole clip once (default training path, temporal downsample).
    per_frame=True: encode each frame separately and concat on latent time.
    """
    if not pil_list:
        return None
    if not per_frame:
        context_video = pipe.preprocess_video(pil_list).to(device=device)
        if context_video.dim() == 5:
            context_video = context_video.squeeze(0)
        context_latents = pipe.vae.encode([context_video], device=pipe.device, tiled=False, tile_size=None, tile_stride=None)
        return context_latents.to(dtype=dtype, device=device)

    encoded = []
    for pil in pil_list:
        frame_video = pipe.preprocess_video([pil]).to(device=device)
        frame_sq = frame_video.squeeze(0) if frame_video.dim() == 5 else frame_video
        if frame_sq.dim() == 3:
            frame_sq = frame_sq.unsqueeze(0)
        lat_one = pipe.vae.encode([frame_sq], device=pipe.device, tiled=False, tile_size=None, tile_stride=None)
        encoded.append(lat_one)
    context_latents = torch.cat(encoded, dim=2).to(dtype=dtype, device=device)
    return context_latents


def _frame_to_pil(f, tw, th):
    if hasattr(f, "convert") and hasattr(f, "resize"):
        return f.convert("RGB").resize((tw, th))
    if isinstance(f, np.ndarray):
        if f.dtype != np.uint8:
            f = (f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
        return Image.fromarray(f).convert("RGB").resize((tw, th))
    if isinstance(f, torch.Tensor):
        fn = f.cpu().numpy()
        if len(fn.shape) == 3 and fn.shape[0] == 3:
            fn = fn.transpose(1, 2, 0)
        fn = (fn * 255).clip(0, 255).astype(np.uint8) if fn.max() <= 1.0 else fn.clip(0, 255).astype(np.uint8)
        return Image.fromarray(fn).convert("RGB").resize((tw, th))
    return f


def run_one_chunk(
    pipe,
    prompt: str,
    use_negative_prompt: str,
    action_path: Optional[str] = None,
    *,
    cam_pose_actions=None,
    context_latents=None,
    num_context_frames: int = 1,
    context_actions_t=None,
    chunk_frames: int = 81,
    h: int = 352,
    w: int = 640,
    seed: int = 0,
    sigma_shift: float = 5.0,
    num_inference_steps: int = 50,
    cfg_scale: float = 5.0,
    inference_noise_level: float = 0.0,
    omit_context_actions: bool = False,  # kept for backward compat, no longer used
    context_position: str = "suffix",
    log_prefix: str = "[multichunk]",
) -> List[Any]:
    """Single chunk generation with explicit context position. VWM-aligned action injection."""
    device = pipe.device
    kwargs_common = dict(
        prompt=prompt,
        negative_prompt=use_negative_prompt,
        height=h,
        width=w,
        num_frames=chunk_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        cfg_scale=cfg_scale,
        sigma_shift=sigma_shift,
        denoising_strength=1.0,
    )
    if action_path is not None:
        kwargs_common["action_path"] = action_path
    elif cam_pose_actions is not None:
        kwargs_common["cam_pose_actions"] = cam_pose_actions

    if context_latents is not None:
        pipe_kw = dict(
            **kwargs_common,
            enable_context_memory=True,
            context_latents=context_latents,
            num_context_frames=num_context_frames,
            context_position=context_position,
            cfg_target_only=True,
            inference_noise_level=inference_noise_level,
        )
        if context_actions_t is not None:
            pipe_kw["context_actions"] = context_actions_t
        with torch.no_grad():
            vid = pipe(**pipe_kw)
    else:
        with torch.no_grad():
            vid = pipe(**kwargs_common, enable_context_memory=False)
    return vid if isinstance(vid, list) else [vid]


def _load_actions_tensor_from_json(
    action_path: Optional[str],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    if not action_path or not os.path.exists(action_path):
        return None
    try:
        with open(action_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        seq = data.get("actions", data)
        items = sorted(
            ((int(k), v) for k, v in seq.items() if str(k).isdigit()),
            key=lambda x: x[0],
        )
        if not items:
            return None
        rows = []
        for _, v in items:
            if isinstance(v, (list, tuple)) and len(v) >= 12:
                rows.append([float(x) for x in v[:12]])
        if not rows:
            return None
        return torch.tensor(rows, device=device, dtype=dtype)
    except Exception:
        return None


def _tail_context_actions(
    src_actions: Optional[torch.Tensor],
    num_ctx: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    nearest_first: bool = False,
) -> Optional[torch.Tensor]:
    if num_ctx <= 0:
        return None
    if src_actions is None or src_actions.numel() == 0:
        return torch.zeros(num_ctx, 12, device=device, dtype=dtype)
    if src_actions.dim() == 3:
        src_actions = src_actions[0]
    if src_actions.shape[0] >= num_ctx:
        out = src_actions[-num_ctx:]
        if nearest_first:
            out = torch.flip(out, dims=[0])
        return out.to(device=device, dtype=dtype)
    pad_n = num_ctx - src_actions.shape[0]
    pad = src_actions[-1:, :].expand(pad_n, src_actions.shape[1])
    out = torch.cat([src_actions, pad], dim=0)
    if nearest_first:
        out = torch.flip(out, dims=[0])
    return out.to(device=device, dtype=dtype)


def sync_pipe_memory_from_training_module(pipe, unwrapped_model: Any) -> Dict[str, Any]:
    """Copy memory-related flags from WanTrainingModule.pipe onto pipe (defensive if pipe handle diverges)."""
    log: Dict[str, Any] = {}
    p = pipe
    m = unwrapped_model
    src = getattr(m, "pipe", None) or p

    def _g(attr, default=None):
        v = getattr(src, attr, None)
        if v is None:
            v = getattr(p, attr, None)
        if v is None:
            v = getattr(m, attr, default)
        return v

    p.use_framepack_memory = bool(_g("use_framepack_memory", False))
    p.context_temporal_decay = float(_g("context_temporal_decay", 1.0) or 1.0)
    p.context_attention_weight = float(_g("context_attention_weight", 1.0) or 1.0)
    p.use_framepack_length_compress = bool(_g("use_framepack_length_compress", False))
    p.framepack_ratio = int(_g("framepack_ratio", 1) or 1)
    p.framepack_length_strategy = str(_g("framepack_length_strategy", "distance_merge") or "distance_merge")
    p.framepack_recent_keep_ratio = float(_g("framepack_recent_keep_ratio", 0.5) or 0.5)
    p.framepack_multiscale_w2 = float(_g("framepack_multiscale_w2", 0.25) or 0.25)
    p.framepack_multiscale_w4 = float(_g("framepack_multiscale_w4", 0.15) or 0.15)
    p.use_spatial_memory = bool(_g("use_spatial_memory", False))
    p.spatial_memory_tokens = int(_g("spatial_memory_tokens", 64) or 64)
    p.use_spatial_memory_legacy = bool(_g("use_spatial_memory_legacy", False))
    p.spatial_memory_inject_mode = str(_g("spatial_memory_inject_mode", "concat_text") or "concat_text")
    sm = getattr(m, "spatial_memory_module", None) or getattr(src, "spatial_memory_module", None) or getattr(p, "spatial_memory_module", None)
    p.spatial_memory_module = sm
    srm = getattr(m, "spatial_memory_readout_module", None) or getattr(src, "spatial_memory_readout_module", None) or getattr(p, "spatial_memory_readout_module", None)
    p.spatial_memory_readout_module = srm
    dit = getattr(p, "dit", None)
    bl0 = dit.blocks[0] if dit is not None and hasattr(dit, "blocks") and len(dit.blocks) > 0 else None
    log.update(
        {
            "use_framepack_memory": p.use_framepack_memory,
            "use_framepack_length_compress": p.use_framepack_length_compress,
            "framepack_ratio": p.framepack_ratio,
            "framepack_length_strategy": p.framepack_length_strategy,
            "use_spatial_memory": p.use_spatial_memory,
            "use_spatial_memory_legacy": p.use_spatial_memory_legacy,
            "spatial_memory_inject_mode": p.spatial_memory_inject_mode,
            "spatial_module": sm is not None,
            "spatial_readout_module": srm is not None,
            "dit_block0_use_block_wise_ssm": bool(getattr(bl0, "use_block_wise_ssm", False)),
            "dit_block0_use_videossm_hybrid": bool(getattr(bl0, "use_videossm_hybrid", False)),
        }
    )
    return log


def run_two_chunk_memory_monitor(
    pipe,
    *,
    prompt: str,
    negative_prompt: str,
    action_path: Optional[str],
    chunk0_action_path: Optional[str] = None,
    chunk1_action_path: Optional[str] = None,
    first_frame_pil,
    context_memory_frames: int,
    chunk_frames: int = 81,
    h: int = 352,
    w: int = 640,
    seed: int = 42,
    sigma_shift: float = 5.0,
    num_inference_steps: int = 50,
    cfg_scale: float = 5.0,
    inference_noise_level: float = 0.0,
    omit_context_actions: bool = False,
    context_source: str = "replay",
    context_position: str = "suffix",
    context_per_frame_vae: bool = False,
    device=None,
    dtype=torch.bfloat16,
    log_prefix: str = "[two_chunk_mem]",
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """
    Chunk1: 1-frame context. Chunk2 context follows context_source:
      - replay: context_frames_for_next_chunk
      - prev_chunk_tail: strict tail frames (nearest-first)
      - causal_prev_prefix: latent-aligned history strictly before next target

    Returns (frames_ch0, frames_ch1, meta). chunk0 defaults left_45 and chunk1 defaults right_45 when provided by caller.
    """
    device = device or pipe.device
    context_source = (context_source or "replay").strip().lower()
    if context_source not in (
        "replay",
        "prev_chunk_tail",
        "causal_prev_prefix",
    ):
        context_source = "replay"
    context_position = (context_position or "suffix").strip().lower()
    if context_position not in ("prefix", "suffix"):
        context_position = "suffix"
    meta: Dict[str, Any] = {
        "n_ctx": int(context_memory_frames),
        "chunk_frames": chunk_frames,
        "context_source": context_source,
        "context_position": context_position,
        "context_per_frame_vae": bool(context_per_frame_vae),
    }

    ff = first_frame_pil
    if isinstance(ff, Image.Image):
        ff = ff.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    else:
        ff = _frame_to_pil(ff, w, h)

    ctx_lat_0 = encode_context_frames(pipe, [ff], device, dtype=dtype, per_frame=bool(context_per_frame_vae))
    num_ctx0 = int(ctx_lat_0.shape[2]) if ctx_lat_0 is not None else 1
    meta["chunk0_num_context_latent"] = num_ctx0

    use_omit_ch0 = omit_context_actions or (num_ctx0 <= 1)
    act0 = chunk0_action_path or action_path
    act1 = chunk1_action_path or action_path
    src_actions0 = _load_actions_tensor_from_json(act0, device=device, dtype=torch.float32)
    meta["chunk0_action_path"] = act0
    meta["chunk1_action_path"] = act1

    frames_ch0 = run_one_chunk(
        pipe,
        prompt,
        negative_prompt,
        act0,
        context_latents=ctx_lat_0,
        num_context_frames=num_ctx0,
        context_actions_t=None,
        chunk_frames=chunk_frames,
        h=h,
        w=w,
        seed=seed,
        sigma_shift=sigma_shift,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        inference_noise_level=inference_noise_level,
        omit_context_actions=use_omit_ch0,
        context_position=context_position,
        log_prefix=log_prefix + " ch0",
    )

    pil_ch0 = [_frame_to_pil(f, w, h) for f in frames_ch0]
    n_ctx = int(context_memory_frames)
    if n_ctx <= 0:
        n_ctx = 1
    protocol_actions = None
    source_rows = (
        src_actions0.detach().cpu().tolist() if src_actions0 is not None else []
    )
    if context_source == "causal_prev_prefix":
        if context_position != "prefix":
            raise ValueError("causal_prev_prefix monitor requires prefix layout")
        prev_pil, protocol_actions, protocol_meta = (
            build_causal_continuation_protocol(pil_ch0, source_rows, n_ctx)
        )
        meta["continuation_protocol"] = protocol_meta
    elif context_source == "prev_chunk_tail":
        prev_pil, protocol_actions, protocol_meta = (
            build_prev_tail_continuation_protocol(
                pil_ch0,
                source_rows,
                n_ctx,
                nearest_first=(context_position == "suffix"),
            )
        )
        meta["continuation_protocol"] = protocol_meta
    else:
        prev_pil = context_frames_for_next_chunk(pil_ch0, n_ctx)
    meta["chunk1_context_count"] = len(prev_pil)

    ctx_lat_1 = encode_context_frames(pipe, prev_pil, device, dtype=dtype, per_frame=bool(context_per_frame_vae))
    num_ctx1 = int(ctx_lat_1.shape[2]) if ctx_lat_1 is not None else len(prev_pil)
    meta["chunk1_num_context_latent"] = num_ctx1

    # Align with training: when context has only 1 latent frame, context actions are omitted.
    # train.py sets omit_context_actions=True when context_memory_frames == 1.
    use_omit_ch1 = omit_context_actions or (num_ctx1 <= 1)
    ca1 = None
    if not use_omit_ch1 and num_ctx1 > 0:
        if protocol_actions is not None:
            ca1 = torch.tensor(
                protocol_actions[:num_ctx1],
                device=device,
                dtype=torch.float32,
            )
        else:
            ca1 = _tail_context_actions(
                src_actions0,
                num_ctx1,
                device=device,
                dtype=torch.float32,
                nearest_first=False,
            )
    meta["chunk1_context_actions_count"] = int(ca1.shape[0]) if ca1 is not None else 0

    frames_ch1 = run_one_chunk(
        pipe,
        prompt,
        negative_prompt,
        act1,
        context_latents=ctx_lat_1,
        num_context_frames=num_ctx1,
        context_actions_t=ca1,
        chunk_frames=chunk_frames,
        h=h,
        w=w,
        seed=seed + 1,
        sigma_shift=sigma_shift,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        inference_noise_level=inference_noise_level,
        omit_context_actions=use_omit_ch1,
        context_position=context_position,
        log_prefix=log_prefix + " ch1",
    )

    meta["note"] = "No cross-chunk SSM/RNN state; only frame-conditioned second chunk (same as replay eval)."
    return frames_ch0, frames_ch1, meta

def load_model(checkpoint_path, model_paths, lora_path=None, lora_alpha=1.0, device="cuda"):
    """Load model from checkpoint"""
    print(f"Loading model from checkpoint: {checkpoint_path}")

    # Load base pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )

    # Load LoRA if specified
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA from: {lora_path}")
        pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha)

    # Load checkpoint if specified
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = safe_load_file(checkpoint_path)
        pipe.dit.load_state_dict(checkpoint, strict=False)

    pipe.enable_vram_management()
    pipe.eval()

    return pipe


def sample_prompts_from_dataset(dataset, num_prompts=5):
    """Randomly sample prompts from dataset"""
    prompts = []
    dataset_size = len(dataset)

    if dataset_size == 0:
        print("Warning: Dataset is empty, using default prompts")
        return ["A cyberpunk city game scene, a character walking through neon-lit streets"] * num_prompts

    # Sample random indices
    indices = random.sample(range(dataset_size), min(num_prompts, dataset_size))

    print(f"Sampling {len(indices)} prompts from dataset (size: {dataset_size})...")
    for idx in indices:
        try:
            sample = dataset[idx]
            if isinstance(sample, dict):
                prompt = sample.get("description") or sample.get("prompt") or sample.get("text", "")
                if prompt:
                    prompts.append(prompt)
                else:
                    print(f"Warning: Sample {idx} has no prompt field, skipping")
            else:
                print(f"Warning: Sample {idx} is not a dict, skipping")
        except Exception as e:
            print(f"Warning: Failed to load sample {idx}: {e}, skipping")

    # Fill with default if not enough prompts
    while len(prompts) < num_prompts:
        prompts.append("A cyberpunk city game scene, a character walking through neon-lit streets")

    return prompts[:num_prompts]


def encode_frames_to_latents(pipe, frames):
    """Encode frames to latents using VAE"""
    pipe.load_models_to_device(["vae"])
    vae = pipe.vae

    latents_list = []
    for frame in frames:
        vid = pipe.preprocess_video([frame]).squeeze(0)
        with torch.no_grad():
            lat = vae.encode([vid], device=pipe.device)[0].unsqueeze(0)
            latents_list.append(lat)

    if latents_list:
        return torch.cat(latents_list, dim=2)
    return None


def generate_long_video(
    pipe,
    prompt,
    negative_prompt="oversaturated colors, overexposed, static, blurry details",
    output_dir="./long_video_output",
    video_name="long_video",
    context_memory_frames=4,
    frames_per_segment=81,
    target_frames=450,  # 30 seconds at 15fps
    height=352,
    width=640,
    num_inference_steps=20,
    cfg_scale=5.0,
    timestep_shift=1.0,
    seed=42,
    fps=15,
):
    """
    Generate long video using iterative context-based generation

    Args:
        pipe: WanVideoPipeline instance
        prompt: Text prompt for generation
        negative_prompt: Negative prompt
        output_dir: Output directory for videos
        video_name: Base name for output video
        context_memory_frames: Number of context frames to use (K)
        frames_per_segment: Frames to generate per segment (default: 81)
        target_frames: Target total frames (default: 450 for 30s at 15fps)
        height: Video height
        width: Video width
        num_inference_steps: Number of inference steps
        cfg_scale: CFG scale
        timestep_shift: Timestep shift
        seed: Random seed
        fps: FPS for output video
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set environment variable for concatenation inference
    os.environ["USE_CONCATENATION_INFERENCE"] = "true"

    all_frames = []
    current_context_latents = None
    current_context_frames = []

    # Calculate number of segments needed
    num_segments = (target_frames + frames_per_segment - 1) // frames_per_segment

    print(f"Generating long video: {target_frames} frames in {num_segments} segments")
    print(f"  - Frames per segment: {frames_per_segment}")
    print(f"  - Context frames: {context_memory_frames}")
    print(f"  - Prompt: {prompt[:100]}...")

    torch.manual_seed(seed)

    for segment_idx in range(num_segments):
        # Calculate frames to generate for this segment
        remaining_frames = target_frames - len(all_frames)
        frames_to_generate = min(frames_per_segment, remaining_frames)

        if frames_to_generate <= 0:
            break

        print(f"\n[{segment_idx + 1}/{num_segments}] Generating {frames_to_generate} frames...")

        # Prepare sampling kwargs
        # First segment: no context (generate from scratch)
        # Subsequent segments: use context from previous segment
        has_context = current_context_latents is not None and segment_idx > 0

        sampling_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": frames_to_generate,
            "num_inference_steps": num_inference_steps,
            "seed": seed + segment_idx,  # Different seed for each segment
            "cfg_scale": cfg_scale,
            "sigma_shift": timestep_shift,
            "denoising_strength": 1.0,
        }

        # Add context memory only if we have context
        if has_context:
            sampling_kwargs["enable_context_memory"] = True
            sampling_kwargs["context_latents"] = current_context_latents
            sampling_kwargs["num_context_frames"] = len(current_context_frames)

        try:
            # Generate frames
            if has_context:
                print(f"  Using {len(current_context_frames)} context frames from previous segment...")

            generated_frames = pipe(**sampling_kwargs)

            if isinstance(generated_frames, list):
                segment_frames = generated_frames
            else:
                segment_frames = [generated_frames] if hasattr(generated_frames, '__iter__') else [generated_frames]

            # Add to all frames
            all_frames.extend(segment_frames)

            # Update context: use last K frames from generated segment
            # These will be used as context for the next segment
            if len(segment_frames) >= context_memory_frames:
                context_frames = segment_frames[-context_memory_frames:]
                current_context_frames = context_frames

                # Encode context frames to latents
                print(f"  Encoding last {context_memory_frames} frames as context for next segment...")
                current_context_latents = encode_frames_to_latents(pipe, context_frames)
            else:
                # If not enough frames, use all frames as context
                current_context_frames = segment_frames
                current_context_latents = encode_frames_to_latents(pipe, segment_frames)

            print(f"  Generated {len(segment_frames)} frames (total: {len(all_frames)}/{target_frames})")

        except Exception as e:
            print(f"  Error generating segment {segment_idx + 1}: {e}")
            traceback.print_exc()
            break

    # Save final video
    if len(all_frames) > 0:
        output_path = os.path.join(output_dir, f"{video_name}.mp4")
        print(f"\nSaving video to: {output_path}")
        print(f"  Total frames: {len(all_frames)}")
        print(f"  Duration: {len(all_frames) / fps:.2f} seconds")

        save_video(all_frames, output_path, fps=fps, quality=5)
        print(f"Video saved: {output_path}")

        # Save prompt
        prompt_path = os.path.join(output_dir, f"{video_name}_prompt.txt")
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)

        return output_path
    else:
        print("Error: No frames generated")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate long videos using iterative context-based generation")

    # Model paths
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha")
    parser.add_argument("--model_paths", type=str, default=None, help="JSON string of model paths (not used if checkpoint_path is set)")

    # Dataset
    parser.add_argument("--dataset_base_path", type=str, required=True, help="Base path to dataset")
    parser.add_argument("--dataset_metadata_path", type=str, required=True, help="Path to dataset metadata CSV")
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of prompts to sample from dataset")

    # Generation parameters
    parser.add_argument("--output_dir", type=str, default="./long_video_output", help="Output directory")
    parser.add_argument("--context_memory_frames", type=int, default=4, help="Number of context frames (K)")
    parser.add_argument("--frames_per_segment", type=int, default=81, help="Frames per segment (default: 81)")
    parser.add_argument("--target_frames", type=int, default=450, help="Target total frames (30s at 15fps)")
    parser.add_argument("--height", type=int, default=352, help="Video height")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--timestep_shift", type=float, default=1.0, help="Timestep shift")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=15, help="FPS for output video")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load dataset for prompt sampling
    print("Loading dataset...")
    dataset_args = wan_parser.parse_args([])  # Create minimal args
    dataset_args.dataset_base_path = args.dataset_base_path
    dataset_args.dataset_metadata_path = args.dataset_metadata_path
    dataset_args.height = args.height
    dataset_args.width = args.width

    dataset = VideoDataset(args=dataset_args)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Sample prompts
    prompts = sample_prompts_from_dataset(dataset, args.num_prompts)
    print(f"Sampled {len(prompts)} prompts")

    # Load model
    model_paths = None
    if args.model_paths:
        model_paths = json.loads(args.model_paths)

    pipe = load_model(
        checkpoint_path=args.checkpoint_path,
        model_paths=model_paths,
        lora_path=args.lora_path,
        lora_alpha=args.lora_alpha,
        device=args.device,
    )

    # Generate videos for each prompt
    output_paths = []
    for idx, prompt in enumerate(prompts):
        print(f"\n{'='*80}")
        print(f"Generating video {idx + 1}/{len(prompts)}")
        print(f"{'='*80}")

        video_name = f"long_video_{idx + 1:03d}"

        output_path = generate_long_video(
            pipe=pipe,
            prompt=prompt,
            output_dir=args.output_dir,
            video_name=video_name,
            context_memory_frames=args.context_memory_frames,
            frames_per_segment=args.frames_per_segment,
            target_frames=args.target_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            timestep_shift=args.timestep_shift,
            seed=args.seed + idx,  # Different seed for each video
            fps=args.fps,
        )

        if output_path:
            output_paths.append(output_path)

    print(f"\n{'='*80}")
    print(f"Generation completed: {len(output_paths)} videos generated")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}")
