"""
Shared multichunk sampling for training monitor and replay scripts.

Two-chunk path matches run_replay_loop_two_chunk: chunk1 with 1-frame context,
chunk2 with context_frames_for_next_chunk from chunk1 output.
No hidden state is carried across chunks (each pipe() is independent diffusion), only PIL frames + latents.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from .context_chunk_utils import context_frames_for_next_chunk, replay_context_from_generated_frames
except ImportError:
    try:
        from src.model_training.context_chunk_utils import context_frames_for_next_chunk, replay_context_from_generated_frames
    except ImportError:
        from context_chunk_utils import context_frames_for_next_chunk, replay_context_from_generated_frames


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
    from PIL import Image

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

    Returns (frames_ch0, frames_ch1, meta). chunk0 defaults left_45 and chunk1 defaults right_45 when provided by caller.
    """
    from PIL import Image

    device = device or pipe.device
    context_source = (context_source or "replay").strip().lower()
    if context_source not in ("replay", "prev_chunk_tail"):
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
    if context_source == "prev_chunk_tail":
        tail = pil_ch0[-n_ctx:]
        prev_pil = list(reversed(tail)) if context_position == "suffix" else tail
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
        ca1 = _tail_context_actions(
            src_actions0,
            num_ctx1,
            device=device,
            dtype=torch.float32,
            nearest_first=(context_source == "prev_chunk_tail" and context_position == "suffix"),
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
