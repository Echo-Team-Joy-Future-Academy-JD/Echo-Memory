#!/usr/bin/env python3
"""
Infer memory-baseline runtime flags from CKPT path (aligned with train/memory_baselines_basic).

与训练脚本逐项对应（exp1_4_4_cam_rt_paper_style/train/memory_baselines_basic/）：

| 路径子串 | 训练脚本 | 推理 pipe / MEM_ARGS |
|----------|----------|----------------------|
| *_framepack_weight_only | run_framepack_baseline.sh | use_framepack_memory, decay=0.9, weight=1.0；**无** length compress |
| *_framepack_lencompress_r2 | run_framepack_lencompress_r2.sh | use_framepack_length_compress, framepack_ratio=2 |
| *_framepack_lencompress_r4 | run_framepack_lencompress_r4.sh | use_framepack_length_compress, framepack_ratio=4 |
| *_spatial_mem | run_spatial_memory_baseline.sh | use_spatial_memory, tokens=64, **legacy=False**（SpatialGridMemory，与 train 默认一致；grid=8 由 ckpt 权重形状在 loop_utils 中还原） |
| *_videossm_hybrid | run_videossm_hybrid_baseline.sh | 仅 context_frames=5；VideoSSM 结构由 loop_utils.load_pipeline_and_ckpt 按 ckpt key 推断 |

共同：训练均为 context_memory_frames=5 → CONTEXT_FRAMES_MEM_OVERRIDE=5。

压缩记忆：训练侧 K→K' 与 RT 对齐在 wan_video_new.framepack_*；推理传 pipe.use_framepack_length_compress + framepack_ratio 即可，与 train.py 写入 pipe 一致。

空间记忆：若 ckpt 中无 spatial_memory_module.*，则无法与训练 SpatialGridMemory 对齐；apply_memory_baseline_pipe 会降级 legacy 并打印警告（与 run_replay_loop_two_chunk 逻辑一致）。

Used by:
- bash: `eval "$(python3 memory_baseline_runtime.py bash-export "$CKPT")"` → MEM_ARGS array + CONTEXT_FRAMES_MEM_OVERRIDE
- Python: apply_memory_baseline_pipe(pipe, ckpt) after load_pipeline_and_ckpt
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class MemoryProfile:
    use_framepack_memory: bool = False
    context_temporal_decay: float = 1.0
    context_attention_weight: float = 1.0
    use_framepack_length_compress: bool = False
    framepack_ratio: int = 2
    framepack_length_strategy: str = "distance_merge"
    framepack_multiscale_w2: float = 0.25
    framepack_multiscale_w4: float = 0.15
    use_spatial_memory: bool = False
    use_spatial_memory_legacy: bool = False  # True = old adaptive pool only (no SpatialGridMemory in ckpt)
    spatial_memory_tokens: int = 64
    spatial_memory_inject_mode: Optional[str] = None
    use_geometry_spatial_memory: bool = False
    geometry_spatial_memory_inject_mode: Optional[str] = None
    use_block_wise_ssm: bool = False
    block_wise_ssm_causal_v2: bool = False
    use_videossm_hybrid: bool = False
    use_moc: bool = False
    moc_temperature: float = 1.0
    # Training uses context_memory_frames=5 for memory_baselines_basic
    context_override: Optional[int] = None
    context_position: Optional[str] = None
    training_context_source: Optional[str] = None


@dataclass(frozen=True)
class MemoryProfileSpec:
    profile_id: str
    ckpt_substrings: Sequence[str]
    paper_tag: str
    train_flags: Sequence[str]
    infer_flags: Sequence[str]
    eval_flags: Sequence[str]
    profile: MemoryProfile


MEMORY_PROFILE_REGISTRY: List[MemoryProfileSpec] = [
    MemoryProfileSpec(
        profile_id="context_k1",
        ckpt_substrings=("context_k1", "ctx_1", "ctx1"),
        paper_tag="raw_context",
        train_flags=("--context_memory_frames 1",),
        infer_flags=(),
        eval_flags=("--context_frames 1",),
        profile=MemoryProfile(context_override=1),
    ),
    MemoryProfileSpec(
        profile_id="context_k5",
        ckpt_substrings=("context_k5", "ctx_5", "ctx5"),
        paper_tag="raw_context",
        train_flags=("--context_memory_frames 5",),
        infer_flags=(),
        eval_flags=("--context_frames 5",),
        profile=MemoryProfile(context_override=5),
    ),
    MemoryProfileSpec(
        profile_id="context_k20",
        ckpt_substrings=("context_k20", "ctx_20", "ctx20"),
        paper_tag="raw_context",
        train_flags=("--context_memory_frames 20",),
        infer_flags=(),
        eval_flags=("--context_frames 20",),
        profile=MemoryProfile(context_override=20),
    ),
    MemoryProfileSpec(
        profile_id="framepack_weight_only",
        ckpt_substrings=(
            "memory_baselines_basic_framepack_weight_only",
            "memory_baselines_basic_abl_framepack_weight_two_chunk",
        ),
        paper_tag="arXiv:2504.12626",
        train_flags=(
            "--use_framepack_memory",
            "--context_temporal_decay 0.9",
            "--context_attention_weight 1.0",
            "--context_memory_frames 81",
        ),
        infer_flags=("--use_framepack_memory", "--context_temporal_decay 0.9", "--context_attention_weight 1.0"),
        eval_flags=("--context_frames 81",),
        profile=MemoryProfile(
            use_framepack_memory=True,
            context_temporal_decay=0.9,
            context_attention_weight=1.0,
            context_override=81,
        ),
    ),
    MemoryProfileSpec(
        profile_id="framepack_lencompress_r2",
        ckpt_substrings=(
            "memory_baselines_basic_framepack_lencompress_r2",
            "memory_baselines_basic_abl_framepack_len_r2_two_chunk",
        ),
        paper_tag="arXiv:2504.12626",
        train_flags=("--use_framepack_length_compress", "--framepack_ratio 2", "--context_memory_frames 81"),
        infer_flags=("--use_framepack_length_compress", "--framepack_ratio 2"),
        eval_flags=("--context_frames 81",),
        profile=MemoryProfile(
            use_framepack_length_compress=True,
            framepack_ratio=2,
            context_override=81,
        ),
    ),
    MemoryProfileSpec(
        profile_id="framepack_lencompress_r4",
        ckpt_substrings=(
            "memory_baselines_basic_framepack_lencompress_r4",
            "memory_baselines_basic_abl_framepack_len_r4_two_chunk",
        ),
        paper_tag="arXiv:2504.12626",
        train_flags=("--use_framepack_length_compress", "--framepack_ratio 4", "--context_memory_frames 81"),
        infer_flags=("--use_framepack_length_compress", "--framepack_ratio 4"),
        eval_flags=("--context_frames 81",),
        profile=MemoryProfile(
            use_framepack_length_compress=True,
            framepack_ratio=4,
            context_override=81,
        ),
    ),
    MemoryProfileSpec(
        profile_id="spatial_mem",
        ckpt_substrings=("memory_baselines_basic_spatial_mem", "spatial_mem"),
        paper_tag="token_grid_baseline",
        train_flags=("--use_spatial_memory", "--spatial_memory_tokens 64", "--context_memory_frames 1"),
        infer_flags=("--use_spatial_memory", "--spatial_memory_tokens 64"),
        eval_flags=("--context_frames 5",),
        profile=MemoryProfile(
            use_spatial_memory=True,
            use_spatial_memory_legacy=False,
            spatial_memory_tokens=64,
            context_override=1,
        ),
    ),
    MemoryProfileSpec(
        profile_id="geometry_spatial_mem",
        ckpt_substrings=("geometry_spatial_mem",),
        paper_tag="geometry_grounded_adaptation_of_arXiv:2506.05284",
        train_flags=(
            "--use_geometry_spatial_memory",
            "--geometry_spatial_memory_tokens 64",
            "--context_memory_frames 1",
        ),
        infer_flags=("--use_geometry_spatial_memory",),
        eval_flags=("--context_frames 1",),
        profile=MemoryProfile(
            use_geometry_spatial_memory=True,
            geometry_spatial_memory_inject_mode="concat_text",
            context_override=1,
        ),
    ),
    MemoryProfileSpec(
        profile_id="framepack_hybrid_r2_weight_two_chunk",
        ckpt_substrings=("memory_baselines_basic_abl_framepack_hybrid_r2_weight_two_chunk",),
        paper_tag="hybrid_compression",
        train_flags=(
            "--use_framepack_memory",
            "--context_temporal_decay 0.95",
            "--context_attention_weight 1.1",
            "--use_framepack_length_compress",
            "--framepack_ratio 2",
            "--context_memory_frames 81",
        ),
        infer_flags=(
            "--use_framepack_memory",
            "--context_temporal_decay 0.95",
            "--context_attention_weight 1.1",
            "--use_framepack_length_compress",
            "--framepack_ratio 2",
        ),
        eval_flags=("--context_frames 81",),
        profile=MemoryProfile(
            use_framepack_memory=True,
            context_temporal_decay=0.95,
            context_attention_weight=1.1,
            use_framepack_length_compress=True,
            framepack_ratio=2,
            context_override=81,
        ),
    ),
    MemoryProfileSpec(
        profile_id="framepack_hybrid_r4_weight_two_chunk",
        ckpt_substrings=("memory_baselines_basic_abl_framepack_hybrid_r4_weight_two_chunk",),
        paper_tag="hybrid_compression",
        train_flags=(
            "--use_framepack_memory",
            "--context_temporal_decay 0.95",
            "--context_attention_weight 1.1",
            "--use_framepack_length_compress",
            "--framepack_ratio 4",
            "--context_memory_frames 81",
        ),
        infer_flags=(
            "--use_framepack_memory",
            "--context_temporal_decay 0.95",
            "--context_attention_weight 1.1",
            "--use_framepack_length_compress",
            "--framepack_ratio 4",
        ),
        eval_flags=("--context_frames 81",),
        profile=MemoryProfile(
            use_framepack_memory=True,
            context_temporal_decay=0.95,
            context_attention_weight=1.1,
            use_framepack_length_compress=True,
            framepack_ratio=4,
            context_override=81,
        ),
    ),
    MemoryProfileSpec(
        profile_id="spatial_concat_text_two_chunk",
        ckpt_substrings=("memory_baselines_basic_abl_spatial_concat_text_two_chunk", "spatial_concat_text_two_chunk"),
        paper_tag="token_grid_baseline",
        train_flags=(
            "--use_spatial_memory",
            "--spatial_memory_tokens 64",
            "--spatial_memory_inject_mode concat_text",
            "--context_memory_frames 1",
        ),
        infer_flags=("--use_spatial_memory", "--spatial_memory_tokens 64", "--spatial_memory_inject_mode concat_text"),
        eval_flags=("--context_frames 1",),
        profile=MemoryProfile(
            use_spatial_memory=True,
            use_spatial_memory_legacy=False,
            spatial_memory_tokens=64,
            spatial_memory_inject_mode="concat_text",
            context_override=1,
        ),
    ),
    MemoryProfileSpec(
        profile_id="spatial_inject_none_two_chunk",
        ckpt_substrings=("memory_baselines_basic_abl_spatial_inject_none_two_chunk", "spatial_inject_none_two_chunk"),
        paper_tag="token_grid_baseline",
        train_flags=(
            "--use_spatial_memory",
            "--spatial_memory_tokens 64",
            "--spatial_memory_inject_mode none",
            "--context_memory_frames 1",
        ),
        infer_flags=("--use_spatial_memory", "--spatial_memory_tokens 64", "--spatial_memory_inject_mode none"),
        eval_flags=("--context_frames 1",),
        profile=MemoryProfile(
            use_spatial_memory=True,
            use_spatial_memory_legacy=False,
            spatial_memory_tokens=64,
            spatial_memory_inject_mode="none",
            context_override=1,
        ),
    ),
    MemoryProfileSpec(
        profile_id="spatial_cross_attn_readout_two_chunk",
        ckpt_substrings=("memory_baselines_basic_abl_spatial_cross_attn_readout_two_chunk", "spatial_cross_attn_readout_two_chunk"),
        paper_tag="token_grid_baseline",
        train_flags=(
            "--use_spatial_memory",
            "--spatial_memory_tokens 64",
            "--spatial_memory_inject_mode cross_attn_readout",
            "--context_memory_frames 1",
        ),
        infer_flags=("--use_spatial_memory", "--spatial_memory_tokens 64", "--spatial_memory_inject_mode cross_attn_readout"),
        eval_flags=("--context_frames 1",),
        profile=MemoryProfile(
            use_spatial_memory=True,
            use_spatial_memory_legacy=False,
            spatial_memory_tokens=64,
            spatial_memory_inject_mode="cross_attn_readout",
            context_override=1,
        ),
    ),
    MemoryProfileSpec(
        profile_id="spatial_cross_attn_readout_t32_g4_two_chunk",
        ckpt_substrings=(
            "memory_baselines_basic_abl_spatial_cross_attn_readout_t32_g4_two_chunk",
            "spatial_cross_attn_readout_t32_g4_two_chunk",
        ),
        paper_tag="token_grid_baseline",
        train_flags=(
            "--use_spatial_memory",
            "--spatial_memory_tokens 32",
            "--spatial_memory_inject_mode cross_attn_readout",
            "--context_memory_frames 1",
        ),
        infer_flags=("--use_spatial_memory", "--spatial_memory_tokens 32", "--spatial_memory_inject_mode cross_attn_readout"),
        eval_flags=("--context_frames 1",),
        profile=MemoryProfile(
            use_spatial_memory=True,
            use_spatial_memory_legacy=False,
            spatial_memory_tokens=32,
            spatial_memory_inject_mode="cross_attn_readout",
            context_override=1,
        ),
    ),
    MemoryProfileSpec(
        profile_id="spatial_legacy_concat_t32_two_chunk",
        ckpt_substrings=("memory_baselines_basic_abl_spatial_legacy_concat_t32_two_chunk",),
        paper_tag="token_grid_baseline",
        train_flags=(
            "--use_spatial_memory",
            "--use_spatial_memory_legacy",
            "--spatial_memory_tokens 32",
            "--spatial_memory_inject_mode concat_text",
            "--context_memory_frames 1",
        ),
        infer_flags=(
            "--use_spatial_memory",
            "--use_spatial_memory_legacy",
            "--spatial_memory_tokens 32",
            "--spatial_memory_inject_mode concat_text",
        ),
        eval_flags=("--context_frames 1",),
        profile=MemoryProfile(
            use_spatial_memory=True,
            use_spatial_memory_legacy=True,
            spatial_memory_tokens=32,
            spatial_memory_inject_mode="concat_text",
            context_override=1,
        ),
    ),
    MemoryProfileSpec(
        profile_id="framepack_len_r8",
        ckpt_substrings=("framepack_len_r8",),
        paper_tag="arXiv:2504.12626",
        train_flags=(
            "--context_source prev_chunk_tail",
            "--context_memory_frames 81",
            "--use_framepack_length_compress",
            "--framepack_ratio 8",
            "--framepack_length_strategy packed_multiscale",
            "--framepack_multiscale_w2 0.25",
            "--framepack_multiscale_w4 0.15",
            "--use_moc",
        ),
        infer_flags=(
            "--use_framepack_length_compress",
            "--framepack_ratio 8",
            "--framepack_length_strategy packed_multiscale",
            "--use_moc",
        ),
        eval_flags=("--context_frames 81",),
        profile=MemoryProfile(
            use_framepack_length_compress=True,
            framepack_ratio=8,
            framepack_length_strategy="packed_multiscale",
            framepack_multiscale_w2=0.25,
            framepack_multiscale_w4=0.15,
            use_moc=True,
            context_override=81,
            context_position="suffix",
            training_context_source="prev_chunk_tail",
        ),
    ),
    MemoryProfileSpec(
        profile_id="block_wise_ssm_causal_v2",
        ckpt_substrings=("block_wise_ssm_causal_v2",),
        paper_tag="causal_block_ssm_v2",
        train_flags=(
            "--use_block_wise_ssm",
            "--block_wise_ssm_causal_v2",
            "--context_source causal_prev_prefix",
            "--context_memory_frames 5",
            "CONTEXT_POSITION=prefix",
        ),
        infer_flags=(
            "load_pipeline_and_ckpt auto-detects residual_logit",
            "--context_position prefix",
        ),
        eval_flags=("--context_frames 5", "--sigma_shift 15"),
        profile=MemoryProfile(
            use_block_wise_ssm=True,
            block_wise_ssm_causal_v2=True,
            context_override=5,
            context_position="prefix",
            training_context_source="causal_prev_prefix",
        ),
    ),
    MemoryProfileSpec(
        profile_id="videossm_hybrid_legacy",
        ckpt_substrings=(
            "memory_baselines_basic_videossm_hybrid",
            "memory_baselines_basic_abl_videossm_hybrid_two_chunk",
        ),
        paper_tag="legacy_hybrid",
        train_flags=("--use_videossm_hybrid", "--context_memory_frames 5"),
        infer_flags=("load_pipeline_and_ckpt auto-infers VideoSSM hybrid from videossm_hybrid.* ckpt keys",),
        eval_flags=("--context_frames 5",),
        profile=MemoryProfile(use_videossm_hybrid=True, context_override=5),
    ),
    MemoryProfileSpec(
        profile_id="block_wise_ssm_two_chunk",
        ckpt_substrings=("memory_baselines_basic_abl_block_wise_ssm_two_chunk",),
        paper_tag="block_wise_recurrence",
        train_flags=("--use_block_wise_ssm", "--context_memory_frames 5"),
        infer_flags=("load_pipeline_and_ckpt auto-infers Block-wise SSM from block_wise_ssm.* ckpt keys",),
        eval_flags=("--context_frames 5",),
        profile=MemoryProfile(use_block_wise_ssm=True, context_override=5),
    ),
    MemoryProfileSpec(
        profile_id="no_memory_extra_two_chunk",
        ckpt_substrings=("memory_baselines_basic_abl_no_memory_extra_two_chunk",),
        paper_tag="anchor_only_reference",
        train_flags=("--context_memory_frames 5",),
        infer_flags=(),
        eval_flags=("--context_frames 5",),
        profile=MemoryProfile(context_override=5),
    ),
]


def profile_registry_payload() -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for spec in MEMORY_PROFILE_REGISTRY:
        payload.append(
            {
                "profile_id": spec.profile_id,
                "paper_tag": spec.paper_tag,
                "ckpt_substrings": list(spec.ckpt_substrings),
                "train_flags": list(spec.train_flags),
                "infer_flags": list(spec.infer_flags),
                "eval_flags": list(spec.eval_flags),
                "context_override": spec.profile.context_override,
            }
        )
    return payload


def infer_memory_profile_spec(ckpt: str) -> Optional[MemoryProfileSpec]:
    c = (ckpt or "").lower()
    best = None
    best_length = -1
    for spec in MEMORY_PROFILE_REGISTRY:
        for token in spec.ckpt_substrings:
            token_lower = token.lower()
            if token_lower in c and len(token_lower) > best_length:
                best = spec
                best_length = len(token_lower)
    return best


def infer_memory_profile(ckpt: str) -> MemoryProfile:
    spec = infer_memory_profile_spec(ckpt)
    if spec is not None:
        return spec.profile
    return MemoryProfile()


def profile_to_argv(p: MemoryProfile) -> List[str]:
    out: List[str] = []
    if p.context_position:
        out.extend(["--context_position", str(p.context_position)])
    if p.training_context_source:
        out.extend(["--training_context_source", str(p.training_context_source)])
    if p.use_framepack_memory:
        out.extend(
            [
                "--use_framepack_memory",
                "--context_temporal_decay",
                str(p.context_temporal_decay),
                "--context_attention_weight",
                str(p.context_attention_weight),
            ]
        )
    if p.use_framepack_length_compress:
        out.extend(
            [
                "--use_framepack_length_compress",
                "--framepack_ratio",
                str(int(p.framepack_ratio)),
                "--framepack_length_strategy",
                str(p.framepack_length_strategy),
            ]
        )
        if p.framepack_length_strategy == "packed_multiscale":
            out.extend(
                [
                    "--framepack_multiscale_w2",
                    str(p.framepack_multiscale_w2),
                    "--framepack_multiscale_w4",
                    str(p.framepack_multiscale_w4),
                ]
            )
    if p.use_spatial_memory:
        out.extend(
            [
                "--use_spatial_memory",
                "--spatial_memory_tokens",
                str(int(p.spatial_memory_tokens)),
            ]
        )
        if p.spatial_memory_inject_mode:
            out.extend(["--spatial_memory_inject_mode", str(p.spatial_memory_inject_mode)])
        if p.use_spatial_memory_legacy:
            out.append("--use_spatial_memory_legacy")
    if p.use_geometry_spatial_memory:
        out.append("--use_geometry_spatial_memory")
        if p.geometry_spatial_memory_inject_mode:
            out.extend(
                [
                    "--geometry_spatial_memory_inject_mode",
                    str(p.geometry_spatial_memory_inject_mode),
                ]
            )
    if p.use_block_wise_ssm:
        out.append("--use_block_wise_ssm")
    if p.use_videossm_hybrid:
        out.append("--use_videossm_hybrid")
    if p.use_moc:
        out.extend(["--use_moc", "--moc_temperature", str(p.moc_temperature)])
    return out


def bash_export(ckpt: str) -> str:
    """Print bash snippet: unset override, MEM_ARGS=(), optional CONTEXT_FRAMES_MEM_OVERRIDE, MEM_ARGS+=."""
    prof = infer_memory_profile(ckpt)
    lines = [
        "unset CONTEXT_FRAMES_MEM_OVERRIDE 2>/dev/null || true",
        "MEM_ARGS=()",
    ]
    if prof.context_override is not None:
        lines.append(f"CONTEXT_FRAMES_MEM_OVERRIDE={int(prof.context_override)}")
    for arg in profile_to_argv(prof):
        lines.append(f"MEM_ARGS+=({shlex.quote(arg)})")
    return "\n".join(lines) + "\n"


def apply_memory_baseline_pipe(pipe, ckpt: str) -> None:
    """Set pipe.* memory flags from ckpt path (no-op for non–memory-baseline paths)."""
    p = infer_memory_profile(ckpt)
    pipe.use_framepack_memory = bool(p.use_framepack_memory)
    pipe.context_temporal_decay = float(p.context_temporal_decay or 1.0)
    pipe.context_attention_weight = float(p.context_attention_weight or 1.0)
    pipe.use_framepack_length_compress = bool(p.use_framepack_length_compress)
    pipe.framepack_ratio = int(p.framepack_ratio or 2)
    pipe.framepack_length_strategy = str(p.framepack_length_strategy)
    pipe.framepack_multiscale_w2 = float(p.framepack_multiscale_w2)
    pipe.framepack_multiscale_w4 = float(p.framepack_multiscale_w4)
    pipe.use_spatial_memory = bool(p.use_spatial_memory)
    pipe.spatial_memory_tokens = int(p.spatial_memory_tokens or 64)
    if getattr(p, "spatial_memory_inject_mode", None):
        pipe.spatial_memory_inject_mode = str(p.spatial_memory_inject_mode)
    pipe.use_spatial_memory_legacy = bool(p.use_spatial_memory_legacy)
    pipe.use_geometry_spatial_memory = bool(p.use_geometry_spatial_memory)
    if p.geometry_spatial_memory_inject_mode:
        pipe.geometry_spatial_memory_inject_mode = str(
            p.geometry_spatial_memory_inject_mode
        )
    pipe.use_block_wise_ssm = bool(p.use_block_wise_ssm)
    pipe.block_wise_ssm_causal_v2 = bool(p.block_wise_ssm_causal_v2)
    pipe.use_videossm_hybrid = bool(p.use_videossm_hybrid)
    pipe.context_position = str(p.context_position or "suffix")
    pipe.training_context_source = p.training_context_source
    pipe.use_moc = bool(p.use_moc)
    if pipe.use_moc:
        from diffsynth.models.memory.mixture_of_contexts import MixtureOfContexts
        pipe.moc_module = MixtureOfContexts(temperature=float(p.moc_temperature))
    # 与 run_replay_loop_two_chunk 一致：宣称 SpatialGridMemory 但 ckpt 未载入模块时，避免 silent 与训练不一致
    if (
        pipe.use_spatial_memory
        and not pipe.use_spatial_memory_legacy
        and getattr(pipe, "spatial_memory_module", None) is None
    ):
        raise RuntimeError(
            "Spatial token-grid profile requested, but checkpoint has no "
            "spatial_memory_module weights. Refusing to replace the trained method "
            "with the legacy adaptive pool."
        )


def effective_context_frames(ckpt: str, env_context_frames: Optional[int]) -> int:
    """For shell helpers: default CONTEXT_FRAMES from env/int, else profile override, else 1."""
    prof = infer_memory_profile(ckpt)
    if env_context_frames is not None:
        return int(env_context_frames)
    if prof.context_override is not None:
        return int(prof.context_override)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Memory baseline CKPT path → MEM_ARGS / pipe flags")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("bash-export", help="Print eval-able bash for MEM_ARGS + CONTEXT_FRAMES_MEM_OVERRIDE")
    p_export.add_argument("ckpt")

    p_ctx = sub.add_parser("print-context", help="Print effective context_frames (int) for replay_gt-style scripts")
    p_ctx.add_argument("ckpt")
    p_ctx.add_argument(
        "--fallback",
        type=int,
        default=None,
        help="If no profile override, use this (else 1)",
    )
    sub.add_parser("list-profiles", help="Print memory profile registry JSON")

    args = ap.parse_args()
    if args.cmd == "bash-export":
        sys.stdout.write(bash_export(args.ckpt))
    elif args.cmd == "print-context":
        prof = infer_memory_profile(args.ckpt)
        v = prof.context_override if prof.context_override is not None else (args.fallback if args.fallback is not None else 1)
        print(int(v))
    elif args.cmd == "list-profiles":
        print(json.dumps(profile_registry_payload(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
