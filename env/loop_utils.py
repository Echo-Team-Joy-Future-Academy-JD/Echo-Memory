"""
推理工具：供 run_replay_loop_two_chunk 及评估脚本使用。
提供：load_pipeline_and_ckpt、load_prompt_for_video、sample_trajectory_samples_from_dataset。

VWM-style 简化版：使用 DiTBlock_w_Action + MLP_CamPose（block 内 action_mlp），
去除 CameraEncoder / camera_encoder_shallow 等冗余路径。
"""

import os
import re
import sys
import csv
import random

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_script_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import SelfAttention, CrossAttention, GateModule, modulate
from diffsynth.models.memory.block_wise_ssm import BlockWiseStateSpaceMemory
from diffsynth.models.memory.videossm_hybrid import HybridStateSpaceMemory

DEFAULT_NEGATIVE_PROMPT = "oversaturated colors, overexposed, static, blurry details"


# ── MLP_CamPose + DiTBlock_w_Action（与训练侧 train.py 完全一致）──────────
class MLP_CamPose(nn.Module):
    def __init__(self, out_dim, pose_dim=12):
        super().__init__()
        self.proj = nn.Linear(pose_dim, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class DiTBlock_w_Action(nn.Module):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6,
                 add_action_attn=False, action_use_temporal_attention=True,
                 use_cam_pose=False, use_block_wise_ssm=False, use_videossm_hybrid=False,
                 videossm_kernel_size=3, videossm_expand=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        if add_action_attn:
            self.self_attn_with_action = SelfAttention(dim, num_heads, eps)
            nn.init.zeros_(self.self_attn_with_action.o.weight)
            nn.init.zeros_(self.self_attn_with_action.o.bias)
        if use_cam_pose:
            self.action_mlp = MLP_CamPose(dim)
        else:
            self.action_mlp = MLP_CamPose(dim)
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.action_use_temporal_attention = action_use_temporal_attention
        self.use_block_wise_ssm = bool(use_block_wise_ssm)
        self.use_videossm_hybrid = bool(use_videossm_hybrid)
        if use_block_wise_ssm:
            self.block_wise_ssm = BlockWiseStateSpaceMemory(dim)
        if use_videossm_hybrid:
            self.videossm_hybrid = HybridStateSpaceMemory(
                dim, kernel_size=videossm_kernel_size, expand=videossm_expand
            )

    def forward(self, x, context, t_mod, freqs, actions=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        num_frames = None
        if actions is not None:
            original_x = x
            actions = self.action_mlp(actions.to(x.dtype)).to(x.dtype)
            bs, num_frames, dim = actions.shape
            actions = actions.reshape(bs, num_frames, 1, dim)
            x = x.reshape(bs, num_frames, -1, dim)
            x = x + actions
            if hasattr(self, "self_attn_with_action"):
                if not self.action_use_temporal_attention:
                    x = x.reshape(bs, -1, dim)
                    x = original_x + self.self_attn_with_action(x, freqs)
                else:
                    from einops import rearrange
                    x = rearrange(x, "b f p d -> (b p) f d")
                    attn_out = self.self_attn_with_action(x)
                    attn_out = rearrange(attn_out, "(b p) f d -> b f p d", b=bs)
                    x = original_x + attn_out.reshape(bs, -1, dim)
            else:
                x = x.reshape(bs, -1, dim)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        if num_frames is not None:
            if hasattr(self, "block_wise_ssm"):
                x = self.block_wise_ssm(x, f=num_frames)
            if hasattr(self, "videossm_hybrid"):
                spatial = x.shape[1] // int(num_frames) if int(num_frames) > 0 else 0
                x = self.videossm_hybrid(x, f=num_frames, h=1, w=spatial)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


# ── Utility functions ─────────────────────────────────────────────────────

def load_pose_rt(json_file, frame_idx):
    """从数据集 camera json 读取单帧 12 维 RT。"""
    from src.model_training.fov_retrieval import load_camera_pose, pose_to_rt
    pose = load_camera_pose(json_file, int(frame_idx))
    if pose is None:
        return None
    return pose_to_rt(pose, constrain_to_xy=True)


def get_relative_rt(rt, ref_rt):
    """单帧相对位姿。"""
    from src.model_training.fov_retrieval import convert_rt_to_relative
    if rt is None or ref_rt is None or len(rt) < 12 or len(ref_rt) < 12:
        return None
    out = convert_rt_to_relative([rt], ref_rt)
    return out[0] if out else None


def load_prompt_for_video(dataset_base, video_name):
    """从 dataset 目录下的 metadata CSV 读取该视频的 prompt。"""
    if not dataset_base or not video_name:
        return None
    vn = str(video_name).replace(".mp4", "").replace(".avi", "").strip()
    for name in ("metadata_full.csv", "metadata.csv", "prompts.csv"):
        path = os.path.join(dataset_base, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("video_name", "").strip() == vn:
                        p = row.get("prompt", "").strip()
                        if p:
                            return p
        except Exception:
            pass
    return None


def sample_trajectory_samples_from_dataset(dataset_base, num_samples=4, num_frames=81, seed=42):
    """从 dataset 枚举 (video_name, start_frame)。"""
    frames_dir = os.path.join(dataset_base, "frames")
    if not os.path.isdir(frames_dir):
        return []
    candidates = []
    for vn in sorted(os.listdir(frames_dir)):
        vd = os.path.join(frames_dir, vn)
        if not os.path.isdir(vd):
            continue
        try:
            names = [f for f in os.listdir(vd) if f.endswith(".png")]
            indices = sorted({int(os.path.splitext(n)[0]) for n in names if n[:-4].isdigit()})
            if not indices:
                continue
            max_idx = max(indices)
            for start in indices:
                if start + num_frames - 1 <= max_idx:
                    candidates.append((vn, start))
        except Exception:
            continue
    if not candidates:
        return []
    rng = random.Random(seed)
    if len(candidates) <= num_samples:
        return candidates
    return [candidates[i] for i in rng.sample(range(len(candidates)), num_samples)]


# ── Pipeline loading (VWM-style) ──────────────────────────────────────────

def _build_action_blocks(
    pipe,
    add_action_attn=False,
    action_use_temporal_attention=True,
    block_wise_block_ids=None,
    videossm_block_ids=None,
):
    """Replace DiT blocks with DiTBlock_w_Action (VWM cam_infer.py style)."""
    dit = pipe.dit
    old_blocks = dit.blocks
    has_image_input = getattr(dit, "has_image_input", False)
    dim = dit.dim
    num_heads = getattr(dit, "num_heads", None) or getattr(old_blocks[0], "num_heads", None)
    ffn_dim = getattr(dit, "ffn_dim", None) or getattr(old_blocks[0], "ffn_dim", None)
    eps = getattr(dit, "eps", 1e-6)

    block_dtype = next(old_blocks[0].parameters()).dtype
    block_device = next(old_blocks[0].parameters()).device

    block_wise_block_ids = set(block_wise_block_ids or [])
    videossm_block_ids = set(videossm_block_ids or [])

    new_blocks = torch.nn.ModuleList()
    for block_id, old_block in enumerate(old_blocks):
        new_block = DiTBlock_w_Action(
            has_image_input=has_image_input,
            dim=dim, num_heads=num_heads, ffn_dim=ffn_dim, eps=eps,
            add_action_attn=add_action_attn,
            action_use_temporal_attention=action_use_temporal_attention,
            use_cam_pose=True,
            use_block_wise_ssm=block_id in block_wise_block_ids,
            use_videossm_hybrid=block_id in videossm_block_ids,
        )
        new_block = new_block.to(dtype=block_dtype, device=block_device)
        for attr in ("self_attn", "cross_attn", "norm1", "norm2", "norm3", "ffn"):
            if hasattr(old_block, attr) and hasattr(new_block, attr):
                getattr(new_block, attr).load_state_dict(getattr(old_block, attr).state_dict())
        if hasattr(old_block, "modulation") and hasattr(new_block, "modulation"):
            with torch.no_grad():
                new_block.modulation.copy_(old_block.modulation.to(dtype=block_dtype))
        new_blocks.append(new_block)

    dit.blocks = new_blocks
    print(f"[loop_utils] Replaced {len(new_blocks)} blocks with DiTBlock_w_Action (MLP_CamPose)")
    if block_wise_block_ids:
        print(f"[loop_utils] Loaded Block-wise SSM slots on blocks: {sorted(block_wise_block_ids)[:8]}{'...' if len(block_wise_block_ids) > 8 else ''}")
    if videossm_block_ids:
        print(f"[loop_utils] Loaded VideoSSM hybrid slots on blocks: {sorted(videossm_block_ids)[:8]}{'...' if len(videossm_block_ids) > 8 else ''}")


def load_pipeline_and_ckpt(
    ckpt_path,
    dit_path,
    text_encoder_path,
    vae_path,
    device="cuda",
    add_action_attn=False,
    action_use_temporal_attention=True,
    tokenizer_path=None,
    # Legacy kwargs accepted but ignored (CameraEncoder removed)
    **kwargs,
):
    """Load WanVideoPipeline, replace blocks with DiTBlock_w_Action, load ckpt (strict=False).

    VWM-style: no CameraEncoder, no complex inference logic. Action is injected
    via MLP_CamPose (nn.Linear(12, dim), zero-init) inside each DiTBlock_w_Action.
    """
    print(f"[loop_utils] Loading pipeline (DiT -> {device})")
    if not tokenizer_path:
        import os as _os
        _base = _os.path.dirname(dit_path)
        _cand = _os.path.join(_base, "google", "umt5-xxl")
        if _os.path.isdir(_cand):
            tokenizer_path = _cand
            print(f"[loop_utils] Auto-detected tokenizer at {tokenizer_path}")
    model_configs = [
        ModelConfig(path=dit_path, offload_device=device),
        ModelConfig(path=text_encoder_path, offload_device="cpu"),
        ModelConfig(path=vae_path, offload_device="cpu"),
    ]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=tokenizer_path) if tokenizer_path else None,
    )

    ckpt = None
    block_wise_block_ids = set()
    videossm_block_ids = set()
    action_attn_block_ids = set()
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = safe_load_file(ckpt_path)
        for key in ckpt.keys():
            m = re.match(r"blocks\.(\d+)\.block_wise_ssm\.", key)
            if m:
                block_wise_block_ids.add(int(m.group(1)))
            m = re.match(r"blocks\.(\d+)\.videossm_hybrid\.", key)
            if m:
                videossm_block_ids.add(int(m.group(1)))
            m = re.match(r"blocks\.(\d+)\.self_attn_with_action\.", key)
            if m:
                action_attn_block_ids.add(int(m.group(1)))

    if action_attn_block_ids and not add_action_attn:
        add_action_attn = True
        print("[loop_utils] Detected action-attention weights in checkpoint; enabling self_attn_with_action")

    # Replace blocks with DiTBlock_w_Action, including memory slots implied by ckpt keys.
    _build_action_blocks(
        pipe,
        add_action_attn=add_action_attn,
        action_use_temporal_attention=action_use_temporal_attention,
        block_wise_block_ids=block_wise_block_ids,
        videossm_block_ids=videossm_block_ids,
    )

    # Load ckpt (strict=False: base model keys match, action_mlp keys are extra)
    if ckpt_path and not os.path.isfile(ckpt_path):
        print(f"[loop_utils] WARNING: ckpt not found: {ckpt_path} — running with base model weights only!")
    if ckpt_path and os.path.isfile(ckpt_path):
        if ckpt is None:
            ckpt = safe_load_file(ckpt_path)
        missing, unexpected = pipe.dit.load_state_dict(ckpt, strict=False)
        action_keys = [k for k in ckpt if "action_mlp" in k]
        if not missing and not unexpected:
            print(f"[loop_utils] Ckpt loaded: {len(ckpt)} keys, perfect match")
        else:
            print(f"[loop_utils] Ckpt loaded: {len(ckpt)} keys, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}, "
                  f"action_mlp_keys={len(action_keys)}")
            if missing:
                for k in sorted(missing)[:5]:
                    print(f"  missing: {k}")
            if unexpected:
                for k in sorted(unexpected)[:5]:
                    print(f"  unexpected: {k}")

        # Optional: load SpatialGridMemory if present in ckpt
        _smsd = {
            k.replace("spatial_memory_module.", "", 1): v
            for k, v in ckpt.items()
            if k.startswith("spatial_memory_module.")
        }
        if _smsd:
            try:
                from diffsynth.models.spatial_grid_memory import SpatialGridMemory
            except ImportError:
                SpatialGridMemory = None
            if SpatialGridMemory is not None:
                dim = pipe.dit.dim
                w = _smsd.get("spatial_to_tokens")
                if w is not None:
                    g2, num_tok = int(w.shape[0]), int(w.shape[1])
                    gsz = int(round(g2 ** 0.5))
                    if gsz * gsz != g2:
                        gsz = 8
                    sm = SpatialGridMemory(dim, grid_size=gsz, num_tokens=num_tok)
                    sm.load_state_dict(_smsd, strict=False)
                    sm = sm.to(dtype=next(pipe.dit.parameters()).dtype, device=next(pipe.dit.parameters()).device)
                    pipe.spatial_memory_module = sm
                    pipe.use_spatial_memory_legacy = False
                    print(f"[loop_utils] Loaded spatial_memory_module (grid={gsz}, tokens={num_tok})")

    if getattr(pipe, "enable_vram_management", None):
        pipe.enable_vram_management()
    return pipe