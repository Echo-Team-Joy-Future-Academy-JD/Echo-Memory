import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsynth.models.memory.block_wise_ssm import BlockWiseStateSpaceMemory
from diffsynth.models.memory.videossm_hybrid import HybridStateSpaceMemory
from diffsynth.models.wan_video_dit import SelfAttention, CrossAttention, GateModule, modulate


class MLP_Action(nn.Module):
    def __init__(self, out_dim, sliding_window_size=3, r=4):
        super().__init__()
        self.proj_action = nn.Linear(r * sliding_window_size * 10, out_dim)
        nn.init.zeros_(self.proj_action.weight)
        nn.init.zeros_(self.proj_action.bias)
        self.sliding_window_size = sliding_window_size
        self.r = r

    def forward(self, x):
        bs, nr, act_dim = x.shape
        r = self.r
        n = nr // r
        actions = x.reshape(bs, n, r, act_dim)
        actions = F.pad(actions, (0, 0, 0, 0, self.sliding_window_size - 1, 1), mode="replicate")
        action_windows = []
        for i in range(self.sliding_window_size):
            action_windows.append(actions[:, i:i + n + 1])
        actions = torch.cat(action_windows, dim=2)
        actions = actions.reshape(bs, n + 1, -1)
        actions = self.proj_action(actions)
        return actions


class MLP_CamPose(nn.Module):
    def __init__(self, out_dim, pose_dim=12):
        super().__init__()
        self.proj = nn.Linear(pose_dim, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class DiTBlock_w_Action(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int,
                 eps: float = 1e-6, add_action_attn=False,
                 action_use_temporal_attention: bool = True, use_cam_pose: bool = False,
                 use_block_wise_ssm: bool = False, use_videossm_hybrid: bool = False,
                 videossm_kernel_size: int = 3, videossm_expand: int = 2):
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
            self.action_mlp = MLP_Action(dim)

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

