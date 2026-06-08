"""
Camera Encoder for RT (Rotation-Translation) matrix injection.
CAM paper [2506.03141]: Maps camera pose to DiT hidden dimension for spatial attention conditioning.

Design notes:
- Primary purpose: dimension alignment (action/RT [12] -> DiT hidden D). One-layer MLP is sufficient; no need for deeper encoder.
- Per-frame MLP: each frame's RT [12] -> hidden_size independently (no temporal context).
- Optional zero-init scale: conditioning starts weak (scale=0) and grows with training for stability.
- shallow: single Linear(12, D) to match CAM "single-layer MLP" wording (ablation). When shallow=True
  and separate_t_r=False, the encoder is exactly one layer (merged MLP): RT [12] -> D.
- separate_t_r: encode translation (t) and rotation (R) with separate MLPs then add, for scale balance.
  Use shallow=True and separate_t_r=False for merged single-layer MLP (RT not split).
- explicit_yaw: add a signed yaw scalar branch (Z-only) so CW/CCW are explicitly encoded; helps when
  the model is insensitive to rotation direction (Zhou et al. CVPR 2019, sign continuity).
- sincos_yaw: add [cos(yaw), sin(yaw)] branch (2D) for direction; sin carries sign explicitly.

Design limitations / caveats:
- No input normalization: t (translation) and R (rotation) have different scales; one Linear(12,D) may be
  sensitive to units. Caller should use consistent RT scale or relative RT; optional input LayerNorm not implemented.
- Per-frame only: no temporal context (each frame encoded independently). Fine for CAM ablation; temporal modeling not supported.
- 16-dim input: layout for flattened 4x4 is unspecified; yaw branches are disabled. Prefer 12-dim in practice.
- explicit_yaw and sincos_yaw can both be True (redundant encoding of yaw); usually use one.
"""

import torch
import torch.nn as nn
from typing import Optional

# For Z-only rotation: yaw = atan2(R_21, R_11); R is row-major [R_11,R_12,R_13, R_21,...]
def _yaw_from_rt_12(rt: torch.Tensor) -> torch.Tensor:
    """rt [..., 12] -> yaw in [-1, 1] (normalized by pi)."""
    R11 = rt[..., 3]
    R21 = rt[..., 6]
    yaw_rad = torch.atan2(R21, R11)
    return yaw_rad / 3.141592653589793


def _sincos_yaw_from_rt_12(rt: torch.Tensor) -> torch.Tensor:
    """rt [..., 12] -> [..., 2] (cos(yaw), sin(yaw))."""
    R11 = rt[..., 3]
    R21 = rt[..., 6]
    yaw_rad = torch.atan2(R21, R11)
    return torch.stack([torch.cos(yaw_rad), torch.sin(yaw_rad)], dim=-1)


class CameraEncoder(nn.Module):
    """
    Encode RT matrices (camera pose) to DiT hidden dimension.
    
    Input: rt_matrices [B, F, 12] or [B, F, 16]
      - 12: [t_x, t_y, t_z, R_11..R_33] (3 translation + 9 rotation), R row-major. No input normalization:
        t and R often differ in scale (e.g. t in meters, R in [-1,1]); single Linear(12,D) may be sensitive to units.
      - 16: 4x4 matrix flattened (layout/order unspecified; yaw branches disabled when rt_dim=16).
    Output: camera_emb [B, F, D] where D = hidden_size (scaled by learnable scale, default 0-init).
    """
    
    def __init__(
        self,
        rt_dim: int = 12,
        hidden_size: int = 5120,
        mlp_hidden_mult: int = 4,
        eps: float = 1e-6,
        zero_init_scale: bool = False,
        full_zero_init: bool = False,
        shallow: bool = False,
        separate_t_r: bool = False,
        explicit_yaw: bool = False,
        sincos_yaw: bool = False,
        conditioning_scale: float = 1.0,
        r_mlp_no_layernorm: bool = False,
    ):
        super().__init__()
        self.rt_dim = rt_dim
        self.hidden_size = hidden_size
        self.zero_init_scale = zero_init_scale
        self.full_zero_init = full_zero_init
        self.shallow = shallow
        self.separate_t_r = separate_t_r
        self.explicit_yaw = explicit_yaw and rt_dim == 12
        self.sincos_yaw = sincos_yaw and rt_dim == 12
        self.conditioning_scale = float(conditioning_scale)
        self.r_mlp_no_layernorm = r_mlp_no_layernorm and separate_t_r
        dtype = torch.get_default_dtype()
        
        if separate_t_r:
            # Plan B: separate t (3) and R (9) encoders for scale balance; only for rt_dim=12.
            assert rt_dim == 12, "separate_t_r only supported for rt_dim=12"
            mid = max(hidden_size // 2, 256)
            self.t_mlp = nn.Sequential(
                nn.Linear(3, mid),
                nn.LayerNorm(mid, eps=eps),
                nn.GELU(),
                nn.Linear(mid, hidden_size),
                nn.LayerNorm(hidden_size, eps=eps),
            )
            if r_mlp_no_layernorm:
                # No LayerNorm on R so sign of R_12/R_21 (yaw direction) is not normalized away.
                self.r_mlp = nn.Sequential(
                    nn.Linear(9, mid),
                    nn.GELU(),
                    nn.Linear(mid, hidden_size),
                )
            else:
                self.r_mlp = nn.Sequential(
                    nn.Linear(9, mid),
                    nn.LayerNorm(mid, eps=eps),
                    nn.GELU(),
                    nn.Linear(mid, hidden_size),
                    nn.LayerNorm(hidden_size, eps=eps),
                )
            self.mlp = None
        elif shallow:
            # Merged single-layer MLP: one Linear(rt_dim, hidden_size), no separate t/R.
            self.mlp = nn.Linear(rt_dim, hidden_size)
            assert isinstance(self.mlp, nn.Linear), "shallow path must be exactly one Linear layer"
            if full_zero_init:
                nn.init.zeros_(self.mlp.weight)
                nn.init.zeros_(self.mlp.bias)
        else:
            mid_dim = hidden_size * mlp_hidden_mult
            self.mlp = nn.Sequential(
                nn.Linear(rt_dim, mid_dim),
                nn.LayerNorm(mid_dim, eps=eps),
                nn.GELU(),
                nn.Linear(mid_dim, mid_dim),
                nn.LayerNorm(mid_dim, eps=eps),
                nn.GELU(),
                nn.Linear(mid_dim, hidden_size),
                nn.LayerNorm(hidden_size, eps=eps),
            )
        
        if self.explicit_yaw:
            self.yaw_embed = nn.Linear(1, hidden_size)
        else:
            self.yaw_embed = None
        if self.sincos_yaw:
            self.sincos_embed = nn.Linear(2, hidden_size)
        else:
            self.sincos_embed = None
        
        # Learnable scale: when zero_init_scale=True, init to 0 so conditioning grows with training (stable).
        # When full_zero_init=True, skip scale (GF-ICL style: Linear output directly, no extra scale).
        if full_zero_init:
            self.scale = None  # no scale, use 1.0 in forward
        else:
            self.scale = nn.Parameter(torch.zeros(1) if zero_init_scale else torch.ones(1))

    def is_single_layer_merged(self) -> bool:
        """True if encoder is exactly one Linear(12, D) with no separate t/R (merged MLP)."""
        return self.shallow and not self.separate_t_r and self.mlp is not None and isinstance(self.mlp, nn.Linear)

    def forward(self, rt_matrices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rt_matrices: [B, F, 12] or [B, F, 16]
        Returns:
            camera_emb: [B, F, hidden_size], scaled by self.scale.
        """
        d = rt_matrices.dtype
        if self.separate_t_r:
            t = rt_matrices[..., :3].to(d)
            r = rt_matrices[..., 3:12].to(d)
            out = self.t_mlp(t) + self.r_mlp(r)
        else:
            out = self.mlp(rt_matrices.to(d))
        if self.yaw_embed is not None and rt_matrices.shape[-1] >= 12:
            yaw_norm = _yaw_from_rt_12(rt_matrices[..., :12]).unsqueeze(-1).to(d)
            out = out + self.yaw_embed(yaw_norm)
        if self.sincos_embed is not None and rt_matrices.shape[-1] >= 12:
            sincos = _sincos_yaw_from_rt_12(rt_matrices[..., :12]).to(d)
            out = out + self.sincos_embed(sincos)
        scale = self.scale.to(d) if self.scale is not None else torch.ones(1, device=out.device, dtype=out.dtype)
        return out * scale * self.conditioning_scale


def expand_camera_emb_to_tokens(
    camera_emb: torch.Tensor,
    num_frames: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Expand per-frame camera_emb [B, F, D] to per-token [B, N, D]
    where N = F * h * w (tokens ordered as frame0_all_patches, frame1_all_patches, ...).

    Dimension alignment (与 DiT patchify 一致):
    - Encoder 输出: 每帧一个向量 [B, F, D]，即相当于 [B, F, 1, D]（F 帧每帧 1 个 embedding）。
    - 对齐方式: 在空间维上把该 1 重复 H×W 次，得到 [B, F, h*w, D]，再展平为 [B, F*h*w, D]。
    - Token 顺序: frame0 的 h*w 个 token 共用 frame0 的 camera_emb，frame1 的 h*w 个 token 共用 frame1 的 camera_emb，与
      wan_video_dit patchify 的 rearrange(..., 'b c f h w -> b (f h w) c') 顺序一致（帧优先，再空间）。

    Args:
        camera_emb: [B, F, D]
        num_frames: F (must equal camera_emb.shape[1]; used for assertion only).
        h, w: spatial grid (patches per frame)
    Returns:
        [B, F*h*w, D]
    """
    B, F, D = camera_emb.shape
    if F != num_frames:
        raise ValueError(f"expand_camera_emb_to_tokens: camera_emb has F={F}, num_frames={num_frames}")
    # [B, F, D] -> [B, F, 1, D] (每帧 1 个) -> expand 到 [B, F, h*w, D] (每帧重复 H×W 次) -> [B, F*h*w, D]
    return camera_emb.unsqueeze(2).expand(B, F, h * w, D).reshape(B, F * h * w, D)
