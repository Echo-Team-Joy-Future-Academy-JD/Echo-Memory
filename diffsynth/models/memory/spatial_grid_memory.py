import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGridMemory(nn.Module):
    def __init__(self, dim: int, grid_size: int = 8, num_tokens: int = 64):
        super().__init__()
        self.dim = int(dim)
        self.grid_size = int(grid_size)
        self.num_tokens = int(num_tokens)
        g2 = self.grid_size * self.grid_size
        # Keep key name aligned with ckpt loading in loop_utils.py (spatial_to_tokens).
        self.spatial_to_tokens = nn.Parameter(torch.zeros(g2, self.num_tokens))
        nn.init.normal_(self.spatial_to_tokens, std=0.02)

    @property
    def mix(self):
        # Backward compatibility for code that referenced the old attribute name.
        return self.spatial_to_tokens

    def forward(self, x_context: torch.Tensor, num_context_frames: int, h: int, w: int):
        # x_context: (B, K*H*W, D)
        if x_context is None or x_context.ndim != 3:
            return x_context
        b, n, d = x_context.shape
        if d != self.dim:
            raise ValueError(f"SpatialGridMemory dim mismatch: x={d} module={self.dim}")
        k = max(int(num_context_frames), 1)
        spatial = int(h) * int(w)
        if n != k * spatial:
            # Best effort fallback: treat x as a flat token map and pool directly.
            x_mean = x_context
        else:
            x_mean = x_context.reshape(b, k, spatial, d).mean(dim=1)  # (B, S, D)

        g2 = self.grid_size * self.grid_size
        pooled = F.adaptive_avg_pool1d(x_mean.transpose(1, 2), g2).transpose(1, 2)  # (B, G2, D)
        mix = torch.softmax(self.spatial_to_tokens, dim=0)  # (G2, M)
        mem = torch.einsum("bgd,gm->bmd", pooled, mix)  # (B, M, D)
        return mem

    def load_state_dict(self, state_dict, strict: bool = True):
        # Compatibility:
        # - old local key: mix
        # - current/baseline key: spatial_to_tokens
        sd = dict(state_dict)
        if "mix" in sd and "spatial_to_tokens" not in sd:
            sd["spatial_to_tokens"] = sd.pop("mix")
        # Ignore deprecated projection keys from prior experiments.
        sd.pop("out.weight", None)
        sd.pop("out.bias", None)
        return super().load_state_dict(sd, strict=False if not strict else strict)


class SpatialCrossAttnReadout(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=int(dim), num_heads=int(num_heads), batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_target: torch.Tensor, mem_tokens: torch.Tensor):
        if x_target is None or mem_tokens is None:
            return x_target
        if x_target.numel() == 0 or mem_tokens.numel() == 0:
            return x_target
        delta, _ = self.attn(x_target, mem_tokens, mem_tokens, need_weights=False)
        return x_target + torch.tanh(self.gate) * delta


def apply_spatial_cross_attn_readout(x_target: torch.Tensor, mem_tokens: torch.Tensor, module: nn.Module = None):
    if module is None:
        module = SpatialCrossAttnReadout(dim=int(x_target.shape[-1]), num_heads=8).to(device=x_target.device, dtype=x_target.dtype)
    return module(x_target, mem_tokens)


def inject_spatial_memory(context: torch.Tensor, mem_tokens: torch.Tensor, mode: str = "concat_text"):
    mode = str(mode or "concat_text").lower()
    if mem_tokens is None or mode == "none":
        return context
    if context is None:
        return mem_tokens
    if mode in ("concat_text", "cross_attn_readout"):
        return torch.cat([context, mem_tokens], dim=1)
    return context

