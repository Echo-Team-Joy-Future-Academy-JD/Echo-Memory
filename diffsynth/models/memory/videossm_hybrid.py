import torch
import torch.nn as nn


class HybridStateSpaceMemory(nn.Module):
    """
    Lightweight legacy VideoSSM hybrid block:
    depthwise temporal conv over per-spatial token trajectories.
    """

    def __init__(self, dim: int, kernel_size: int = 3, expand: int = 2):
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        hidden = int(dim) * max(int(expand), 1)
        pad = self.kernel_size - 1  # causal-like left padding
        self.in_proj = nn.Linear(self.dim, hidden)
        self.dw = nn.Conv1d(hidden, hidden, kernel_size=self.kernel_size, groups=hidden, padding=pad)
        self.out_proj = nn.Linear(hidden, self.dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, f: int, h: int, w: int, **_kwargs):
        # x: (B, F*H*W, D)
        if x is None or x.ndim != 3:
            return x
        b, n, d = x.shape
        f = int(f)
        hw = int(h) * int(w)
        if d != self.dim or f <= 1 or n != f * hw:
            return x
        x4 = x.reshape(b, f, hw, d).permute(0, 2, 1, 3).reshape(b * hw, f, d)  # (B*HW, F, D)
        y = self.in_proj(x4)
        y = y.transpose(1, 2)  # (B*HW, hidden, F)
        y = self.dw(y)[..., :f]  # causal crop
        y = y.transpose(1, 2)
        y = self.out_proj(y)
        y = y.reshape(b, hw, f, d).permute(0, 2, 1, 3).reshape(b, n, d)
        return x + torch.tanh(self.gate) * y

