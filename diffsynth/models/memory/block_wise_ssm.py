import torch
import torch.nn as nn


class BlockWiseStateSpaceMemory(nn.Module):
    """
    Paper-aligned block-wise recurrent SSM.

    This module is intentionally separate from VideoSSM hybrid. It performs a
    recurrent state update along the latent time axis for each spatial token
    trajectory, and is attached to selected DiT blocks.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.in_proj = nn.Linear(self.dim, self.dim * 2)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.decay_logit = nn.Parameter(torch.zeros(self.dim))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, f: int, **_kwargs):
        # x: (B, F*S, D), where S is spatial tokens per latent frame.
        if x is None or x.ndim != 3:
            return x
        b, n, d = x.shape
        f = int(f or 0)
        if d != self.dim or f <= 1 or n % f != 0:
            return x

        spatial = n // f
        x_seq = x.reshape(b, f, spatial, d).permute(0, 2, 1, 3).reshape(b * spatial, f, d)
        update, update_gate = self.in_proj(x_seq).chunk(2, dim=-1)
        update = torch.tanh(update)
        update_gate = torch.sigmoid(update_gate)
        decay = torch.sigmoid(self.decay_logit).to(dtype=x.dtype, device=x.device).view(1, d)

        state = torch.zeros(x_seq.shape[0], d, dtype=x.dtype, device=x.device)
        outputs = []
        for t in range(f):
            state = decay * state + (1.0 - decay) * update[:, t, :]
            outputs.append(state * update_gate[:, t, :])
        y = torch.stack(outputs, dim=1)
        y = self.out_proj(y)
        y = y.reshape(b, spatial, f, d).permute(0, 2, 1, 3).reshape(b, n, d)
        return x + torch.tanh(self.gate) * y
