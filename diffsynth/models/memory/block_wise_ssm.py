import torch
import torch.nn as nn


class BlockWiseStateSpaceMemory(nn.Module):
    """
    Paper-aligned block-wise recurrent SSM.

    This module is intentionally separate from VideoSSM hybrid. It performs a
    recurrent state update along the latent time axis for each spatial token
    trajectory, and is attached to selected DiT blocks.
    """

    def __init__(
        self,
        dim: int,
        causal_v2: bool = False,
        min_residual_scale: float = 0.1,
        max_residual_scale: float = 0.2,
    ):
        super().__init__()
        self.dim = int(dim)
        self.causal_v2 = bool(causal_v2)
        self.min_residual_scale = float(min_residual_scale)
        self.max_residual_scale = float(max_residual_scale)
        self.in_proj = nn.Linear(self.dim, self.dim * 2)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.decay_logit = nn.Parameter(torch.zeros(self.dim))
        if self.causal_v2:
            self.norm = nn.LayerNorm(self.dim)
            # Keep the recurrent contribution representable under BF16 updates.
            self.residual_logit = nn.Parameter(torch.zeros(1))
        else:
            self.gate = nn.Parameter(torch.zeros(1))
        self.last_stats = {}

    def forward(
        self,
        x: torch.Tensor,
        f: int,
        *,
        num_context_frames: int = 0,
        context_position: str = "suffix",
        **_kwargs,
    ):
        # x: (B, F*S, D), where S is spatial tokens per latent frame.
        if x is None or x.ndim != 3:
            return x
        b, n, d = x.shape
        f = int(f or 0)
        if d != self.dim or f <= 1 or n % f != 0:
            return x
        if self.causal_v2:
            if str(context_position).lower() != "prefix":
                raise ValueError(
                    "Causal Block-SSM v2 requires context_position='prefix'"
                )
            if int(num_context_frames or 0) <= 0 or int(num_context_frames) >= f:
                raise ValueError(
                    "Causal Block-SSM v2 requires 0 < num_context_frames < f; "
                    f"got context={num_context_frames}, f={f}"
                )

        spatial = n // f
        x_seq = x.reshape(b, f, spatial, d).permute(0, 2, 1, 3).reshape(b * spatial, f, d)
        ssm_input = self.norm(x_seq) if self.causal_v2 else x_seq
        update, update_gate = self.in_proj(ssm_input).chunk(2, dim=-1)
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
        if self.causal_v2:
            floor = min(max(self.min_residual_scale, 0.0), 1.0)
            ceiling = min(max(self.max_residual_scale, floor), 1.0)
            residual_scale = floor + (ceiling - floor) * torch.sigmoid(
                self.residual_logit
            )
            self.last_stats = {
                "residual_scale": residual_scale.detach(),
                "decay_mean": decay.detach().mean(),
                "state_norm": state.detach().float().norm(dim=-1).mean(),
                "contribution_norm": y.detach().float().norm(dim=-1).mean(),
            }
            return x + residual_scale.to(dtype=x.dtype) * y
        return x + torch.tanh(self.gate) * y
