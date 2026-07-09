import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureOfContexts(nn.Module):
    """Reweight context frames by target-context similarity before DiT blocks."""

    def __init__(self, temperature: float = 1.0, top_k: int = 0):
        super().__init__()
        self.temperature = float(temperature or 1.0)
        self.top_k = int(top_k or 0)

    def forward(
        self,
        x: torch.Tensor,
        num_context_frames: int,
        f: int,
        h: int,
        w: int,
        context_position: str = "prefix",
    ) -> torch.Tensor:
        if x is None or x.ndim != 3:
            return x
        b, n, d = x.shape
        f = int(f)
        h = int(h)
        w = int(w)
        k = min(int(num_context_frames), f)
        hw = h * w
        if k <= 0 or f <= k or hw <= 0 or n != f * hw:
            return x

        x4 = x.reshape(b, f, hw, d)
        if str(context_position).lower() == "suffix":
            target = x4[:, : f - k]
            context = x4[:, f - k :]
            context_slice = (slice(None), slice(f - k, f))
        else:
            context = x4[:, :k]
            target = x4[:, k:]
            context_slice = (slice(None), slice(0, k))
        if target.numel() == 0 or context.numel() == 0:
            return x

        query = F.normalize(target.mean(dim=(1, 2)), dim=-1)  # (B, D)
        keys = F.normalize(context.mean(dim=2), dim=-1)  # (B, K, D)
        logits = torch.einsum("bd,bkd->bk", query, keys)
        temperature = max(float(self.temperature), 1e-6)
        logits = logits / temperature

        if self.top_k > 0 and self.top_k < k:
            keep = min(self.top_k, k)
            top_values, top_indices = torch.topk(logits, k=keep, dim=-1)
            masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
            logits = masked.scatter(dim=-1, index=top_indices, src=top_values)

        weights = torch.softmax(logits, dim=-1).to(dtype=x4.dtype)
        # Preserve average context magnitude while making the selected chunks dominant.
        weights = weights * float(k)
        x4 = x4.clone()
        x4[context_slice] = context * weights.view(b, k, 1, 1)
        return x4.reshape(b, n, d)

