import torch


def apply_framepack_token_weights(
    x: torch.Tensor,
    num_context_frames: int,
    f: int,
    h: int,
    w: int,
    context_position: str = "prefix",
    use_framepack_memory: bool = False,
    context_temporal_decay: float = 1.0,
    context_attention_weight: float = 1.0,
):
    if x is None or x.ndim != 3:
        return x
    if not use_framepack_memory or int(num_context_frames) <= 0:
        return x
    b, n, d = x.shape
    f = int(f)
    if f <= 0 or n != f * int(h) * int(w):
        return x

    hw = int(h) * int(w)
    x4 = x.reshape(b, f, hw, d)
    k = min(int(num_context_frames), f)
    decay = float(context_temporal_decay)
    gain = float(context_attention_weight)
    if context_position == "suffix":
        ctx_start = f - k
        ctx_end = f
        # Suffix: first context frame is nearest boundary to target.
        distances = torch.arange(k, device=x.device, dtype=x.dtype)
    else:
        ctx_start = 0
        ctx_end = k
        # Prefix: last context frame is nearest boundary to target.
        distances = torch.arange(k - 1, -1, -1, device=x.device, dtype=x.dtype)

    weights = gain * torch.pow(torch.tensor(decay, device=x.device, dtype=x.dtype), distances)
    x4[:, ctx_start:ctx_end, :, :] = x4[:, ctx_start:ctx_end, :, :] * weights.view(1, k, 1, 1)
    return x4.reshape(b, n, d)

