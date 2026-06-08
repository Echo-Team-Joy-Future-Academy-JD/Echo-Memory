import torch
import torch.nn.functional as F


def _compress_weights(ratio: int, strategy: str = "distance_merge", recent_keep_ratio: float = 0.5, device=None, dtype=None):
    if ratio <= 1:
        return None
    strategy = str(strategy or "distance_merge").lower()
    # Baseline-aligned default: non-overlapping mean pool on each r-frame group.
    if strategy in ("distance_merge", "mean", "uniform"):
        return None
    if strategy in ("recent_weighted", "weighted_recent"):
        # Optional weighted variant (kept for compatibility experiments).
        idx = torch.arange(ratio, device=device, dtype=dtype)
        w = (1.0 - float(recent_keep_ratio)) + float(recent_keep_ratio) * ((idx + 1.0) / float(ratio))
        w = w / w.sum()
        return w
    return torch.full((ratio,), 1.0 / float(ratio), device=device, dtype=dtype)


def framepack_length_compress_context_latents(
    context_latents: torch.Tensor,
    framepack_ratio: int,
    strategy: str = "distance_merge",
    recent_keep_ratio: float = 0.5,
    multiscale_w2: float = 0.25,
    multiscale_w4: float = 0.15,
):
    # context_latents: (B, C, K, H, W)
    if context_latents is None:
        return None, 0, 0, 0
    if context_latents.ndim != 5:
        raise ValueError(f"context_latents must be 5D (B,C,K,H,W), got {tuple(context_latents.shape)}")
    r = int(framepack_ratio)
    if r <= 1:
        k = int(context_latents.shape[2])
        return context_latents, k, k, k

    b, c, k_orig, h, w = context_latents.shape
    pad = (r - (k_orig % r)) % r
    if pad > 0:
        pad_lat = context_latents[:, :, -1:, :, :].repeat(1, 1, pad, 1, 1)
        context_latents = torch.cat([context_latents, pad_lat], dim=2)
    k_pad = int(context_latents.shape[2])
    new_k = k_pad // r

    grouped = context_latents.reshape(b, c, new_k, r, h, w)
    strategy = str(strategy or "distance_merge").lower()
    if strategy in ("packed_multiscale", "multiscale_packed", "multi_scale_packed"):
        base = grouped.mean(dim=3)

        # Base-code inspired approximation: aggregate history with extra low-res spatial views
        # (1x/2x/4x) and fuse back to the packed latent stream.
        x2 = F.avg_pool3d(context_latents, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x4 = F.avg_pool3d(context_latents, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        x2 = F.interpolate(x2, size=(k_pad, h, w), mode="trilinear", align_corners=False)
        x4 = F.interpolate(x4, size=(k_pad, h, w), mode="trilinear", align_corners=False)
        b2 = x2.reshape(b, c, new_k, r, h, w).mean(dim=3)
        b4 = x4.reshape(b, c, new_k, r, h, w).mean(dim=3)
        w2 = float(multiscale_w2 or 0.0)
        w4 = float(multiscale_w4 or 0.0)
        w1 = max(1e-6, 1.0 - w2 - w4)
        s = w1 + w2 + w4
        out = (w1 * base + w2 * b2 + w4 * b4) / s
    else:
        cw = _compress_weights(r, strategy=strategy, recent_keep_ratio=recent_keep_ratio, device=context_latents.device, dtype=context_latents.dtype)
        if cw is None:
            out = grouped.mean(dim=3)
        else:
            out = (grouped * cw.view(1, 1, 1, r, 1, 1)).sum(dim=3)
    return out, int(new_k), int(k_pad), int(k_orig)


def framepack_align_context_actions_to_latents(
    context_actions,
    K_orig_latent: int,
    K_after_pad: int,
    framepack_ratio: int,
    device=None,
    dtype=None,
    strategy: str = "distance_merge",
    recent_keep_ratio: float = 0.5,
):
    if context_actions is None:
        return None
    x = context_actions
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device, dtype=dtype or torch.float32)
    else:
        if device is not None:
            x = x.to(device=device)
        if dtype is not None:
            x = x.to(dtype=dtype)
    if x.ndim not in (2, 3):
        raise ValueError(f"context_actions must be 2D/3D, got shape {tuple(x.shape)}")
    r = int(framepack_ratio)
    if r <= 1:
        return x

    if x.ndim == 2:
        # (K, D)
        k, d = x.shape
        k_expected = int(K_orig_latent)
        if k < k_expected:
            raise ValueError(f"context_actions shorter than K_orig_latent: {k} < {k_expected}")
        x = x[:k_expected, :]
        pad = int(K_after_pad) - k_expected
        if pad > 0:
            x = torch.cat([x, x[-1:, :].repeat(pad, 1)], dim=0)
        new_k = int(K_after_pad) // r
        grouped = x.reshape(new_k, r, d)
        cw = _compress_weights(r, strategy=str(strategy or "distance_merge").lower(), recent_keep_ratio=recent_keep_ratio, device=x.device, dtype=x.dtype)
        return grouped.mean(dim=1) if cw is None else (grouped * cw.view(1, r, 1)).sum(dim=1)

    # (B, K, D)
    b, k, d = x.shape
    k_expected = int(K_orig_latent)
    if k < k_expected:
        raise ValueError(f"context_actions shorter than K_orig_latent: {k} < {k_expected}")
    x = x[:, :k_expected, :]
    pad = int(K_after_pad) - k_expected
    if pad > 0:
        x = torch.cat([x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)
    new_k = int(K_after_pad) // r
    grouped = x.reshape(b, new_k, r, d)
    cw = _compress_weights(r, strategy=str(strategy or "distance_merge").lower(), recent_keep_ratio=recent_keep_ratio, device=x.device, dtype=x.dtype)
    return grouped.mean(dim=2) if cw is None else (grouped * cw.view(1, 1, r, 1)).sum(dim=2)

