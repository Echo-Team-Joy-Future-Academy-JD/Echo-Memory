"""FramePack-Length latent/RT alignment (e.g. K=5, r=4). Run: PYTHONPATH=. python3 tests/test_framepack_memory_align.py"""
import torch

from diffsynth.models.memory.framepack_length import (
    framepack_align_context_actions_to_latents,
    framepack_length_compress_context_latents,
)
from diffsynth.models.memory.framepack_weight import apply_framepack_token_weights


def test_k5_r4_latent_and_actions():
    B, C, H, W = 1, 16, 8, 8
    K = 5
    r = 4
    lat = torch.randn(B, C, K, H, W)
    out, new_k, K_pad, K_orig = framepack_length_compress_context_latents(lat, r)
    assert K_orig == 5
    pad = (r - (K % r)) % r
    assert K_pad == K + pad == 8
    assert new_k == 2
    assert out.shape[2] == 2
    ca = torch.randn(K, 12)
    aligned = framepack_align_context_actions_to_latents(
        ca, K_orig, K_pad, r, device=lat.device, dtype=lat.dtype
    )
    assert aligned.shape == (2, 12)


def test_framepack_weight_preserves_shape_suffix():
    D = 64
    f, h, w = 5, 2, 2
    num_ctx = 2
    N = f * h * w
    x = torch.randn(1, N, D)
    y = apply_framepack_token_weights(
        x,
        num_context_frames=num_ctx,
        f=f,
        h=h,
        w=w,
        context_position="suffix",
        use_framepack_memory=True,
        context_temporal_decay=0.9,
        context_attention_weight=1.0,
    )
    assert y.shape == x.shape


if __name__ == "__main__":
    test_k5_r4_latent_and_actions()
    test_framepack_weight_preserves_shape_suffix()
    print("test_framepack_memory_align: ok")
