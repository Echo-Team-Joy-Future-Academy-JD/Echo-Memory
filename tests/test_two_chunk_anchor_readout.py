"""Two-chunk action/context alignment + spatial readout sanity tests.

Run:
  PYTHONPATH=. python3 tests/test_two_chunk_anchor_readout.py
"""
import json
import os
import tempfile

import torch

from diffsynth.models.memory.spatial_grid_memory import SpatialCrossAttnReadout, apply_spatial_cross_attn_readout
from src.model_training.multichunk_sample_utils import _load_actions_tensor_from_json, _tail_context_actions


def test_tail_context_actions_from_left45():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "action_rotation_left_45.json")
        data = {
            str(i): [0.0, 0.0, 0.0, 1.0 - 1e-4 * i, -0.01 * i, 0.0, 0.01 * i, 1.0 - 1e-4 * i, 0.0, 0.0, 0.0, 1.0]
            for i in range(81)
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
        acts = _load_actions_tensor_from_json(p, device=torch.device("cpu"), dtype=torch.float32)
        assert acts is not None and acts.shape == (81, 12)
        tail = _tail_context_actions(acts, 5, device=torch.device("cpu"), dtype=torch.float32)
        assert tail is not None and tail.shape == (5, 12)
        assert torch.allclose(tail[0], acts[-5], atol=1e-6)
        assert torch.allclose(tail[-1], acts[-1], atol=1e-6)


def test_spatial_cross_attn_readout_shape():
    B, Nt, Nm, D = 1, 128, 64, 64
    x_target = torch.randn(B, Nt, D)
    mem = torch.randn(B, Nm, D)
    mod = SpatialCrossAttnReadout(dim=D, num_heads=8)
    y = apply_spatial_cross_attn_readout(x_target, mem, mod)
    assert y.shape == x_target.shape


if __name__ == "__main__":
    test_tail_context_actions_from_left45()
    test_spatial_cross_attn_readout_shape()
    print("test_two_chunk_anchor_readout: ok")
