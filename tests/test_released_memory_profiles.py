import torch

from diffsynth.models.memory.block_wise_ssm import BlockWiseStateSpaceMemory
from env.memory_baseline_runtime import infer_memory_profile
from src.model_training.multichunk_sample_utils import (
    build_causal_continuation_protocol,
    build_prev_tail_continuation_protocol,
)


def _rt(x):
    return [float(x), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_causal_ssm_v2_is_active_and_rejects_suffix():
    module = BlockWiseStateSpaceMemory(8, causal_v2=True)
    x = torch.randn(1, 6 * 2, 8, requires_grad=True)
    out = module(
        x, f=6, num_context_frames=2, context_position="prefix"
    )
    out.square().mean().backward()
    assert module.residual_logit.grad is not None
    assert module.last_stats["contribution_norm"].item() > 0

    try:
        module(x, f=6, num_context_frames=2, context_position="suffix")
    except ValueError:
        pass
    else:
        raise AssertionError("causal SSM must reject suffix layout")


def test_generated_history_protocols_keep_real_relative_rt():
    frames = list(range(81))
    actions = [_rt(index) for index in range(21)]

    causal_frames, causal_actions, _ = build_causal_continuation_protocol(
        frames, actions, 5
    )
    assert causal_frames == [60, 64, 68, 72, 76]
    assert [round(row[0]) for row in causal_actions] == [-5, -4, -3, -2, -1]

    tail_frames, tail_actions, _ = build_prev_tail_continuation_protocol(
        frames, actions, 81, nearest_first=True
    )
    assert tail_frames[:3] == [80, 79, 78]
    assert round(tail_actions[0][0]) == 0
    assert round(tail_actions[-1][0]) == -20


def test_released_profile_contracts():
    ssm = infer_memory_profile("block_wise_ssm_causal_v2/epoch-0.safetensors")
    assert ssm.context_position == "prefix"
    assert ssm.training_context_source == "causal_prev_prefix"
    assert ssm.block_wise_ssm_causal_v2
    assert not ssm.use_moc

    framepack = infer_memory_profile("framepack_len_r8/epoch-0.safetensors")
    assert framepack.context_position == "suffix"
    assert framepack.training_context_source == "prev_chunk_tail"
    assert framepack.framepack_ratio == 8
    assert framepack.framepack_length_strategy == "packed_multiscale"
