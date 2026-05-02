"""Quick consistency checks for multichunk-aligned context selection (run: PYTHONPATH=. python3 tests/test_context_chunk_utils.py)."""
import os
import sys

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from src.model_training.context_chunk_utils import (
    context_frames_for_next_chunk,
    replay_context_global_indices,
    replay_context_actions_from_segment_actions,
    prev_chunk_tail_global_indices,
)


def test_replay_indices_match_frame_order():
    n, K = 81, 5
    frames = list(range(n))
    picked = context_frames_for_next_chunk(frames, K)
    idxs = replay_context_global_indices(n, K)
    assert [frames[i] for i in idxs] == picked


def test_replay_actions_align():
    n, K = 81, 5
    actions = [[float(i)] * 12 for i in range(n)]
    out = replay_context_actions_from_segment_actions(actions, n, K)
    idxs = replay_context_global_indices(n, K)
    assert out is not None
    assert len(out) == len(idxs)
    for row, i in zip(out, idxs):
        assert row[0] == float(i)


def test_prev_chunk_tail_indices():
    assert prev_chunk_tail_global_indices(10, 3) == [7, 8, 9]
    assert prev_chunk_tail_global_indices(10, 3, nearest_first=True) == [9, 8, 7]
    assert prev_chunk_tail_global_indices(2, 5) is None


if __name__ == "__main__":
    test_replay_indices_match_frame_order()
    test_replay_actions_align()
    test_prev_chunk_tail_indices()
    print("test_context_chunk_utils: ok")
