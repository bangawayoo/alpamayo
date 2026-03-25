"""Tests for segment hidden state extraction and stashing equivalence.

Verifies that:
1. _extract_segment_hidden produces correct segment boundaries
2. The stash path (segment_hidden in rollout results) produces the same
   output as the standalone extract_segment_hidden path
"""

import torch
import pytest

from alpamayo_r1.training.sft_rollout import _extract_segment_hidden


# Fake special token IDs
SPECIAL_TOKEN_IDS = {
    "cot_start": 100,
    "cot_end": 101,
    "traj_future_start": 102,
    "traj_future_end": 103,
    "traj_history_start": 104,
    "traj_history": 105,
    "traj_history_end": 106,
}
TRAJ_TOKEN_START_IDX = 1000
TRAJ_VOCAB_SIZE = 768


def _make_hidden_states(seq_len: int, dim: int = 64) -> torch.Tensor:
    """Create deterministic hidden states: (1, seq_len, dim)."""
    return torch.arange(seq_len * dim, dtype=torch.float32).reshape(1, seq_len, dim)


class TestExtractSegmentHidden:
    """Test _extract_segment_hidden directly."""

    def test_basic_extraction(self):
        """Normal completion: CoC tokens + cot_end + traj_future_start + traj tokens + traj_future_end."""
        # completion_ids: [10, 11, 101(cot_end), 102(traj_start), 1000, 1001, 1002, 103(traj_end)]
        completion_ids = [10, 11, SPECIAL_TOKEN_IDS["cot_end"], SPECIAL_TOKEN_IDS["traj_future_start"],
                          1000, 1001, 1002, SPECIAL_TOKEN_IDS["traj_future_end"]]

        # hidden_states: (1, 1 + len(completion_ids), D) — first token is last prompt token
        hidden = _make_hidden_states(1 + len(completion_ids))

        seg_hidden, seg_map = _extract_segment_hidden(
            hidden, completion_ids, SPECIAL_TOKEN_IDS,
            TRAJ_TOKEN_START_IDX, TRAJ_VOCAB_SIZE,
        )

        # h_obs: hidden state at position 0 (last prompt token)
        assert seg_hidden["h_obs"].shape == (1, 64)
        assert torch.equal(seg_hidden["h_obs"], hidden[0, 0:1, :])

        # h_coc: hidden state after cot_end (position 2+1=3 in hidden)
        assert seg_hidden["h_coc"].shape == (1, 64)
        assert torch.equal(seg_hidden["h_coc"], hidden[0, 3:4, :])

        # h_traj: hidden states at traj token positions (1000, 1001, 1002 at indices 4,5,6)
        assert seg_hidden["h_traj"].shape == (3, 64)

        # segment_map
        assert seg_map["coc_len"] == 3  # tokens before traj_future_start
        assert seg_map["traj_len"] == 3  # number of traj tokens
        assert seg_map["traj_positions"] == [4, 5, 6]
        assert seg_map["total_len"] == len(completion_ids)

    def test_no_traj_tokens(self):
        """Completion with CoC but no trajectory tokens."""
        completion_ids = [10, 11, 12, SPECIAL_TOKEN_IDS["cot_end"]]
        hidden = _make_hidden_states(1 + len(completion_ids))

        seg_hidden, seg_map = _extract_segment_hidden(
            hidden, completion_ids, SPECIAL_TOKEN_IDS,
            TRAJ_TOKEN_START_IDX, TRAJ_VOCAB_SIZE,
        )

        assert seg_hidden["h_traj"].shape == (0, 64)
        assert seg_map["traj_len"] == 0
        assert seg_map["traj_positions"] == []

    def test_no_cot_end(self):
        """Completion without cot_end — h_coc falls back to h_obs."""
        completion_ids = [10, 11, 12]
        hidden = _make_hidden_states(1 + len(completion_ids))

        seg_hidden, seg_map = _extract_segment_hidden(
            hidden, completion_ids, SPECIAL_TOKEN_IDS,
            TRAJ_TOKEN_START_IDX, TRAJ_VOCAB_SIZE,
        )

        # h_coc should fall back to h_obs
        assert torch.equal(seg_hidden["h_coc"], seg_hidden["h_obs"])

    def test_traj_token_boundary(self):
        """Tokens at exact boundary of traj vocab range."""
        # Token at start_idx is in range, token at start_idx + vocab_size is out
        in_range = TRAJ_TOKEN_START_IDX + TRAJ_VOCAB_SIZE - 1  # last valid
        out_range = TRAJ_TOKEN_START_IDX + TRAJ_VOCAB_SIZE  # first invalid

        completion_ids = [SPECIAL_TOKEN_IDS["traj_future_start"], in_range, out_range]
        hidden = _make_hidden_states(1 + len(completion_ids))

        seg_hidden, seg_map = _extract_segment_hidden(
            hidden, completion_ids, SPECIAL_TOKEN_IDS,
            TRAJ_TOKEN_START_IDX, TRAJ_VOCAB_SIZE,
        )

        # Only in_range should be detected as traj token
        assert seg_map["traj_len"] == 1
        assert seg_map["traj_positions"] == [1]


class TestStashEquivalence:
    """Verify stash path produces same structure as standalone extraction."""

    def test_stash_keys_present(self):
        """Segment hidden stash should have h_obs, h_coc, h_traj keys."""
        completion_ids = [10, SPECIAL_TOKEN_IDS["cot_end"],
                          SPECIAL_TOKEN_IDS["traj_future_start"],
                          1000, 1001, SPECIAL_TOKEN_IDS["traj_future_end"]]
        hidden = _make_hidden_states(1 + len(completion_ids))

        seg_hidden, seg_map = _extract_segment_hidden(
            hidden, completion_ids, SPECIAL_TOKEN_IDS,
            TRAJ_TOKEN_START_IDX, TRAJ_VOCAB_SIZE,
        )

        assert "h_obs" in seg_hidden
        assert "h_coc" in seg_hidden
        assert "h_traj" in seg_hidden
        assert "coc_len" in seg_map
        assert "traj_len" in seg_map
        assert "traj_positions" in seg_map
        assert "total_len" in seg_map

    def test_deterministic(self):
        """Same inputs produce identical outputs."""
        completion_ids = [10, 11, SPECIAL_TOKEN_IDS["cot_end"],
                          SPECIAL_TOKEN_IDS["traj_future_start"],
                          1000, 1001, SPECIAL_TOKEN_IDS["traj_future_end"]]
        hidden = _make_hidden_states(1 + len(completion_ids))

        seg1, map1 = _extract_segment_hidden(
            hidden, completion_ids, SPECIAL_TOKEN_IDS,
            TRAJ_TOKEN_START_IDX, TRAJ_VOCAB_SIZE,
        )
        seg2, map2 = _extract_segment_hidden(
            hidden, completion_ids, SPECIAL_TOKEN_IDS,
            TRAJ_TOKEN_START_IDX, TRAJ_VOCAB_SIZE,
        )

        assert torch.equal(seg1["h_obs"], seg2["h_obs"])
        assert torch.equal(seg1["h_coc"], seg2["h_coc"])
        assert torch.equal(seg1["h_traj"], seg2["h_traj"])
        assert map1 == map2
