"""Unit tests for advantage conditioning and self-play loop modules."""

import pytest
import torch

from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo_r1.training.advantage_conditioning import (
    ADV_TOKEN_NAMES,
    ADV_TOKEN_STRINGS,
    AdvantageBuffer,
    AdvCondDataset,
    build_conditioned_sequence,
    compute_value_targets,
    train_segment_value_head,
)
from alpamayo_r1.training.selfplay_loop import RolloutReplayBuffer, ScenePartitioner


# ---------------------------------------------------------------------------
# Fake token IDs for testing (no real tokenizer needed)
# ---------------------------------------------------------------------------

FAKE_ADV_TOKEN_IDS = {name: 200000 + i for i, name in enumerate(ADV_TOKEN_NAMES)}
# Fake traj_future_start token ID for testing split placement
FAKE_TRAJ_START_ID = 999999


class TestAdvantageBuffer:
    def test_initial_empty(self):
        buf = AdvantageBuffer(k_obs=50, k_traj=50)
        eps = buf.compute_thresholds()
        assert eps == (0.0, 0.0)

    def test_update_and_binarize(self):
        buf = AdvantageBuffer(k_obs=50, k_traj=50)
        obs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        traj = [0.0] * 10
        buf.update(obs, traj)

        # Median of obs = 5.5, so 6+ should be positive (strict >)
        i_obs, i_traj = buf.binarize(6.0, 0.1)
        assert i_obs is True
        assert i_traj is True

        i_obs, i_traj = buf.binarize(1.0, -1.0)
        assert i_obs is False
        assert i_traj is False

    def test_binarize_uses_strict_greater_than(self):
        """Binarization uses > (strict), not >= (inclusive)."""
        buf = AdvantageBuffer(k_obs=50, k_traj=50)
        buf.update([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        eps_obs, eps_traj = buf.compute_thresholds()
        # At the exact threshold, should be False (strict >)
        i_obs, i_traj = buf.binarize(eps_obs, eps_traj)
        assert i_obs is False
        assert i_traj is False

    def test_ema_updates(self):
        buf = AdvantageBuffer(ema_alpha=0.5)
        buf.update([10.0], [0.0])
        assert buf._ema_obs == 10.0

        buf.update([0.0], [0.0])
        assert buf._ema_obs == pytest.approx(5.0)

    def test_state_dict_roundtrip(self):
        buf = AdvantageBuffer(k_obs=30, k_traj=50)
        buf.update([1.0, 2.0], [5.0, 6.0])

        state = buf.state_dict()
        buf2 = AdvantageBuffer(k_obs=30, k_traj=50)
        buf2.load_state_dict(state)

        assert list(buf2._buf_obs) == [1.0, 2.0]
        assert list(buf2._buf_traj) == [5.0, 6.0]
        assert buf2._ema_obs == buf._ema_obs


class TestBuildConditionedSequence:
    def test_split_placement(self):
        """Tokens are placed at causal positions: adv_obs before CoC, adv_traj between CoC and traj."""
        # completion_ids: [CoC tokens..., traj_future_start, traj tokens...]
        completion = [10, 11, FAKE_TRAJ_START_ID, 20, 21]
        result = build_conditioned_sequence(
            prompt_ids=[1, 2, 3],
            completion_ids=completion,
            i_obs=True,
            i_traj=True,
            adv_token_ids=FAKE_ADV_TOKEN_IDS,
            traj_future_start_id=FAKE_TRAJ_START_ID,
            p_drop=0.0,
        )
        assert not result["is_unconditional"]
        # Expected: [1,2,3] [obs_pos] [10,11] [traj_pos] [TRAJ_START,20,21]
        ids = result["input_ids"]
        assert ids[3] == FAKE_ADV_TOKEN_IDS["adv_obs_pos"]
        assert ids[4:6] == [10, 11]  # CoC part
        assert ids[6] == FAKE_ADV_TOKEN_IDS["adv_traj_pos"]
        assert ids[7] == FAKE_TRAJ_START_ID  # traj part starts

    def test_negative_labels(self):
        """Mixed labels produce neg tokens at correct positions."""
        completion = [10, FAKE_TRAJ_START_ID, 20]
        result = build_conditioned_sequence(
            prompt_ids=[1, 2],
            completion_ids=completion,
            i_obs=False,
            i_traj=False,
            adv_token_ids=FAKE_ADV_TOKEN_IDS,
            traj_future_start_id=FAKE_TRAJ_START_ID,
            p_drop=1.0,  # Even 100% drop: non-all-positive → conditional
        )
        assert not result["is_unconditional"]
        assert result["input_ids"][2] == FAKE_ADV_TOKEN_IDS["adv_obs_neg"]
        assert result["input_ids"][4] == FAKE_ADV_TOKEN_IDS["adv_traj_neg"]

    def test_unconditional_dropout(self):
        """All-positive with p_drop=1.0 -> always unconditional (no tokens inserted)."""
        completion = [10, 11, FAKE_TRAJ_START_ID, 20]
        result = build_conditioned_sequence(
            prompt_ids=[1, 2, 3],
            completion_ids=completion,
            i_obs=True,
            i_traj=True,
            adv_token_ids=FAKE_ADV_TOKEN_IDS,
            traj_future_start_id=FAKE_TRAJ_START_ID,
            p_drop=1.0,
        )
        assert result["is_unconditional"]
        # No conditioning tokens: prompt(3) + completion(4) = 7
        assert len(result["input_ids"]) == 7
        assert result["labels"][:3] == [IGNORE_INDEX] * 3
        assert result["labels"][3:] == completion

    def test_labels_mask_conditioning_tokens(self):
        """Conditioning tokens get IGNORE_INDEX in labels, completion tokens get real IDs."""
        completion = [10, FAKE_TRAJ_START_ID, 20, 21]
        result = build_conditioned_sequence(
            prompt_ids=[1, 2],
            completion_ids=completion,
            i_obs=True,
            i_traj=True,
            adv_token_ids=FAKE_ADV_TOKEN_IDS,
            traj_future_start_id=FAKE_TRAJ_START_ID,
            p_drop=0.0,
        )
        labels = result["labels"]
        # prompt(2) + adv_obs(1) = 3 IGNORE_INDEX
        assert labels[:3] == [IGNORE_INDEX] * 3
        # CoC token (supervised)
        assert labels[3] == 10
        # adv_traj = IGNORE_INDEX
        assert labels[4] == IGNORE_INDEX
        # traj tokens (supervised)
        assert labels[5:] == [FAKE_TRAJ_START_ID, 20, 21]

    def test_attention_mask_all_ones(self):
        result = build_conditioned_sequence(
            prompt_ids=[1, 2],
            completion_ids=[10, FAKE_TRAJ_START_ID, 20],
            i_obs=True,
            i_traj=False,
            adv_token_ids=FAKE_ADV_TOKEN_IDS,
            traj_future_start_id=FAKE_TRAJ_START_ID,
            p_drop=0.0,
        )
        assert all(m == 1 for m in result["attention_mask"])
        assert len(result["attention_mask"]) == len(result["input_ids"])


class TestAdvCondDataset:
    def test_basic(self):
        rollouts = [
            {"prompt_ids": [1, 2], "completion_ids": [10, FAKE_TRAJ_START_ID, 20]},
            {"prompt_ids": [3, 4], "completion_ids": [12, FAKE_TRAJ_START_ID, 22]},
        ]
        labels = [
            {"i_obs": True, "i_traj": True},
            {"i_obs": False, "i_traj": False},
        ]
        ds = AdvCondDataset(
            rollouts,
            labels,
            FAKE_ADV_TOKEN_IDS,
            p_drop=0.0,
            traj_future_start_id=FAKE_TRAJ_START_ID,
        )
        assert len(ds) == 2
        item0 = ds[0]
        assert "input_ids" in item0
        assert "labels" in item0
        assert "is_unconditional" in item0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Mismatched lengths"):
            AdvCondDataset([{"prompt_ids": [1], "completion_ids": [2]}], [], FAKE_ADV_TOKEN_IDS)


class TestAdvTokenStrings:
    def test_all_tokens_exist(self):
        """Verify all 6 advantage tokens are defined in SPECIAL_TOKENS."""
        for name in ADV_TOKEN_NAMES:
            assert name in ADV_TOKEN_STRINGS
            assert ADV_TOKEN_STRINGS[name] == f"<|{name}|>"


class TestScenePartitioner:
    def test_partitions_are_disjoint(self):
        clip_ids = [f"clip_{i}" for i in range(30)]
        sp = ScenePartitioner(clip_ids, num_iterations=3, seed=42)

        p0 = set(sp.get_fresh_scenes(0))
        p1 = set(sp.get_fresh_scenes(1))
        p2 = set(sp.get_fresh_scenes(2))

        # No overlap
        assert p0 & p1 == set()
        assert p0 & p2 == set()
        assert p1 & p2 == set()

        # All scenes covered
        assert p0 | p1 | p2 == set(clip_ids)

    def test_partition_sizes_approximately_equal(self):
        clip_ids = [f"clip_{i}" for i in range(17)]
        sp = ScenePartitioner(clip_ids, num_iterations=3)
        sizes = [len(sp.get_fresh_scenes(i)) for i in range(3)]
        # 17/3 = 5.67, so partitions should be 6, 6, 5
        assert sum(sizes) == 17
        assert max(sizes) - min(sizes) <= 1

    def test_historical_scenes_accumulate(self):
        clip_ids = [f"clip_{i}" for i in range(20)]
        sp = ScenePartitioner(clip_ids, num_iterations=4, seed=42)

        assert sp.get_historical_scenes(0) == []
        assert set(sp.get_historical_scenes(1)) == set(sp.get_fresh_scenes(0))
        assert set(sp.get_historical_scenes(2)) == set(sp.get_fresh_scenes(0)) | set(
            sp.get_fresh_scenes(1)
        )

    def test_out_of_bounds_raises(self):
        sp = ScenePartitioner(["a", "b", "c"], num_iterations=2)
        with pytest.raises(IndexError):
            sp.get_fresh_scenes(2)


class TestRolloutReplayBuffer:
    def test_add_and_sample(self):
        buf = RolloutReplayBuffer(max_size=100)
        rollouts = [{"prompt_ids": [1], "completion_ids": [2]}]
        labels = [{"i_obs": True, "i_traj": False}]

        buf.add(rollouts, labels, iteration=0)
        assert len(buf) == 1

        sampled_r, sampled_l = buf.sample(1)
        assert len(sampled_r) == 1
        assert sampled_r[0]["prompt_ids"] == [1]
        assert sampled_l[0]["i_traj"] is False

    def test_eviction_on_max_size(self):
        buf = RolloutReplayBuffer(max_size=3)
        for i in range(5):
            buf.add(
                [{"prompt_ids": [i], "completion_ids": [i]}],
                [{"i_obs": True, "i_traj": True}],
                iteration=i,
            )
        assert len(buf) == 3

    def test_sample_empty(self):
        buf = RolloutReplayBuffer()
        r, l = buf.sample(5)
        assert r == []
        assert l == []


class TestValueHeadTraining:
    """Test value head training functions with a mock SegmentValueHead."""

    def _make_mock_data(self, n=4, T_traj=8, D=32):
        """Create synthetic segment hidden states and reward stash."""
        segment_hidden_stash = []
        segment_reward_stash = []
        completion_segment_map = []

        for i in range(n):
            segment_hidden_stash.append(
                {
                    "h_obs": torch.randn(1, D),
                    "h_coc": torch.randn(1, D),
                    "h_traj": torch.randn(T_traj, D),
                }
            )
            segment_reward_stash.append(
                {
                    "r_traj": 0.5 + 0.3 * i,
                    "r_reason": 0.4 + 0.1 * i,
                    "r_consist": 0.3,
                    "r_traj_per_step": [0.5 / T_traj] * T_traj,
                }
            )
            completion_segment_map.append(
                {
                    "coc_len": 20,
                    "traj_len": T_traj,
                    "traj_positions": list(range(T_traj)),
                    "total_len": 20 + T_traj,
                }
            )

        return segment_hidden_stash, segment_reward_stash, completion_segment_map

    def test_compute_value_targets(self):
        """Verify returns-to-go are computed correctly."""
        _, reward_stash, seg_map = self._make_mock_data(n=2, T_traj=4)
        g_obs, g_coc, g_traj = compute_value_targets(
            reward_stash, seg_map, reward_weights=(0.5, 0.25, 0.25)
        )
        assert len(g_obs) == 2
        assert len(g_coc) == 2
        assert len(g_traj) == 2

        # G(s_obs) = w_reason * r_reason + traj_total
        # G(s_coc) = traj_total (excludes reasoning reward)
        assert g_obs[0] > g_coc[0]  # obs includes reasoning reward

        # G(s_traj) should be decreasing (return-to-go shrinks)
        assert g_traj[0][0].item() >= g_traj[0][-1].item()

    def test_train_reduces_loss(self):
        """Verify that training for multiple epochs reduces the loss."""
        from alpamayo_r1.training.value_head import SegmentValueHead

        D = 32
        vh = SegmentValueHead(hidden_dim=D)
        optimizer = torch.optim.Adam(vh.parameters(), lr=1e-3)

        hidden_stash, reward_stash, seg_map = self._make_mock_data(n=8, T_traj=4, D=D)
        g_obs, g_coc, g_traj = compute_value_targets(
            reward_stash, seg_map, reward_weights=(0.5, 0.25, 0.25)
        )

        # Train for 1 epoch to get initial loss
        m1 = train_segment_value_head(
            vh, optimizer, hidden_stash, g_obs, g_coc, g_traj, num_epochs=1
        )
        # Train for 50 more epochs
        m2 = train_segment_value_head(
            vh, optimizer, hidden_stash, g_obs, g_coc, g_traj, num_epochs=50
        )
        assert m2["loss"] < m1["loss"], f"Loss should decrease: {m1['loss']} -> {m2['loss']}"

    def test_train_empty_data(self):
        """Training with no data should return zero loss."""
        from alpamayo_r1.training.value_head import SegmentValueHead

        vh = SegmentValueHead(hidden_dim=32)
        optimizer = torch.optim.Adam(vh.parameters(), lr=1e-3)
        metrics = train_segment_value_head(vh, optimizer, [], [], [], [], num_epochs=5)
        assert metrics["loss"] == 0.0
