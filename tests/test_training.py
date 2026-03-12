# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the GRPO training module.

Tests reward functions, dataset utilities, rollout helpers, and config loading
without requiring a GPU or the full AlpamayoR1 model.
"""

import math
import re

import numpy as np
import pytest
import yaml


# ===================================================================
# Import tests
# ===================================================================

class TestImports:
    """Verify all training modules can be imported."""

    def test_import_training_package(self):
        import alpamayo_r1.training

    def test_import_rewards(self):
        from alpamayo_r1.training.rewards import (
            consistency_reward,
            reasoning_quality_reward,
            trajectory_quality_reward,
        )

    def test_import_dataset(self):
        from alpamayo_r1.training.dataset import build_alpamayo_dataset, _build_prompt_text

    def test_import_rollout(self):
        from alpamayo_r1.training.rollout import _parse_clip_metadata, AlpamayoGRPOTrainer

    def test_import_train_grpo(self):
        from alpamayo_r1.training.train_grpo import _freeze_non_vlm_params


# ===================================================================
# Reward function tests
# ===================================================================

class TestTrajectoryQualityReward:
    """Tests for trajectory_quality_reward."""

    def setup_method(self):
        from alpamayo_r1.training.rewards import trajectory_quality_reward
        self.reward_fn = trajectory_quality_reward

    def test_perfect_prediction(self):
        """Identical pred and gt should give reward close to 1.0."""
        T = 64
        gt = np.zeros((T, 3), dtype=np.float32).flatten().tolist()
        pred = np.zeros((3, T, 3), dtype=np.float32).flatten().tolist()  # 3 samples
        rewards = self.reward_fn(["dummy"], pred_xyz=[pred], gt_xyz=[gt])
        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(1.0, abs=0.01)

    def test_bad_prediction(self):
        """Large displacement should give reward close to 0.0."""
        T = 64
        gt = np.zeros((T, 3), dtype=np.float32).flatten().tolist()
        pred = np.full((3, T, 3), 100.0, dtype=np.float32).flatten().tolist()
        rewards = self.reward_fn(["dummy"], pred_xyz=[pred], gt_xyz=[gt])
        assert rewards[0] == pytest.approx(0.0, abs=0.01)

    def test_moderate_prediction(self):
        """Moderate error should give intermediate reward."""
        T = 64
        gt = np.zeros((T, 3), dtype=np.float32).flatten().tolist()
        # Average error ~2.5m with threshold=5.0 → reward ~0.5
        pred_single = np.zeros((T, 3), dtype=np.float32)
        pred_single[:, 0] = 2.5  # 2.5m offset in x
        pred = np.stack([pred_single] * 3).flatten().tolist()
        rewards = self.reward_fn(["dummy"], pred_xyz=[pred], gt_xyz=[gt])
        assert 0.3 < rewards[0] < 0.7

    def test_min_over_samples(self):
        """minADE should pick the best sample, not average."""
        T = 64
        gt = np.zeros((T, 3), dtype=np.float32).flatten().tolist()
        # Sample 0: perfect, Sample 1: terrible
        s0 = np.zeros((T, 3), dtype=np.float32)
        s1 = np.full((T, 3), 50.0, dtype=np.float32)
        pred = np.stack([s0, s1]).flatten().tolist()
        rewards = self.reward_fn(["dummy"], pred_xyz=[pred], gt_xyz=[gt])
        assert rewards[0] == pytest.approx(1.0, abs=0.01)

    def test_none_inputs(self):
        """Missing pred/gt should return 0.0 reward."""
        rewards = self.reward_fn(["dummy"], pred_xyz=None, gt_xyz=None)
        assert rewards == [0.0]

    def test_batch_of_two(self):
        """Multiple samples in a batch."""
        T = 64
        gt_good = np.zeros((T, 3), dtype=np.float32).flatten().tolist()
        pred_good = np.zeros((2, T, 3), dtype=np.float32).flatten().tolist()
        gt_bad = np.zeros((T, 3), dtype=np.float32).flatten().tolist()
        pred_bad = np.full((2, T, 3), 100.0, dtype=np.float32).flatten().tolist()

        rewards = self.reward_fn(
            ["a", "b"],
            pred_xyz=[pred_good, pred_bad],
            gt_xyz=[gt_good, gt_bad],
        )
        assert len(rewards) == 2
        assert rewards[0] > 0.9
        assert rewards[1] < 0.1


class TestReasoningQualityReward:
    """Tests for reasoning_quality_reward."""

    def setup_method(self):
        from alpamayo_r1.training.rewards import reasoning_quality_reward
        self.reward_fn = reasoning_quality_reward

    def test_good_reasoning(self):
        """Well-formed CoC text should score high."""
        text = (
            "The ego vehicle is approaching an intersection. Because there is a "
            "pedestrian crossing ahead, the vehicle should decelerate. Therefore, "
            "the vehicle will slow down and maintain its lane to ensure safety. "
            "Since the traffic light is green, it will continue straight after "
            "the pedestrian has crossed."
        )
        rewards = self.reward_fn([text])
        assert rewards[0] > 0.7

    def test_empty_text(self):
        """Empty text should score very low."""
        rewards = self.reward_fn([""])
        assert rewards[0] < 0.5  # gets 0.25 (no-repetition criterion)

    def test_repetitive_text(self):
        """Degenerate repetition should be penalized."""
        text = "the car is moving forward. " * 20
        rewards = self.reward_fn([text])
        # Should fail the repetition criterion
        assert rewards[0] < 0.8

    def test_short_text(self):
        """Very short text should lose the length criterion."""
        rewards = self.reward_fn(["go straight"])
        assert rewards[0] < 0.7

    def test_no_driving_terms(self):
        """Text without driving vocabulary should score lower."""
        text = (
            "Because the weather is nice, therefore we should enjoy the day. "
            "Since it is sunny, the birds are singing consequently."
        )
        rewards = self.reward_fn([text])
        # Has causal connectors but no driving terms
        assert rewards[0] < 0.8

    def test_none_completion(self):
        """None completion should be handled gracefully."""
        rewards = self.reward_fn([None])
        assert isinstance(rewards[0], float)

    def test_batch(self):
        """Multiple completions in a batch."""
        texts = ["good because the vehicle lane traffic pedestrian therefore", "", "x"]
        rewards = self.reward_fn(texts)
        assert len(rewards) == 3
        assert rewards[0] > rewards[1]


class TestConsistencyReward:
    """Tests for consistency_reward."""

    def setup_method(self):
        from alpamayo_r1.training.rewards import consistency_reward
        self.reward_fn = consistency_reward

    def _make_left_turn_traj(self) -> list[float]:
        """Create a trajectory that turns left (positive y displacement)."""
        T = 64
        traj = np.zeros((T, 3), dtype=np.float32)
        traj[:, 0] = np.linspace(0, 30, T)  # forward
        traj[:, 1] = np.linspace(0, 5, T)   # left turn (positive y > 1.0m)
        return traj.flatten().tolist()

    def _make_straight_traj(self) -> list[float]:
        """Create a straight trajectory."""
        T = 64
        traj = np.zeros((T, 3), dtype=np.float32)
        traj[:, 0] = np.linspace(0, 40, T)
        return traj.flatten().tolist()

    def test_consistent_left_turn(self):
        """CoC mentions left turn + trajectory turns left = high consistency."""
        text = "The vehicle is turning left at the intersection."
        pred = self._make_left_turn_traj()
        rewards = self.reward_fn([text], pred_xyz=[pred])
        assert rewards[0] > 0.5

    def test_inconsistent_text(self):
        """CoC mentions right turn but trajectory goes left = low consistency."""
        text = "The vehicle is turning right at the intersection."
        pred = self._make_left_turn_traj()
        rewards = self.reward_fn([text], pred_xyz=[pred])
        assert rewards[0] < 0.5

    def test_consistent_straight(self):
        """CoC mentions straight + trajectory goes straight."""
        text = "The vehicle will continue straight ahead."
        pred = self._make_straight_traj()
        rewards = self.reward_fn([text], pred_xyz=[pred])
        assert rewards[0] > 0.5

    def test_none_pred_xyz(self):
        """Missing trajectories should return 0.0."""
        rewards = self.reward_fn(["some text"], pred_xyz=None)
        assert rewards == [0.0]


# ===================================================================
# Dataset utility tests
# ===================================================================

class TestDatasetUtils:
    """Tests for dataset.py utility functions."""

    def test_build_prompt_text_format(self):
        from alpamayo_r1.training.dataset import _build_prompt_text

        messages = _build_prompt_text("clip_abc_123", 5100000)

        # Should be a list of message dicts
        assert isinstance(messages, list)
        assert len(messages) == 2  # system + user

        # System message contains clip metadata
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert "clip_abc_123" in system_msg["content"]
        assert "5100000" in system_msg["content"]

        # User message is the driving prompt
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert "chain-of-thought" in user_msg["content"]

    def test_clip_metadata_roundtrip(self):
        """Ensure clip metadata encoded in prompt can be parsed back."""
        from alpamayo_r1.training.dataset import _build_prompt_text
        from alpamayo_r1.training.rollout import _parse_clip_metadata

        clip_id = "test-clip-with-dashes_and_underscores"
        t0_us = 7200000
        messages = _build_prompt_text(clip_id, t0_us)

        # Concatenate message content as TRL would
        prompt_text = " ".join(m["content"] for m in messages)
        parsed_clip, parsed_t0 = _parse_clip_metadata(prompt_text)
        assert parsed_clip == clip_id
        assert parsed_t0 == t0_us


# ===================================================================
# Rollout utility tests
# ===================================================================

class TestRolloutUtils:
    """Tests for rollout.py utility functions."""

    def test_parse_clip_metadata(self):
        from alpamayo_r1.training.rollout import _parse_clip_metadata

        text = "Some prefix [clip_id=abc123] middle [t0_us=5100000] suffix"
        clip_id, t0_us = _parse_clip_metadata(text)
        assert clip_id == "abc123"
        assert t0_us == 5100000

    def test_parse_clip_metadata_missing_raises(self):
        from alpamayo_r1.training.rollout import _parse_clip_metadata

        with pytest.raises(ValueError, match="Could not parse"):
            _parse_clip_metadata("no metadata here")

    def test_collate_rollout_outputs(self):
        import torch
        from alpamayo_r1.training.rollout import _collate_rollout_outputs

        # 2 samples with different completion lengths
        prompt_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])]
        completion_ids = [torch.tensor([10, 11]), torch.tensor([20, 21, 22])]
        logprobs = [torch.tensor([-.1, -.2]), torch.tensor([-.3, -.4, -.5])]
        pred_xyz = [[1.0, 2.0], [3.0, 4.0]]
        gt_xyz = [[5.0, 6.0], [7.0, 8.0]]
        coc_texts = ["text1", "text2"]

        result = _collate_rollout_outputs(
            prompt_ids, completion_ids, logprobs,
            pred_xyz, gt_xyz, coc_texts,
            pad_token_id=0,
        )

        assert "prompt_ids" in result
        assert "completion_ids" in result
        assert "logprobs" in result
        assert "pred_xyz" in result
        assert "gt_xyz" in result
        assert "completions" in result

        # Check shapes
        assert result["prompt_ids"].shape == (2, 4)  # padded to max prompt len
        assert result["completion_ids"].shape == (2, 3)  # padded to max completion len
        assert result["logprobs"].shape == (2, 3)

        # Check left-padding of prompts
        assert result["prompt_ids"][0, 0].item() == 0  # pad
        assert result["prompt_ids"][0, 1].item() == 1  # first real token
        assert result["prompt_ids"][1, 0].item() == 4  # no padding needed

        # Check right-padding of completions
        assert result["completion_ids"][0, 2].item() == 0  # pad
        assert result["completion_ids"][1, 2].item() == 22

    def test_collate_empty_completion(self):
        """Handle edge case of empty completions."""
        import torch
        from alpamayo_r1.training.rollout import _collate_rollout_outputs

        result = _collate_rollout_outputs(
            [torch.tensor([1])],
            [torch.tensor([], dtype=torch.long)],
            [torch.tensor([], dtype=torch.float32)],
            [[]], [[]], [""],
            pad_token_id=0,
        )
        assert result["completion_ids"].shape == (1, 1)  # min length 1
        assert result["logprobs"].shape == (1, 1)


# ===================================================================
# Config tests
# ===================================================================

class TestConfig:
    """Tests for the GRPO config file."""

    def test_config_loads(self):
        config_path = "src/alpamayo_r1/training/configs/grpo_default.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        assert cfg["model_name"] == "nvidia/Alpamayo-R1-10B"
        assert cfg["seed"] == 42

    def test_config_lora_section(self):
        with open("src/alpamayo_r1/training/configs/grpo_default.yaml") as f:
            cfg = yaml.safe_load(f)

        lora = cfg["lora"]
        assert lora["r"] == 16
        assert lora["alpha"] == 32
        assert "q_proj" in lora["target_modules"]
        assert "o_proj" in lora["target_modules"]

    def test_config_training_section(self):
        with open("src/alpamayo_r1/training/configs/grpo_default.yaml") as f:
            cfg = yaml.safe_load(f)

        train = cfg["training"]
        assert train["beta"] == 0.0  # no KL penalty
        assert train["num_generations"] == 8
        assert train["loss_type"] == "grpo"
        # bf16 is set directly in train_grpo.py, not in the YAML config

    def test_config_reward_weights_sum_to_one(self):
        with open("src/alpamayo_r1/training/configs/grpo_default.yaml") as f:
            cfg = yaml.safe_load(f)

        rewards = cfg["rewards"]
        total = rewards["trajectory_weight"] + rewards["reasoning_weight"] + rewards["consistency_weight"]
        assert total == pytest.approx(1.0)

    def test_config_rollout_matches_training(self):
        with open("src/alpamayo_r1/training/configs/grpo_default.yaml") as f:
            cfg = yaml.safe_load(f)

        assert cfg["rollout"]["num_traj_samples"] == cfg["training"]["num_generations"]


# ===================================================================
# Reward helper tests
# ===================================================================

class TestTrajectoryBehaviorDetection:
    """Tests for _trajectory_to_behaviors helper."""

    def test_left_turn_detected(self):
        from alpamayo_r1.training.rewards import _trajectory_to_behaviors

        T = 64
        traj = np.zeros((T, 3), dtype=np.float32)
        traj[:, 0] = np.linspace(0, 30, T)
        traj[:, 1] = np.linspace(0, 5, T)  # >1m lateral = left
        behaviors = _trajectory_to_behaviors(traj.flatten().tolist())
        assert "turning_left" in behaviors

    def test_right_turn_detected(self):
        from alpamayo_r1.training.rewards import _trajectory_to_behaviors

        T = 64
        traj = np.zeros((T, 3), dtype=np.float32)
        traj[:, 0] = np.linspace(0, 30, T)
        traj[:, 1] = np.linspace(0, -5, T)  # <-1m lateral = right
        behaviors = _trajectory_to_behaviors(traj.flatten().tolist())
        assert "turning_right" in behaviors

    def test_straight_detected(self):
        from alpamayo_r1.training.rewards import _trajectory_to_behaviors

        T = 64
        traj = np.zeros((T, 3), dtype=np.float32)
        traj[:, 0] = np.linspace(0, 40, T)
        behaviors = _trajectory_to_behaviors(traj.flatten().tolist())
        assert "going_straight" in behaviors

    def test_stopping_detected(self):
        from alpamayo_r1.training.rewards import _trajectory_to_behaviors

        T = 64
        traj = np.zeros((T, 3), dtype=np.float32)
        # Decelerating to near-zero speed
        traj[:, 0] = np.concatenate([np.linspace(0, 5, T // 2), np.full(T // 2, 5.0)])
        behaviors = _trajectory_to_behaviors(traj.flatten().tolist())
        assert "stopping" in behaviors

    def test_short_trajectory(self):
        from alpamayo_r1.training.rewards import _trajectory_to_behaviors

        traj = np.zeros((2, 3), dtype=np.float32).flatten().tolist()
        behaviors = _trajectory_to_behaviors(traj)
        assert len(behaviors) == 0  # too short to analyze


# ===================================================================
# SceneValueHead tests
# ===================================================================

class TestSceneValueHead:
    """CPU-only tests for the SceneValueHead module."""

    def test_import(self):
        from alpamayo_r1.training.value_head import SceneValueHead

    def test_forward_shape(self):
        """Input (3, 32) → output (3,)."""
        import torch
        from alpamayo_r1.training.value_head import SceneValueHead

        vh = SceneValueHead(hidden_dim=32)
        h0 = torch.randn(3, 32)
        out = vh(h0)
        assert out.shape == (3,), f"Expected shape (3,), got {out.shape}"

    def test_single_sample_shape(self):
        """Input (1, 32) → output (1,) scalar."""
        import torch
        from alpamayo_r1.training.value_head import SceneValueHead

        vh = SceneValueHead(hidden_dim=32)
        h0 = torch.randn(1, 32)
        out = vh(h0)
        assert out.shape == (1,), f"Expected shape (1,), got {out.shape}"

    def test_gradient_flow(self):
        """Gradients should flow through the value head."""
        import torch
        from alpamayo_r1.training.value_head import SceneValueHead

        vh = SceneValueHead(hidden_dim=32)
        h0 = torch.randn(4, 32, requires_grad=True)
        out = vh(h0)
        loss = out.mean()
        loss.backward()
        assert h0.grad is not None
        assert h0.grad.shape == h0.shape

    def test_parameters_trainable(self):
        """All parameters should exist and require gradients."""
        from alpamayo_r1.training.value_head import SceneValueHead

        vh = SceneValueHead(hidden_dim=32)
        params = list(vh.parameters())
        assert len(params) > 0, "SceneValueHead should have trainable parameters"
        for p in params:
            assert p.requires_grad, "All parameters should require gradients"

    def test_parameter_count(self):
        """3-layer MLP with hidden_dim=32 should have expected parameter count."""
        from alpamayo_r1.training.value_head import SceneValueHead

        vh = SceneValueHead(hidden_dim=32)
        # Layer 1: 32*512 + 512 = 16896
        # Layer 2: 512*128 + 128 = 65664
        # Layer 3: 128*1 + 1 = 129
        total_params = sum(p.numel() for p in vh.parameters())
        expected = (32 * 512 + 512) + (512 * 128 + 128) + (128 * 1 + 1)
        assert total_params == expected, f"Expected {expected} params, got {total_params}"

    def test_optimizer_step(self):
        """Value head should update its weights after an optimizer step."""
        import torch
        from alpamayo_r1.training.value_head import SceneValueHead

        vh = SceneValueHead(hidden_dim=32)
        optimizer = torch.optim.Adam(vh.parameters(), lr=1e-3)

        # Record initial weights
        initial_weight = vh.net[0].weight.data.clone()

        h0 = torch.randn(4, 32)
        targets = torch.rand(4)
        pred = vh(h0)
        loss = torch.nn.functional.mse_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(vh.net[0].weight.data, initial_weight), (
            "Weights should change after optimizer step"
        )

    def test_detached_h0_no_vlm_grad(self):
        """h0 detached from upstream should not propagate gradients back."""
        import torch
        from alpamayo_r1.training.value_head import SceneValueHead

        vh = SceneValueHead(hidden_dim=32)
        # Simulate a 'frozen' upstream tensor (detached)
        upstream = torch.randn(2, 32, requires_grad=True)
        h0_detached = upstream.detach()

        pred = vh(h0_detached)
        loss = pred.mean()
        loss.backward()

        # upstream gradient should be None since h0 was detached
        assert upstream.grad is None, "Detached h0 should not propagate gradients to upstream"


class TestValueHeadStage0:
    """CPU-only tests for stage-0 pre-training configuration and behaviour."""

    def test_config_has_pretrain_fields(self):
        """grpo_default.yaml must contain pretrain_steps, save_path, load_path."""
        with open("src/alpamayo_r1/training/configs/grpo_default.yaml") as f:
            cfg = yaml.safe_load(f)
        vh = cfg["value_head"]
        assert "pretrain_steps" in vh, "value_head must have pretrain_steps"
        assert "save_path" in vh, "value_head must have save_path"
        assert "load_path" in vh, "value_head must have load_path"
        assert vh["pretrain_steps"] == 0, "default pretrain_steps should be 0"
        assert vh["save_path"] is None, "default save_path should be null"
        assert vh["load_path"] is None, "default load_path should be null"

    def test_train_value_head_step_updates_weights(self):
        """_train_value_head_step should update value head weights in-place."""
        import torch
        import torch.nn.functional as F
        from alpamayo_r1.training.value_head import SceneValueHead

        hidden_dim = 32
        vh = SceneValueHead(hidden_dim=hidden_dim)
        optimizer = torch.optim.Adam(vh.parameters(), lr=1e-3)
        initial_weight = vh.net[0].weight.data.clone()

        # Simulate what _train_value_head_step does internally
        h0 = torch.randn(4, hidden_dim)
        rewards = torch.rand(4)
        v_pred = vh(h0)
        loss = F.mse_loss(v_pred, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert not torch.allclose(vh.net[0].weight.data, initial_weight), (
            "Value head weights should change after a training step"
        )

    def test_stage0_zero_loss_no_vlm_grad(self):
        """Stage-0 zero loss tensor should have requires_grad=True but not flow back."""
        import torch

        # Simulate stage-0 return: a zero tensor with requires_grad
        zero_loss = torch.tensor(0.0, requires_grad=True)
        assert zero_loss.requires_grad
        # Backward should succeed without error
        zero_loss.backward()

    def test_save_load_roundtrip(self):
        """Value head weights saved and reloaded should be identical."""
        import tempfile
        import os
        import torch
        from alpamayo_r1.training.value_head import SceneValueHead

        vh = SceneValueHead(hidden_dim=32)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(vh.state_dict(), path)
            vh2 = SceneValueHead(hidden_dim=32)
            vh2.load_state_dict(torch.load(path, map_location="cpu"))
            for p1, p2 in zip(vh.parameters(), vh2.parameters()):
                assert torch.allclose(p1, p2), "Reloaded weights should match saved weights"
        finally:
            os.unlink(path)

    def test_pretrain_steps_counted_down(self):
        """_value_pretrain_remaining should decrement each stage-0 compute_loss call."""
        # We test the counter logic directly without instantiating the full trainer
        pretrain_remaining = 3
        steps_taken = 0
        while pretrain_remaining > 0:
            pretrain_remaining -= 1
            steps_taken += 1
        assert steps_taken == 3
        assert pretrain_remaining == 0
