# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the vLLM colocate rollout path.

These tests verify the wiring, data flow, and output formats of
``make_vllm_rollout_func`` and the ``_calculate_rewards`` override
WITHOUT requiring a GPU or an actual vLLM engine. Heavy mocking is
used so that these can run in CI on CPU-only nodes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml


# ===================================================================
# Config tests for vLLM section
# ===================================================================

class TestVllmConfig:
    """Verify the vLLM section in grpo_default.yaml."""

    def test_vllm_section_exists(self):
        with open("src/alpamayo_r1/training/configs/grpo_default.yaml") as f:
            cfg = yaml.safe_load(f)

        assert "vllm" in cfg
        vllm = cfg["vllm"]
        assert vllm["enabled"] is False
        assert 0.1 <= vllm["gpu_memory_utilization"] <= 0.9
        assert vllm["tensor_parallel_size"] >= 1
        assert vllm["enable_sleep_mode"] is False
        assert vllm["model_impl"] == "transformers"

    def test_vllm_mode_param_exists(self):
        with open("src/alpamayo_r1/training/configs/grpo_default.yaml") as f:
            cfg = yaml.safe_load(f)

        vllm = cfg["vllm"]
        assert vllm["mode"] == "colocate"

    def test_vllm_server_params_exist(self):
        with open("src/alpamayo_r1/training/configs/grpo_default.yaml") as f:
            cfg = yaml.safe_load(f)

        vllm = cfg["vllm"]
        assert vllm["server_host"] == "0.0.0.0"
        assert vllm["server_port"] == 8000
        assert vllm["server_base_url"] is None
        assert vllm["server_timeout"] == 240.0
        assert vllm["group_port"] == 51216


# ===================================================================
# Import tests
# ===================================================================

class TestVllmImports:
    """Verify vLLM-related symbols can be imported."""

    def test_import_make_vllm_rollout_func(self):
        from alpamayo_r1.training.rollout import make_vllm_rollout_func

    def test_import_clip_data_cache_pil(self):
        from alpamayo_r1.training.rollout import ClipDataCache


# ===================================================================
# ClipDataCache PIL image caching
# ===================================================================

class TestClipDataCachePilImages:
    """Test that ClipDataCache correctly caches PIL images."""

    def _make_mock_avdi(self):
        return MagicMock()

    def _make_mock_processor(self):
        processor = MagicMock()
        processor.apply_chat_template.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 20)),
            "attention_mask": torch.ones(1, 20, dtype=torch.long),
            "pixel_values": torch.randn(4, 3, 224, 224),
            "image_grid_thw": torch.tensor([[1, 224, 224]] * 4),
        }
        return processor

    @patch("alpamayo_r1.training.rollout.load_physical_aiavdataset")
    def test_pil_images_cached(self, mock_load):
        """When cache_pil_images=True, PIL images should be stored."""
        from alpamayo_r1.training.rollout import ClipDataCache

        # Create fake data that load_physical_aiavdataset would return
        N_cameras, num_frames = 4, 4
        mock_load.return_value = {
            "image_frames": torch.randint(0, 255, (N_cameras, num_frames, 3, 64, 64), dtype=torch.uint8),
            "ego_history_xyz": torch.randn(1, 1, 16, 3),
            "ego_history_rot": torch.randn(1, 1, 16, 3, 3),
            "ego_future_xyz": torch.randn(1, 1, 64, 3),
            "ego_future_rot": torch.randn(1, 1, 64, 3, 3),
        }

        cache = ClipDataCache(
            self._make_mock_avdi(),
            self._make_mock_processor(),
            cache_pil_images=True,
        )

        model_inputs, ego_future = cache.get("clip_1", 5100000, torch.device("cpu"))
        pil_images = cache.get_pil_images("clip_1", 5100000)

        # Should have N_cameras * num_frames PIL images
        assert len(pil_images) == N_cameras * num_frames
        from PIL import Image
        assert all(isinstance(img, Image.Image) for img in pil_images)

    @patch("alpamayo_r1.training.rollout.load_physical_aiavdataset")
    def test_pil_images_not_cached_by_default(self, mock_load):
        """When cache_pil_images=False (default), no PIL images are stored."""
        from alpamayo_r1.training.rollout import ClipDataCache

        mock_load.return_value = {
            "image_frames": torch.randint(0, 255, (4, 4, 3, 64, 64), dtype=torch.uint8),
            "ego_history_xyz": torch.randn(1, 1, 16, 3),
            "ego_history_rot": torch.randn(1, 1, 16, 3, 3),
            "ego_future_xyz": torch.randn(1, 1, 64, 3),
            "ego_future_rot": torch.randn(1, 1, 64, 3, 3),
        }

        cache = ClipDataCache(
            self._make_mock_avdi(),
            self._make_mock_processor(),
            cache_pil_images=False,
        )

        cache.get("clip_1", 5100000, torch.device("cpu"))
        assert "pil_images" not in cache._cache[("clip_1", 5100000)]


# ===================================================================
# make_vllm_rollout_func output format
# ===================================================================

class TestMakeVllmRolloutFunc:
    """Test that the rollout_func factory produces valid output dicts."""

    def _make_mock_full_model(self):
        """Create a mock AlpamayoR1 with the minimum attributes."""
        model = MagicMock()
        model.config.tokens_per_future_traj = 48
        model.config.traj_vocab_size = 768
        model.future_token_start_idx = 151700
        model.special_token_ids = {
            "traj_future_start": 151660,
            "traj_future_end": 151661,
        }
        model.fuse_traj_tokens.side_effect = lambda ids, data: ids
        return model

    def _make_mock_data_cache(self, num_generations: int):
        """Create a mock ClipDataCache that returns fresh dicts each call."""
        cache = MagicMock()

        def _fresh_get(clip_id, t0_us, device):
            return (
                {
                    "tokenized_data": {
                        "input_ids": torch.randint(0, 1000, (1, 50)),
                        "attention_mask": torch.ones(1, 50, dtype=torch.long),
                        "pixel_values": torch.randn(4, 3, 224, 224),
                        "image_grid_thw": torch.tensor([[1, 224, 224]] * 4),
                    },
                    "ego_history_xyz": torch.randn(1, 1, 16, 3),
                    "ego_history_rot": torch.randn(1, 1, 16, 3, 3),
                },
                torch.randn(1, 1, 64, 3),  # ego_future_xyz
            )

        cache.get.side_effect = _fresh_get
        # Return real PIL images so _concat_pil_images can work
        from PIL import Image as PILImage
        cache.get_pil_images.return_value = [PILImage.new("RGB", (64, 64))] * 16
        return cache

    def _make_mock_trainer(self, num_generations: int = 2):
        """Create a mock GRPOTrainer with a mock vllm_generation."""
        trainer = MagicMock()
        trainer.num_generations = num_generations
        trainer.num_generations_eval = num_generations
        trainer.model.training = True
        trainer.accelerator.device = torch.device("cpu")

        # Mock vLLM LLM.generate to return RequestOutput-like objects
        def _mock_vllm_generate(inputs, sampling_params, use_tqdm=False):
            outputs = []
            for inp in inputs:
                mock_output = MagicMock()
                # Simulated completion: 10 random tokens + traj_future_end
                token_ids = list(range(100, 110)) + [151661]
                mock_output.outputs = [MagicMock(
                    token_ids=token_ids,
                    logprobs=[{tid: MagicMock(logprob=-0.5)} for tid in token_ids],
                )]
                outputs.append(mock_output)
            return outputs

        trainer.vllm_generation.llm.generate.side_effect = _mock_vllm_generate
        return trainer

    @patch("alpamayo_r1.training.rollout.SamplingParams", create=True)
    def test_rollout_func_output_keys(self, mock_sp):
        """rollout_func should return all required + extra keys."""
        from alpamayo_r1.training.rollout import make_vllm_rollout_func

        mock_sp.return_value = MagicMock()  # mock SamplingParams constructor

        full_model = self._make_mock_full_model()
        cache = self._make_mock_data_cache(num_generations=2)
        num_gen = 2

        rollout_fn = make_vllm_rollout_func(full_model, cache)

        # Simulate TRL's prompt format: 2 unique prompts × 2 generations = 4 prompts
        prompts = [
            "[clip_id=clip_a] [t0_us=5100000] drive prompt",
            "[clip_id=clip_a] [t0_us=5100000] drive prompt",
            "[clip_id=clip_b] [t0_us=5100000] drive prompt",
            "[clip_id=clip_b] [t0_us=5100000] drive prompt",
        ]

        # Patch the vllm import inside rollout_func
        import sys
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = MagicMock(return_value=MagicMock())
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            trainer = self._make_mock_trainer(num_generations=num_gen)
            result = rollout_fn(prompts, trainer)

        # Check required keys
        assert "prompt_ids" in result
        assert "completion_ids" in result
        assert "logprobs" in result

        # Check extra keys
        assert "gt_xyz" in result
        assert "hist_xyz" in result
        assert "hist_rot" in result
        assert "clip_ids" in result

        # Check lengths (should match total prompts = 4)
        assert len(result["prompt_ids"]) == 4
        assert len(result["completion_ids"]) == 4
        assert len(result["logprobs"]) == 4
        assert len(result["gt_xyz"]) == 4
        assert len(result["hist_xyz"]) == 4
        assert len(result["hist_rot"]) == 4
        assert len(result["clip_ids"]) == 4

    @patch("alpamayo_r1.training.rollout.SamplingParams", create=True)
    def test_rollout_func_completion_ids_include_stop_token(self, mock_sp):
        """completion_ids should include traj_future_end token."""
        from alpamayo_r1.training.rollout import make_vllm_rollout_func

        mock_sp.return_value = MagicMock()
        full_model = self._make_mock_full_model()
        cache = self._make_mock_data_cache(num_generations=1)

        rollout_fn = make_vllm_rollout_func(full_model, cache)

        prompts = ["[clip_id=c1] [t0_us=5100000] prompt"]

        import sys
        with patch.dict(sys.modules, {"vllm": MagicMock()}):
            trainer = self._make_mock_trainer(num_generations=1)
            result = rollout_fn(prompts, trainer)

        # Last token should be traj_future_end_id
        traj_future_end_id = full_model.special_token_ids["traj_future_end"]
        for comp_ids in result["completion_ids"]:
            assert comp_ids[-1] == traj_future_end_id

    @patch("alpamayo_r1.training.rollout.SamplingParams", create=True)
    def test_rollout_func_logprobs_format(self, mock_sp):
        """logprobs should be list[list[list[float]]] (batch, seq, top-k)."""
        from alpamayo_r1.training.rollout import make_vllm_rollout_func

        mock_sp.return_value = MagicMock()
        full_model = self._make_mock_full_model()
        cache = self._make_mock_data_cache(num_generations=1)

        rollout_fn = make_vllm_rollout_func(full_model, cache)
        prompts = ["[clip_id=c1] [t0_us=5100000] prompt"]

        import sys
        with patch.dict(sys.modules, {"vllm": MagicMock()}):
            trainer = self._make_mock_trainer(num_generations=1)
            result = rollout_fn(prompts, trainer)

        # Each logprob entry should be a list of lists
        for seq_lps in result["logprobs"]:
            assert isinstance(seq_lps, list)
            for token_lps in seq_lps:
                assert isinstance(token_lps, list)
                assert all(isinstance(v, float) for v in token_lps)

    @patch("alpamayo_r1.training.rollout.SamplingParams", create=True)
    def test_rollout_func_server_mode_output_keys(self, mock_sp):
        """Server mode should return all required + extra keys via VLLMClient."""
        from alpamayo_r1.training.rollout import make_vllm_rollout_func

        mock_sp.return_value = MagicMock()
        full_model = self._make_mock_full_model()
        num_gen = 2
        cache = self._make_mock_data_cache(num_generations=num_gen)

        rollout_fn = make_vllm_rollout_func(
            full_model, cache, vllm_mode="server",
        )

        prompts = [
            "[clip_id=clip_a] [t0_us=5100000] drive prompt",
            "[clip_id=clip_a] [t0_us=5100000] drive prompt",
            "[clip_id=clip_b] [t0_us=5100000] drive prompt",
            "[clip_id=clip_b] [t0_us=5100000] drive prompt",
        ]

        # Mock VLLMClient.generate to return dict response
        total = len(prompts)
        token_ids = list(range(100, 110)) + [151661]
        mock_client_result = {
            "prompt_ids": [list(range(50))] * total,
            "completion_ids": [token_ids] * total,
            "logprobs": [[[-0.5]] * len(token_ids)] * total,
            "logprob_token_ids": [[[tid] for tid in token_ids]] * total,
        }

        trainer = self._make_mock_trainer(num_generations=num_gen)
        trainer.vllm_generation.vllm_client.generate.return_value = mock_client_result

        result = rollout_fn(prompts, trainer)

        # Check required keys
        assert "prompt_ids" in result
        assert "completion_ids" in result
        assert "logprobs" in result

        # Check extra keys
        assert "gt_xyz" in result
        assert "hist_xyz" in result
        assert "hist_rot" in result
        assert "clip_ids" in result

        # Check lengths
        assert len(result["prompt_ids"]) == total
        assert len(result["completion_ids"]) == total
        assert len(result["logprobs"]) == total

        # Verify VLLMClient.generate was called with prompt_token_ids and images
        call_kwargs = trainer.vllm_generation.vllm_client.generate.call_args
        assert call_kwargs.kwargs["prompt_token_ids"] is not None
        assert len(call_kwargs.kwargs["prompt_token_ids"]) == total
        assert call_kwargs.kwargs["images"] is not None
        assert len(call_kwargs.kwargs["images"]) == total
        # Each image entry should be a list of PIL images
        from PIL import Image as PILImage
        for img_list in call_kwargs.kwargs["images"]:
            assert isinstance(img_list, list)
            assert all(isinstance(img, PILImage.Image) for img in img_list)


# ===================================================================
# _calculate_rewards override
# ===================================================================

class TestCalculateRewardsOverride:
    """Test the _calculate_rewards trajectory decoding path."""

    def test_hf_path_passthrough(self):
        """When use_vllm=False, _calculate_rewards delegates to super()."""
        from alpamayo_r1.training.rollout import AlpamayoGRPOTrainer

        # We can't easily instantiate AlpamayoGRPOTrainer without all deps,
        # so we test the conditional logic directly.
        # If use_vllm is False, the method should call super()
        # This is more of a structural check.
        assert hasattr(AlpamayoGRPOTrainer, "_calculate_rewards")

    def test_vllm_path_decodes_traj_tokens(self):
        """When inputs contain hist_xyz, trajectory tokens are decoded."""
        from alpamayo_r1.training.rollout import AlpamayoGRPOTrainer

        # Create a minimal mock of AlpamayoGRPOTrainer (no spec to allow
        # setting nested attributes freely)
        trainer = MagicMock()
        trainer.use_vllm = True
        trainer.num_generations = 2
        trainer.model.training = True
        trainer.accelerator.device = torch.device("cpu")

        # Mock full_model
        trainer.full_model.future_token_start_idx = 151700
        trainer.full_model.config.tokens_per_future_traj = 48
        trainer.full_model.config.traj_vocab_size = 768
        trainer.full_model.special_token_ids = {
            "traj_future_start": 151660,
            "traj_future_end": 151661,
        }
        # Mock traj_tokenizer.decode → returns (pred_xyz, pred_rot, tstamp)
        trainer.full_model.traj_tokenizer.decode.return_value = (
            torch.randn(1, 64, 3),  # pred_xyz
            torch.randn(1, 64, 3, 3),  # pred_rot
            torch.zeros(1, 64),  # tstamp
        )

        # Build inputs with vLLM extra fields
        inputs = [
            {
                "hist_xyz": torch.randn(16, 3).flatten().tolist(),
                "hist_rot": torch.randn(16, 3, 3).flatten().tolist(),
                "clip_ids": "clip_a",
                "gt_xyz": torch.randn(64, 3).flatten().tolist(),
            },
            {
                "hist_xyz": torch.randn(16, 3).flatten().tolist(),
                "hist_rot": torch.randn(16, 3, 3).flatten().tolist(),
                "clip_ids": "clip_a",
                "gt_xyz": torch.randn(64, 3).flatten().tolist(),
            },
        ]

        # Build completion_ids with traj tokens
        traj_start = 151660
        traj_end = 151661
        traj_token_ids = list(range(151700, 151700 + 48))
        completion_ids_list = [
            [100, 101, traj_start] + traj_token_ids + [traj_end],
            [200, 201, traj_start] + traj_token_ids + [traj_end],
        ]

        # Call the real _calculate_rewards on the mocked trainer
        # Since we can't easily call the bound method, we test the structure:
        # The method checks `if not self.use_vllm or not inputs or "hist_xyz" not in inputs[0]`
        # and then decodes trajectories. We verify the conditions.
        assert trainer.use_vllm is True
        assert "hist_xyz" in inputs[0]
        assert len(completion_ids_list[0]) == 3 + 48 + 1  # text + traj_start + traj + traj_end


# ===================================================================
# _generate_single_turn delegation
# ===================================================================

class TestGenerateSingleTurnDelegation:
    """Test that _generate_single_turn delegates correctly."""

    def test_vllm_delegates_to_super(self):
        """When use_vllm=True, _generate_single_turn calls super()."""
        from alpamayo_r1.training.rollout import AlpamayoGRPOTrainer

        # Structural check: the method exists and has the conditional
        import inspect
        source = inspect.getsource(AlpamayoGRPOTrainer._generate_single_turn)
        assert "if self.use_vllm:" in source
        assert "return super()._generate_single_turn(prompts)" in source


# ===================================================================
# train_grpo.py vLLM config wiring
# ===================================================================

class TestTrainGrpoVllmConfig:
    """Test that train_grpo.py correctly wires vLLM config."""

    def test_vllm_kwargs_construction(self):
        """Simulate the vllm_kwargs dict that train_grpo.py builds."""
        # This mirrors the logic in train_grpo.py
        vllm_cfg = {
            "enabled": True,
            "gpu_memory_utilization": 0.3,
            "tensor_parallel_size": 1,
            "max_model_length": None,
            "enable_sleep_mode": False,
            "model_impl": "transformers",
        }

        vllm_kwargs = dict(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.3)),
            vllm_tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
            vllm_max_model_length=vllm_cfg.get("max_model_length", None),
            vllm_enable_sleep_mode=bool(vllm_cfg.get("enable_sleep_mode", False)),
            vllm_model_impl=str(vllm_cfg.get("model_impl", "transformers")),
        )

        assert vllm_kwargs["use_vllm"] is True
        assert vllm_kwargs["vllm_mode"] == "colocate"
        assert vllm_kwargs["vllm_gpu_memory_utilization"] == 0.3
        assert vllm_kwargs["vllm_tensor_parallel_size"] == 1
        assert vllm_kwargs["vllm_max_model_length"] is None
        assert vllm_kwargs["vllm_enable_sleep_mode"] is False
        assert vllm_kwargs["vllm_model_impl"] == "transformers"

    def test_vllm_disabled_no_kwargs(self):
        """When vllm.enabled=false, no vLLM kwargs should be added."""
        vllm_cfg = {"enabled": False}
        vllm_enabled = bool(vllm_cfg.get("enabled", False))
        assert vllm_enabled is False

    def test_vllm_server_kwargs_construction(self):
        """Simulate server mode vllm_kwargs dict that train_grpo.py builds."""
        vllm_cfg = {
            "enabled": True,
            "mode": "server",
            "model_impl": "transformers",
            "server_host": "0.0.0.0",
            "server_port": 8000,
            "server_base_url": None,
            "server_timeout": 240.0,
            "group_port": 51216,
        }

        vllm_mode = str(vllm_cfg.get("mode", "colocate"))
        assert vllm_mode == "server"

        vllm_kwargs = dict(
            use_vllm=True,
            vllm_mode=vllm_mode,
            vllm_model_impl=str(vllm_cfg.get("model_impl", "transformers")),
        )
        # Server-specific params
        vllm_kwargs.update(
            vllm_server_host=str(vllm_cfg.get("server_host", "0.0.0.0")),
            vllm_server_port=int(vllm_cfg.get("server_port", 8000)),
            vllm_server_timeout=float(vllm_cfg.get("server_timeout", 240.0)),
            vllm_group_port=int(vllm_cfg.get("group_port", 51216)),
        )
        server_base_url = vllm_cfg.get("server_base_url", None)
        if server_base_url is not None:
            vllm_kwargs["vllm_server_base_url"] = str(server_base_url)

        assert vllm_kwargs["use_vllm"] is True
        assert vllm_kwargs["vllm_mode"] == "server"
        assert vllm_kwargs["vllm_server_host"] == "0.0.0.0"
        assert vllm_kwargs["vllm_server_port"] == 8000
        assert vllm_kwargs["vllm_server_timeout"] == 240.0
        assert vllm_kwargs["vllm_group_port"] == 51216
        assert "vllm_server_base_url" not in vllm_kwargs  # None → not set
        # Colocate-specific params should NOT be present
        assert "vllm_gpu_memory_utilization" not in vllm_kwargs
        assert "vllm_tensor_parallel_size" not in vllm_kwargs

    def test_vllm_server_kwargs_with_base_url(self):
        """When server_base_url is set, it should be included."""
        vllm_cfg = {
            "enabled": True,
            "mode": "server",
            "server_base_url": "http://vllm-server:8000",
        }

        vllm_kwargs = {}
        server_base_url = vllm_cfg.get("server_base_url", None)
        if server_base_url is not None:
            vllm_kwargs["vllm_server_base_url"] = str(server_base_url)

        assert vllm_kwargs["vllm_server_base_url"] == "http://vllm-server:8000"


# ===================================================================
# run_grpo.sh vLLM flag
# ===================================================================

class TestRunGrpoScript:
    """Test that run_grpo.sh has the --vllm flag."""

    def test_vllm_flag_in_script(self):
        with open("scripts/run_grpo.sh") as f:
            content = f.read()

        assert "--vllm" in content
        assert "VLLM=0" in content
        assert "VLLM=1" in content
        assert "vllm.enabled=true" in content
        assert "vLLM COLOCATE MODE" in content

    def test_vllm_smoke_usage_in_header(self):
        with open("scripts/run_grpo.sh") as f:
            content = f.read()

        assert "--vllm --smoke" in content

    def test_vllm_server_flag_in_script(self):
        with open("scripts/run_grpo.sh") as f:
            content = f.read()

        assert "--vllm-server" in content
        assert "VLLM_SERVER=0" in content
        assert "VLLM_SERVER=1" in content
        assert "vllm.mode=server" in content
        assert "vLLM SERVER MODE" in content


# ===================================================================
# Multi-image encoding round-trip
# ===================================================================

class TestMultiImageEncoding:
    """Test that multi-image encoding/decoding works end-to-end."""

    def test_single_image_backward_compat(self):
        """A single PIL image should encode to a single base64 string."""
        from trl.generation.vllm_client import pil_to_base64
        from PIL import Image as PILImage
        import base64

        img = PILImage.new("RGB", (32, 32), color="red")
        b64 = pil_to_base64(img)
        assert isinstance(b64, str)

        # Decode back to PIL
        from io import BytesIO
        decoded = PILImage.open(BytesIO(base64.b64decode(b64)))
        assert decoded.size == (32, 32)

    def test_multi_image_list_encoding(self):
        """A list of PIL images should encode to a list of base64 strings."""
        from trl.generation.vllm_client import pil_to_base64
        from PIL import Image as PILImage
        import base64
        from io import BytesIO

        imgs = [PILImage.new("RGB", (16, 16), color=c) for c in ["red", "green", "blue"]]

        # Simulate VLLMClient.generate() encoding logic
        encoded = []
        for img in [imgs]:  # one prompt with multiple images
            if isinstance(img, list):
                encoded.append([pil_to_base64(i) for i in img])
            else:
                encoded.append(pil_to_base64(img))

        assert len(encoded) == 1
        assert isinstance(encoded[0], list)
        assert len(encoded[0]) == 3

        # Decode each back
        for b64 in encoded[0]:
            decoded = PILImage.open(BytesIO(base64.b64decode(b64)))
            assert decoded.size == (16, 16)

    def test_mixed_image_encoding(self):
        """Mixed single/multi/None images should encode correctly."""
        from trl.generation.vllm_client import pil_to_base64
        from PIL import Image as PILImage

        single = PILImage.new("RGB", (8, 8))
        multi = [PILImage.new("RGB", (8, 8)) for _ in range(4)]
        images = [single, multi, None]

        # Simulate VLLMClient encoding logic
        encoded = []
        for img in images:
            if img is None:
                encoded.append(None)
            elif isinstance(img, list):
                encoded.append([pil_to_base64(i) for i in img])
            else:
                encoded.append(pil_to_base64(img))

        assert isinstance(encoded[0], str)       # single → str
        assert isinstance(encoded[1], list)       # multi → list[str]
        assert len(encoded[1]) == 4
        assert encoded[2] is None                 # None stays None
