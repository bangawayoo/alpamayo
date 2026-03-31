"""Iterative self-play loop for advantage-conditioned SFT.

Orchestrates three phases per iteration:
1. ROLLOUT: Generate G completions per scene from current policy
2. EVALUATE: Score with rewards + value head, compute and binarize advantages
3. TRAIN: Build advantage-conditioned dataset, SFT + expert CFM

By default, each iteration continues from the previous iteration's trained
weights (LoRA merged into base). Set ``reset_to_base: true`` to reload pi_0
every iteration (RECAP-style reset-to-checkpoint).

Manages strict data partitioning (each scene used for fresh rollouts in exactly
one iteration) and a replay buffer for historical rollouts.

See docs/advantage-conditioning.md for the full design specification.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import random
import time
from pathlib import Path

from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from physical_ai_av import PhysicalAIAVDatasetInterface
from transformers import TrainingArguments

from alpamayo_r1.training.advantage_conditioning import (
    AdvantageBuffer,
    AdvantageEmbedding,
    AdvCondDataset,
    compute_advantage_token_ids,
    compute_segment_advantages_from_rollouts,
    compute_value_targets,
    precompute_conditioned_sequences,
    train_segment_value_head,
)
from alpamayo_r1.training.profiler import StageProfiler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data partitioning
# ---------------------------------------------------------------------------


class ScenePartitioner:
    """Partition dataset scenes across iterations to prevent overfitting.

    Each scene (clip_id) is assigned to exactly one iteration for fresh
    rollouts. After that iteration, it is only available as historical
    replay data. This prevents the model from overfitting to the same
    driving clips across iterations.

    Args:
        all_clip_ids: Full list of clip IDs from the dataset.
        num_iterations: Number of self-play iterations.
        seed: Random seed for shuffling.
    """

    def __init__(self, all_clip_ids: list[str], num_iterations: int, seed: int = 42) -> None:
        self.num_iterations = num_iterations

        # Shuffle and split into N approximately equal chunks
        rng = random.Random(seed)
        shuffled = list(all_clip_ids)
        rng.shuffle(shuffled)

        # np.array_split handles uneven division gracefully
        if num_iterations > 0:
            chunks = np.array_split(shuffled, num_iterations)
            self.partitions: list[list[str]] = [chunk.tolist() for chunk in chunks]
        else:
            # num_iterations=0: pre-training only, no partitioning needed
            self.partitions = []

        logger.info(
            "Partitioned %d scenes across %d iterations: %s",
            len(all_clip_ids),
            num_iterations,
            [len(p) for p in self.partitions],
        )

    def get_fresh_scenes(self, iteration: int) -> list[str]:
        """Scenes for fresh rollouts at this iteration (never seen before)."""
        if iteration >= self.num_iterations:
            raise IndexError(f"Iteration {iteration} >= num_iterations {self.num_iterations}")
        return self.partitions[iteration]

    def get_historical_scenes(self, iteration: int) -> list[str]:
        """All scenes from previous iterations (available for replay only)."""
        return [cid for i in range(iteration) for cid in self.partitions[i]]


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class RolloutReplayBuffer:
    """Stores rollout results + advantage labels from all past iterations.

    Rollouts from previous iterations are available as historical training
    data, mixed in at a configurable replay_ratio. Advantage labels can be
    recomputed with the current value head to avoid staleness.

    Args:
        max_size: Maximum number of rollout entries.
    """

    def __init__(self, max_size: int = 50000) -> None:
        self._entries: list[dict] = []
        self._max_size = max_size

    def add(
        self,
        rollout_results: list[dict],
        adv_labels: list[dict],
        iteration: int,
    ) -> None:
        """Add current iteration's rollouts to the buffer."""
        for rollout, label in zip(rollout_results, adv_labels):
            if len(self._entries) >= self._max_size:
                # Evict oldest entry
                self._entries.pop(0)
            self._entries.append(
                {
                    "rollout": rollout,
                    "adv_label": label,
                    "iteration": iteration,
                }
            )
        logger.info(
            "Added %d rollouts from iter %d to replay buffer (total: %d)",
            len(rollout_results),
            iteration,
            len(self._entries),
        )

    def sample(self, n: int, rng: random.Random | None = None) -> tuple[list[dict], list[dict]]:
        """Sample n historical entries.

        Args:
            n: Number of entries to sample.
            rng: Random number generator (for reproducibility).

        Returns:
            (rollout_results, adv_labels) tuple of sampled entries.
        """
        if not self._entries:
            return [], []
        rng = rng or random.Random()
        n = min(n, len(self._entries))
        sampled = rng.sample(self._entries, n)
        rollouts = [e["rollout"] for e in sampled]
        labels = [e["adv_label"] for e in sampled]
        return rollouts, labels

    def recompute_labels(
        self,
        advantage_buffer: AdvantageBuffer,
        segment_hidden_stash: list[dict] | None = None,
        segment_reward_stash: list[dict] | None = None,
        completion_segment_map: list[dict] | None = None,
        value_head: torch.nn.Module | None = None,
        reward_weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
    ) -> None:
        """Recompute advantage labels for all buffer entries using current value head.

        This prevents stale advantage labels from degrading training quality
        as the value head improves across iterations.
        """
        if value_head is None or segment_hidden_stash is None:
            logger.debug("No value head or hidden stash — skipping label recomputation")
            return

        # Recompute advantages (return minus baseline)
        new_advantages = compute_segment_advantages_from_rollouts(
            segment_hidden_stash=segment_hidden_stash,
            segment_reward_stash=segment_reward_stash,
            completion_segment_map=completion_segment_map,
            value_head=value_head,
            reward_weights=reward_weights,
        )

        # Update labels in buffer (skip GT-augmented entries — keep i_traj=True)
        n_recomputed = 0
        for entry, adv in zip(self._entries, new_advantages):
            if entry["rollout"].get("is_gt_augmented", False):
                continue
            i_obs, i_traj = advantage_buffer.binarize(adv["a_obs"], adv["a_traj"])
            entry["adv_label"] = {"i_obs": i_obs, "i_traj": i_traj}
            n_recomputed += 1

        n_gt = len(self._entries) - n_recomputed
        logger.info(
            "Recomputed advantage labels for %d buffer entries (%d GT-augmented skipped)",
            n_recomputed,
            n_gt,
        )

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Checkpoint loading utilities
# ---------------------------------------------------------------------------


def load_value_head_checkpoint(
    path: str | Path,
    hidden_dim: int = 4096,
) -> torch.nn.Module:
    """Load a saved SegmentValueHead checkpoint.

    Args:
        path: Path to the ``value_head.pt`` file.
        hidden_dim: Hidden dimension (must match the saved checkpoint).

    Returns:
        SegmentValueHead module with loaded weights.
    """
    from alpamayo_r1.training.value_head import SegmentValueHead

    value_head = SegmentValueHead(hidden_dim=hidden_dim)
    state = torch.load(path, map_location="cpu", weights_only=False)
    value_head.load_state_dict(state)
    logger.info("Loaded value head from %s", path)
    return value_head


def load_expert_checkpoint(
    full_model: Any,
    path: str | Path,
) -> Any:
    """Load expert + projection weights from a checkpoint.

    Supports two layouts:

    * **New (PEFT)**: ``expert_adapter/`` directory alongside the checkpoint,
      saved via ``PeftModel.save_pretrained``.  Loaded with
      ``PeftModel.from_pretrained`` — config is embedded in adapter_config.json.
    * **Legacy**: All weights (full or LoRA) inside ``expert_checkpoint.pt``.

    Projection layers (``action_in_proj``, ``action_out_proj``) are always
    loaded from ``expert_checkpoint.pt``.

    Args:
        full_model: AlpamayoR1 instance (mutated in place).
        path: Path to ``expert_checkpoint.pt``.

    Returns:
        The same ``full_model`` reference (mutated).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    adapter_dir = Path(path).parent / "expert_adapter"
    if adapter_dir.exists():
        # New format: PEFT adapter saved via save_pretrained
        from peft import PeftModel

        full_model.expert = PeftModel.from_pretrained(full_model.expert, str(adapter_dir))
        logger.info("Loaded expert LoRA adapter from %s", adapter_dir)
    elif checkpoint.get("expert_lora", False):
        # Legacy format: LoRA weights embedded in expert_checkpoint.pt.
        # Read config from resolved_config.yaml in the output directory.
        # remove this after next training run
        import yaml

        config_path = Path(path).parent.parent.parent / "resolved_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Legacy expert LoRA checkpoint at {path} requires "
                f"resolved_config.yaml at {config_path} for LoRA config"
            )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        expert_lora_cfg = cfg.get("expert_finetune", {}).get("expert_lora", {})

        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=int(expert_lora_cfg["r"]),
            lora_alpha=int(expert_lora_cfg["alpha"]),
            lora_dropout=float(expert_lora_cfg.get("dropout", 0.0)),
            target_modules=list(expert_lora_cfg["target_modules"]),
            task_type="FEATURE_EXTRACTION",
        )
        full_model.expert = get_peft_model(full_model.expert, lora_config)
        full_model.expert.load_state_dict(checkpoint["expert"], strict=False)
        logger.info(
            "Loaded legacy expert LoRA weights from %s (r=%d, alpha=%d)",
            path,
            lora_config.r,
            lora_config.lora_alpha,
        )
    elif "expert" in checkpoint:
        full_model.expert.load_state_dict(checkpoint["expert"], strict=False)
        logger.info("Loaded full expert weights from %s", path)

    full_model.action_in_proj.load_state_dict(checkpoint["action_in_proj"])
    full_model.action_out_proj.load_state_dict(checkpoint["action_out_proj"])
    logger.info("Loaded projection layers from %s", path)
    return full_model


def load_adv_embedding_checkpoint(
    full_model: Any,
    path: str | Path,
) -> Any:
    """Load AdvantageEmbedding from a saved .pt checkpoint.

    Args:
        full_model: AlpamayoR1 instance (used to derive vocab_size/hidden_size).
        path: Path to the ``adv_embedding.pt`` file.

    Returns:
        AdvantageEmbedding module with loaded weights.
    """
    from alpamayo_r1.training.advantage_conditioning import (
        AdvantageEmbedding,
        compute_advantage_token_ids,
    )

    vocab_size = full_model.vlm.config.text_config.vocab_size
    hidden_size = full_model.vlm.config.text_config.hidden_size
    adv_token_ids = compute_advantage_token_ids(vocab_size)
    adv_embedding = AdvantageEmbedding(hidden_size, adv_token_ids)
    state = torch.load(path, map_location="cpu", weights_only=False)
    adv_embedding.load_state_dict(state)
    logger.info("Loaded advantage embedding from %s", path)
    return adv_embedding


def _is_expert_lora_checkpoint(iter_dir: Path) -> bool:
    """Check whether an iteration directory has an expert LoRA checkpoint.

    Checks for the new ``expert_adapter/`` dir first, then falls back to the
    ``expert_lora`` flag inside ``expert_checkpoint.pt``.
    """
    if (iter_dir / "expert_adapter" / "adapter_config.json").exists():
        return True
    pt_path = iter_dir / "expert_checkpoint.pt"
    if pt_path.exists():
        probe = torch.load(pt_path, map_location="cpu", weights_only=False)
        return probe.get("expert_lora", False)
    return False


def load_vlm_from_iterations(
    base_model_name: str,
    output_dir: str | Path,
    up_to_iteration: int,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = None,
    merge: bool = True,
) -> Any:
    """Reconstruct a model by iteratively merging per-iteration LoRA adapters.

    Starting from the base model, loads and merges VLM LoRA adapters from
    ``iter_0/final/`` through ``iter_{up_to_iteration}/final/``. Expert LoRA
    checkpoints are also iteratively applied and merged; non-LoRA expert
    checkpoints are loaded from the latest iteration only (they contain full
    weights). Value head and advantage embedding are loaded from the final
    iteration.

    Args:
        base_model_name: HuggingFace model name or path for the base AlpamayoR1.
        output_dir: Root output directory containing ``iter_*/final/`` subdirs.
        up_to_iteration: Last iteration index (inclusive) to merge.
        dtype: Model dtype.
        device_map: Device map for model loading (e.g. ``"auto"``).

    Returns:
        Dict with keys ``full_model``, ``value_head`` (or None),
        ``adv_embedding`` (or None).
    """
    from peft import PeftModel

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    output_dir = Path(output_dir)

    logger.info("Loading base model from %s", base_model_name)
    kwargs = {}
    if device_map is not None:
        kwargs["device_map"] = device_map
    full_model = AlpamayoR1.from_pretrained(base_model_name, dtype=dtype, **kwargs)

    # Iteratively apply and merge VLM LoRA adapters
    for i in range(up_to_iteration + 1):
        adapter_path = output_dir / f"iter_{i}" / "final"
        adapter_config = adapter_path / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(
                f"Expected VLM adapter at {adapter_path} for iteration {i}, "
                f"but adapter_config.json not found"
            )
        logger.info("Loading VLM LoRA adapter from %s", adapter_path)
        full_model.vlm = PeftModel.from_pretrained(full_model.vlm, str(adapter_path))
        if merge:
            full_model.vlm = full_model.vlm.to(torch.float32)
            full_model.vlm = full_model.vlm.merge_and_unload()
            full_model.vlm = full_model.vlm.to(dtype)
            logger.info("Merged VLM LoRA from iteration %d", i)
        else:
            logger.info("Loaded VLM LoRA from iteration %d (unmerged)", i)

    # Load expert checkpoint(s). For LoRA-based expert, iteratively apply and
    # merge from each iteration (same stacking logic as VLM LoRA). For non-LoRA
    # expert, just load the latest iteration's full weights.
    latest_dir = output_dir / f"iter_{up_to_iteration}" / "final"
    latest_expert = latest_dir / "expert_checkpoint.pt"
    if not latest_expert.exists():
        raise FileNotFoundError(f"Expected expert checkpoint at {latest_expert} but file not found")
    if _is_expert_lora_checkpoint(latest_dir):
        # Expert uses LoRA — stack from iter_0 through iter_N
        for i in range(up_to_iteration + 1):
            iter_final = output_dir / f"iter_{i}" / "final"
            expert_path = iter_final / "expert_checkpoint.pt"
            if not expert_path.exists():
                raise FileNotFoundError(
                    f"Expected expert checkpoint at {expert_path} for iteration {i}, "
                    f"but file not found"
                )
            load_expert_checkpoint(full_model, expert_path)
            if hasattr(full_model.expert, "merge_and_unload"):
                full_model.expert = full_model.expert.merge_and_unload()
            if hasattr(full_model.expert, "peft_config"):
                delattr(full_model.expert, "peft_config")
            logger.info("Merged expert LoRA from iteration %d", i)
    else:
        # Non-LoRA expert — full weights from the latest iteration
        load_expert_checkpoint(full_model, latest_expert)

    # Load value head from the latest iteration
    value_head = None
    latest_vh = latest_dir / "value_head.pt"
    if latest_vh.exists():
        value_head = load_value_head_checkpoint(latest_vh)

    # Load advantage embedding from the latest iteration
    adv_embedding = None
    latest_adv = latest_dir / "adv_embedding.pt"
    if latest_adv.exists():
        adv_embedding = load_adv_embedding_checkpoint(full_model, latest_adv)

    logger.info(
        "Reconstructed model through iteration %d (VLM merged, expert loaded, vh=%s, adv=%s)",
        up_to_iteration,
        value_head is not None,
        adv_embedding is not None,
    )
    return {"full_model": full_model, "value_head": value_head, "adv_embedding": adv_embedding}


# ---------------------------------------------------------------------------
# Self-play loop
# ---------------------------------------------------------------------------


class SelfPlayLoop:
    """Orchestrate the iterative rollout -> evaluate -> train loop.

    Each iteration:
    1. Generate G completions per fresh scene (never used before)
    2. Score and binarize advantages
    3. Build advantage-conditioned dataset, train SFT + expert

    Args:
        cfg: Hydra config dict.
        full_model: AlpamayoR1 instance.
        avdi: PhysicalAI-AV dataset interface.
        processor: HuggingFace processor/tokenizer.
        all_clip_ids: Full list of clip IDs from the dataset.
    """

    def __init__(
        self,
        cfg: dict,
        full_model: Any,
        avdi: PhysicalAIAVDatasetInterface,
        processor: Any,
        all_clip_ids: list[str],
    ) -> None:
        self.cfg = cfg
        self.full_model = full_model
        self.avdi = avdi
        self.processor = processor
        self.all_clip_ids = list(all_clip_ids)
        self._pretrain_clip_ids: list[str] = []

        adv_cfg = cfg.get("advantage_conditioning", {})
        num_iterations = int(adv_cfg.get("num_iterations", 5))
        self.num_iterations = num_iterations

        # If a pretrained value head is being loaded, also load the clip IDs
        # used during pre-training so they can be excluded from partitioning.
        vh_cfg = cfg.get("value_head", {})
        load_path = vh_cfg.get("load_path")
        if load_path:
            clip_ids_path = Path(load_path).parent / "clip_ids.json"
            if clip_ids_path.exists():
                with open(clip_ids_path) as f:
                    self._pretrain_clip_ids = json.load(f)
                logger.info(
                    "Loaded %d pretrain clip IDs from %s — excluding from partitioner",
                    len(self._pretrain_clip_ids),
                    clip_ids_path,
                )
            else:
                logger.warning(
                    "Value head load_path set but no clip_ids.json found at %s",
                    clip_ids_path,
                )

        partitioner_clip_ids = [
            cid for cid in all_clip_ids if cid not in set(self._pretrain_clip_ids)
        ]
        self.partitioner = ScenePartitioner(
            partitioner_clip_ids, num_iterations, seed=cfg.get("seed", 42)
        )
        self.replay_buffer = RolloutReplayBuffer(
            max_size=int(adv_cfg.get("replay_buffer_max_size", 50000))
        )
        self.advantage_buffer = AdvantageBuffer(
            k_obs=float(adv_cfg.get("k_obs", 30)),
            k_traj=float(adv_cfg.get("k_traj", 30)),
            ema_alpha=float(adv_cfg.get("ema_alpha", 0.99)),
        )

        self.base_model_path = cfg.get("base_model_name", None) or cfg.get(
            "model_name", "nvidia/Alpamayo-R1-10B"
        )
        self.current_policy_path = self.base_model_path

        # Advantage conditioning: compute token IDs and (optionally) trainable embedding
        self.adv_enabled = bool(adv_cfg.get("enabled", True))
        self.adv_mode = adv_cfg.get("adv_mode", "embedding")
        if self.adv_enabled:
            if self.adv_mode == "text":
                from alpamayo_r1.training.advantage_conditioning import (
                    compute_text_advantage_token_ids,
                )

                self.adv_token_ids = compute_text_advantage_token_ids(full_model.tokenizer)
                self.adv_embedding = None
                logger.info("Advantage conditioning: text mode (plain-text tokens)")
            else:
                vocab_size = full_model.vlm.config.text_config.vocab_size
                self.adv_token_ids = compute_advantage_token_ids(vocab_size)
                hidden_size = full_model.vlm.config.text_config.hidden_size
                self.adv_embedding = AdvantageEmbedding(hidden_size, self.adv_token_ids)
                logger.info("Advantage conditioning: embedding mode (learned hooks)")
        else:
            self.adv_token_ids = None
            self.adv_embedding = None
            logger.info("Advantage conditioning disabled — training as plain SFT")

        # Replay ratio: fraction of training data from historical replay buffer
        self.replay_ratio = float(adv_cfg.get("replay_ratio", 0.3))

        # Value head TensorBoard logging
        self._tb_writer = None
        self._vh_global_step = 0

        # Profiling state (wall-clock + CPU RAM + VRAM)
        train_cfg = self.cfg.get("training", {})
        self._profiling_enabled = bool(train_cfg.get("enable_profiling", True))
        profile_dir = Path(train_cfg.get("output_dir", "outputs/sft_advcond")) / "profiles"
        rank = dist.get_rank() if dist.is_initialized() else 0
        self._profile_log_path = profile_dir / f"selfplay_profile_rank{rank}.jsonl"
        self._profiler = StageProfiler(
            enabled=self._profiling_enabled,
            log_path=self._profile_log_path,
            logger=logger,
            capture_memory=True,
            emit_human_logs=False,
        )
        if self._profiling_enabled:
            logger.info("Profiling enabled: %s", self._profile_log_path)
        else:
            logger.info("Profiling disabled (training.enable_profiling=false)")
        self._active_iteration: int | None = None

    def _get_tb_writer(self):
        """Lazily create a TensorBoard SummaryWriter for value head metrics.

        Only rank 0 creates a writer to avoid duplicate/garbled data when
        running under DDP.
        """
        if self._tb_writer is not None:
            return self._tb_writer
        # Only rank 0 should write TensorBoard logs
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter

            output_dir = self.cfg.get("training", {}).get("output_dir", "outputs/sft_advcond")
            log_dir = str(Path(output_dir) / "logs" / "value_head")
            self._tb_writer = SummaryWriter(log_dir=log_dir)
            logger.info("Value head TensorBoard logging at %s", log_dir)
        except ImportError:
            logger.debug("tensorboard not installed — value head TB logging disabled")
        return self._tb_writer

    def _memory_snapshot(self) -> dict[str, float]:
        """Collect a single memory snapshot for profiling."""
        return self._profiler.memory_snapshot()

    def _append_profile_record(self, record: dict[str, Any]) -> None:
        """Append one structured profile record to a separate JSONL log file."""
        self._profiler.append_event(record)

    def _profile_section(
        self,
        *,
        iteration: int | None,
        phase: str,
        stage: str,
        gpu_active: bool,
        extra: dict[str, Any] | None = None,
    ):
        """Profile one section and emit JSONL + human-readable summary logs."""
        baseline_key = None
        if iteration is not None and iteration >= 0:
            baseline_key = f"iter_{iteration}"

        return self._profiler.section(
            iteration=iteration,
            phase=phase,
            stage=stage,
            gpu_active=gpu_active,
            extra=extra,
            baseline_key=baseline_key,
        )

    def run(self) -> None:
        """Run all iterations of the self-play loop.

        If value head pre-training is configured, runs Stage 0 first to
        bootstrap the value head with sensible predictions before the first
        iteration computes advantages.
        """
        self._append_profile_record(
            {
                "event": "run_start",
                "num_iterations": self.num_iterations,
                "profile_log_path": str(self._profile_log_path),
            }
        )

        vh_cfg = self.cfg.get("value_head", {})
        pretrain_scenes = int(vh_cfg.get("pretrain_scenes", 0))
        if pretrain_scenes > 0 and vh_cfg.get("enabled", False):
            self.pretrain_value_head(num_scenes=pretrain_scenes)

        for i in range(self.num_iterations):
            logger.warning("=" * 60)
            logger.warning("SELF-PLAY ITERATION %d / %d", i + 1, self.num_iterations)
            logger.warning("=" * 60)
            self.run_iteration(i)

        self._append_profile_record({"event": "run_end"})
        if self._tb_writer is not None:
            self._tb_writer.close()

    def pretrain_value_head(
        self,
        num_scenes: int = 50,
        num_epochs: int | None = None,
    ) -> None:
        """Stage 0: Pre-train the value head on rollouts from pi_0.

        Two-phase pipeline to minimize GPU bubbles:

        Phase 1 (rollout): Process scenes in small batches. For each batch,
        generate completions, compute rewards, and extract h_obs via a single
        prompt-only VLM forward per scene (not per completion). Accumulate
        all (h_obs, reward) pairs into CPU memory (~16 MB per 1K completions).

        Phase 2 (train): Train the value head on all accumulated data at once
        for num_epochs. This avoids repeated GPU mode-switching between
        inference and training.

        The RolloutEngine is created once and reused (pi_0 and expert are frozen).
        After pre-training, saves value_head.pt and clip_ids.json so that
        pretrain scenes can be excluded from self-play partitioning.

        Args:
            num_scenes: Number of scenes to generate rollouts for.
            num_epochs: Training epochs (default: value_head.pretrain_epochs
                from config, or 50).
        """
        from alpamayo_r1.training.sft_rollout import RolloutEngine

        logger.warning("=" * 60)
        logger.warning("STAGE 0: VALUE HEAD PRE-TRAINING (%d scenes)", num_scenes)
        logger.warning("=" * 60)

        vh_cfg = self.cfg.get("value_head", {})
        if num_epochs is None:
            num_epochs = vh_cfg.get("pretrain_epochs", None)
            num_epochs = int(num_epochs) if num_epochs is not None else 50

        adv_cfg = self.cfg.get("advantage_conditioning", {})
        rollout_cfg = self.cfg.get("rollout", {})
        G = int(adv_cfg.get("completions_per_scene", 8))
        t0_us = int(self.cfg.get("data", {}).get("t0_us", 5_100_000))
        pretrain_batch_scenes = int(vh_cfg.get("pretrain_batch_scenes", 4))

        # Sample scenes for pre-training from all available clip_ids
        rng = random.Random(self.cfg.get("seed", 42))
        shuffled = list(self.all_clip_ids)
        rng.shuffle(shuffled)
        pretrain_scenes = shuffled[:num_scenes]

        # Shrink the data cache for pre-training: scenes are processed
        # sequentially and never revisited, so we only need a small cache.
        # The default max_size=200 wastes ~10-20 GB of CPU RAM holding
        # processed image tensors that will never be reused.
        data_cache = self._get_data_cache()
        original_cache_max = data_cache._max_size
        data_cache._max_size = max(pretrain_batch_scenes * 2, 8)
        logger.info(
            "Pre-training: reduced data cache from %d to %d entries",
            original_cache_max,
            data_cache._max_size,
        )

        # Create RolloutEngine once — pi_0 and expert are frozen throughout
        engine = RolloutEngine(
            full_model=self.full_model,
            processor=self.processor,
            data_cache=data_cache,
            rollout_cfg=rollout_cfg,
            adv_token_ids=self.adv_token_ids,
        )

        # Create value head and optimizer once before the loop
        vh_log_interval = int(vh_cfg.get("log_interval", 10))
        vh_batch_size = int(vh_cfg.get("batch_size", 64))
        value_head = self._get_or_create_value_head()
        reward_weights = self._get_reward_weights()

        # Wrap value head in DDP for synchronized multi-GPU training
        vh_for_training = value_head
        if dist.is_initialized() and dist.get_world_size() > 1:
            vh_device = torch.device("cuda", dist.get_rank())
            value_head.to(vh_device)
            vh_for_training = torch.nn.parallel.DistributedDataParallel(
                value_head, device_ids=[vh_device]
            )

        num_iters = math.ceil(len(pretrain_scenes) / pretrain_batch_scenes)
        logger.info(
            "Iterative pre-training: %d scenes, batch=%d, %d iterations, %d epochs/batch, G=%d",
            len(pretrain_scenes),
            pretrain_batch_scenes,
            num_iters,
            num_epochs,
            G,
        )

        from tqdm import tqdm

        pretrain_rank = dist.get_rank() if dist.is_initialized() else 0
        last_loss = float("nan")
        output_dir = Path(self.cfg.get("training", {}).get("output_dir", "outputs/sft_advcond"))
        pretrain_dir = output_dir / "pretrained"
        checkpoint_interval = int(vh_cfg.get("pretrain_checkpoint_interval", 100))
        start_iter = 0

        # ---------------------------------------------------------------
        # Phase 1: Accumulate rollout data across all batches
        # ---------------------------------------------------------------
        # Generate rollouts in batches (GPU-bound), but accumulate h_obs and
        # rewards into a single pool. This avoids switching between rollout
        # and training modes repeatedly, and the VH trains on all data at once.
        all_hidden_stash: list[dict] = []
        all_reward_stash: list[dict] = []

        with self._profile_section(
            iteration=-1,
            phase="PRETRAIN",
            stage="rollout_accumulate",
            gpu_active=True,
            extra={"num_iters": num_iters, "batch_scenes": pretrain_batch_scenes},
        ):
            pbar = tqdm(
                total=num_iters,
                initial=start_iter,
                desc="VH pretrain (rollout)",
                unit="iter",
                disable=pretrain_rank != 0,
            )
            for i in range(start_iter, num_iters):
                chunk_start = i * pretrain_batch_scenes
                chunk_end = chunk_start + pretrain_batch_scenes
                chunk_scenes = pretrain_scenes[chunk_start:chunk_end]
                if not chunk_scenes:
                    break

                # Prefetch next iteration's scenes so data loads overlap with GPU work
                next_start = chunk_end
                next_end = next_start + pretrain_batch_scenes
                data_cache.prefetch(pretrain_scenes[next_start:next_end], t0_us)

                # 1. Generate rollouts
                rollout_results = engine.generate_completions(chunk_scenes, t0_us, G)
                if not rollout_results:
                    logger.warning("No rollouts from iter %d — skipping", i + 1)
                    pbar.update(1)
                    continue

                # 2. Compute rewards (CPU)
                reward_stash = engine.compute_rewards(rollout_results)

                # 3. Extract h_obs via prompt-only forward (1 per scene, not per completion)
                segment_hidden_stash, _ = engine.extract_prompt_hidden(rollout_results)

                all_hidden_stash.extend(segment_hidden_stash)
                all_reward_stash.extend(reward_stash)

                logger.debug(
                    "Pretrain rollout %d/%d: %d completions (total accumulated: %d)",
                    i + 1,
                    num_iters,
                    len(rollout_results),
                    len(all_hidden_stash),
                )
                pbar.update(1)
                pbar.set_postfix(completions=len(all_hidden_stash))

                del rollout_results, reward_stash, segment_hidden_stash

                # Keep ranks synchronized — without this, faster ranks drift ahead
                # and eventually trigger NCCL collective timeouts.
                if dist.is_initialized():
                    dist.barrier(
                        device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None
                    )

                # Periodic GC to free rollout intermediates
                if (i + 1) % checkpoint_interval == 0:
                    gc.collect()

            pbar.close()

        # ---------------------------------------------------------------
        # Phase 2: Train value head on all accumulated data at once
        # ---------------------------------------------------------------
        # Barrier so all ranks enter DDP training together
        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None)

        if all_hidden_stash:
            g_obs = compute_value_targets(
                segment_reward_stash=all_reward_stash,
                reward_weights=reward_weights,
            )
            logger.info(
                "Training value head on %d completions for %d epochs",
                len(all_hidden_stash), num_epochs,
            )
            with self._profile_section(
                iteration=-1,
                phase="PRETRAIN",
                stage="train_value_head",
                gpu_active=True,
                extra={"num_epochs": num_epochs, "batch_size": vh_batch_size},
            ):
                metrics = train_segment_value_head(
                    value_head=vh_for_training,
                    optimizer=self._value_head_optimizer,
                    segment_hidden_stash=all_hidden_stash,
                    g_obs_list=g_obs,
                    num_epochs=num_epochs,
                    batch_size=vh_batch_size,
                    log_interval=vh_log_interval,
                    tb_writer=self._get_tb_writer(),
                    global_step_offset=self._vh_global_step,
                )
            self._vh_global_step += metrics.get("total_steps", 0)
            last_loss = metrics.get("loss", float("nan"))
            del all_hidden_stash, all_reward_stash, g_obs

        # Restore original cache size for subsequent self-play iterations
        data_cache._max_size = original_cache_max

        # Save value head and pretrain clip IDs (rank 0 only to avoid race)
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            pretrain_dir.mkdir(parents=True, exist_ok=True)
            torch.save(value_head.state_dict(), pretrain_dir / "value_head.pt")
            with open(pretrain_dir / "clip_ids.json", "w") as f:
                json.dump(pretrain_scenes, f)
            logger.info(
                "Saved pretrained value head and %d clip IDs to %s (final loss=%.4f)",
                len(pretrain_scenes),
                pretrain_dir,
                last_loss,
            )

        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None)

        # Exclude pretrain scenes from self-play partitioning
        self._pretrain_clip_ids = pretrain_scenes
        partitioner_clip_ids = [
            cid for cid in self.all_clip_ids if cid not in set(self._pretrain_clip_ids)
        ]
        self.partitioner = ScenePartitioner(
            partitioner_clip_ids,
            self.num_iterations,
            seed=self.cfg.get("seed", 42),
        )

    def run_iteration(self, iteration: int) -> None:
        """Run a single iteration of the self-play loop.

        Phase 1: ROLLOUT — Generate completions from current policy
        Phase 2: EVALUATE — Score, compute advantages, binarize
        Phase 3: TRAIN — SFT with advantage conditioning + expert CFM
        Phase 4: BOOKKEEPING — Update replay buffer, save checkpoints
        """
        fresh_scenes = self.partitioner.get_fresh_scenes(iteration)
        logger.info(
            "Iteration %d: %d fresh scenes, %d historical in replay buffer",
            iteration,
            len(fresh_scenes),
            len(self.replay_buffer),
        )

        iter_start = time.time()
        self._active_iteration = iteration
        self._profiler.set_baseline(f"iter_{iteration}")

        # ----- Phase 1: ROLLOUT -----
        logger.warning("Phase 1: ROLLOUT — generating completions")
        with self._profile_section(
            iteration=iteration,
            phase="ROLLOUT",
            stage="phase_total",
            gpu_active=True,
            extra={"num_fresh_scenes": len(fresh_scenes)},
        ):
            rollout_results = self._rollout_phase(fresh_scenes, iteration)
        self._get_data_cache().log_stats("Phase 1")

        # Save rollout clip IDs for debugging
        train_cfg = self.cfg.get("training", {})
        iter_dir = Path(train_cfg.get("output_dir", "outputs/sft_advcond")) / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        rollout_clip_ids = [r["clip_id"] for r in rollout_results]
        with open(iter_dir / "rollout_clip_ids.json", "w") as f:
            json.dump(rollout_clip_ids, f, indent=2)
        logger.debug("Saved %d rollout clip IDs to %s", len(rollout_clip_ids), iter_dir)

        # ----- Phase 2: EVALUATE -----
        logger.warning("Phase 2: EVALUATE — scoring and binarizing advantages")
        with self._profile_section(
            iteration=iteration,
            phase="EVALUATE",
            stage="phase_total",
            gpu_active=True,
            extra={"num_rollouts": len(rollout_results)},
        ):
            adv_labels, advantages = self._evaluate_phase(rollout_results, iteration)
        self._get_data_cache().log_stats("Phase 2")

        # ----- Phase 2.5: GT AUGMENTATION (optional) -----
        adv_cfg = self.cfg.get("advantage_conditioning", {})
        if adv_cfg.get("augment_with_gt", False):
            logger.info("Phase 2.5: GT AUGMENTATION — selecting bottom-k%% + GT positives")
            with self._profile_section(
                iteration=iteration,
                phase="GT_AUGMENT",
                stage="phase_total",
                gpu_active=False,
            ):
                rollout_results, adv_labels = self._augment_negative_traj_with_gt(
                    rollout_results, adv_labels, advantages
                )

        # ----- Cleanup before Phase 3 -----
        with self._profile_section(
            iteration=iteration,
            phase="CLEANUP",
            stage="pre_train_cleanup",
            gpu_active=False,
        ):
            del advantages
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self._profiling_enabled:
            snap = self._memory_snapshot()
            logger.warning(
                "PROFILE-CLEANUP it=%d RSS=%.2fGB VRAM alloc/reserved/used=%.2f/%.2f/%.2fGB",
                iteration,
                snap["rss_gb"],
                snap["cuda_alloc_gb"],
                snap["cuda_reserved_gb"],
                snap["cuda_used_gb"],
            )

        # ----- Phase 3: TRAIN -----
        logger.warning("Phase 3: TRAIN — advantage-conditioned SFT + expert CFM")
        with self._profile_section(
            iteration=iteration,
            phase="TRAIN",
            stage="phase_total",
            gpu_active=True,
            extra={"num_train_samples": len(rollout_results)},
        ):
            self._train_phase(rollout_results, adv_labels, iteration)

        # ----- Phase 4: BOOKKEEPING -----
        logger.info("Phase 4: BOOKKEEPING — updating replay buffer and checkpoints")
        with self._profile_section(
            iteration=iteration,
            phase="BOOKKEEPING",
            stage="phase_total",
            gpu_active=False,
        ):
            self.replay_buffer.add(rollout_results, adv_labels, iteration)
            advcond_cfg = self.cfg.get("advantage_conditioning", {})
            if advcond_cfg.get("skip_checkpoint", False):
                logger.info("Skipping checkpoint save (skip_checkpoint=true)")
            else:
                self._save_checkpoint(iteration)

        iter_total_s = time.time() - iter_start
        iter_end_mem = self._memory_snapshot()
        iter_growth = self._profiler.diff_from_baseline(f"iter_{iteration}", iter_end_mem)
        self._append_profile_record(
            {
                "event": "iteration_complete",
                "iteration": iteration,
                "duration_s": round(iter_total_s, 4),
                "mem_growth_vs_iter_start": {
                    "rss_gb": round(iter_growth.get("rss_gb", 0.0), 4),
                    "cuda_reserved_gb": round(iter_growth.get("cuda_reserved_gb", 0.0), 4),
                },
                "end_mem": iter_end_mem,
            }
        )
        if self._profiling_enabled:
            logger.warning(
                "PROFILE-ITER it=%d total=%.2fs | RSS growth=%.2fGB | VRAM reserved growth=%.2fGB",
                iteration,
                iter_total_s,
                iter_growth.get("rss_gb", 0.0),
                iter_growth.get("cuda_reserved_gb", 0.0),
            )
        self._active_iteration = None

    def _rollout_phase(self, fresh_scenes: list[str], iteration: int) -> list[dict]:
        """Generate G completions per fresh scene from current policy.

        Uses the RolloutEngine to generate completions. On the first iteration,
        uses the base model (pi_0). On subsequent iterations, loads pi_n.

        Returns:
            List of dicts per completion with:
            {prompt_ids, completion_ids, pred_xyz, gt_xyz, coc_text,
             clip_id, t0_us, completion_prefix}
        """
        from alpamayo_r1.training.sft_rollout import RolloutEngine

        adv_cfg = self.cfg.get("advantage_conditioning", {})
        rollout_cfg = self.cfg.get("rollout", {})
        G = int(adv_cfg.get("completions_per_scene", 8))
        t0_us = int(self.cfg.get("data", {}).get("t0_us", 5_100_000))

        # Build the rollout engine with the current model
        data_cache = self._get_data_cache()
        engine = RolloutEngine(
            full_model=self.full_model,
            processor=self.processor,
            data_cache=data_cache,
            rollout_cfg=rollout_cfg,
            adv_token_ids=self.adv_token_ids,
        )

        # Generate completions (each rank processes its shard)
        local_results = engine.generate_completions(fresh_scenes, t0_us, G)

        # Gather results across all ranks so every rank has the full set
        if dist.is_initialized() and dist.get_world_size() > 1:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_results)
            results = [r for rank_results in gathered for r in rank_results]
            del gathered, local_results
            logger.info(
                "Rollout phase complete: %d completions from %d scenes (gathered from %d ranks)",
                len(results),
                len(fresh_scenes),
                dist.get_world_size(),
            )
        else:
            results = local_results
            logger.info(
                "Rollout phase complete: %d completions from %d scenes",
                len(results),
                len(fresh_scenes),
            )
        return results

    def _evaluate_phase(
        self, rollout_results: list[dict], iteration: int | None = None
    ) -> tuple[list[dict], list[dict]]:
        """Score completions and binarize advantages.

        1. Compute rewards using reward functions
        2. Extract segment hidden states via teacher-forced VLM forward
        3. Compute per-segment advantages using CURRENT value head (before update)
        4. Update value head on current rollout data (for next iteration)
        5. Update the advantage buffer
        6. Binarize advantages into conditioning labels

        Returns:
            (adv_labels, advantages): binarized labels and raw advantage dicts.
        """
        from alpamayo_r1.training.sft_rollout import RolloutEngine

        rollout_cfg = self.cfg.get("rollout", {})
        data_cache = self._get_data_cache()
        engine = RolloutEngine(
            full_model=self.full_model,
            processor=self.processor,
            data_cache=data_cache,
            rollout_cfg=rollout_cfg,
            adv_token_ids=self.adv_token_ids,
        )

        # 1. Compute rewards
        with self._profile_section(
            iteration=iteration,
            phase="EVALUATE",
            stage="compute_rewards",
            gpu_active=False,
            extra={"num_completions": len(rollout_results)},
        ):
            reward_stash = engine.compute_rewards(rollout_results)
        if reward_stash:
            r_traj_vals = [r["r_traj"] for r in reward_stash]
            r_reason_vals = [r["r_reason"] for r in reward_stash]
            r_consist_vals = [r["r_consist"] for r in reward_stash]
            logger.info(
                "  Rewards: traj=[%.3f, %.3f, %.3f] reason=[%.3f, %.3f, %.3f] consist=[%.3f, %.3f, %.3f]",
                min(r_traj_vals),
                np.mean(r_traj_vals),
                max(r_traj_vals),
                min(r_reason_vals),
                np.mean(r_reason_vals),
                max(r_reason_vals),
                min(r_consist_vals),
                np.mean(r_consist_vals),
                max(r_consist_vals),
            )

        # 2. Extract segment hidden states
        vh_cfg = self.cfg.get("value_head", {})
        if vh_cfg.get("enabled", False) and vh_cfg.get("segment_level", False):
            extract_mode = vh_cfg.get("extract_mode", "segment")
            if extract_mode == "prompt":
                with self._profile_section(
                    iteration=iteration,
                    phase="EVALUATE",
                    stage="extract_prompt_hidden",
                    gpu_active=True,
                    extra={"num_completions": len(rollout_results)},
                ):
                    segment_hidden_stash, completion_segment_map = engine.extract_prompt_hidden(
                        rollout_results
                    )
            else:
                with self._profile_section(
                    iteration=iteration,
                    phase="EVALUATE",
                    stage="extract_segment_hidden",
                    gpu_active=True,
                    extra={"num_completions": len(rollout_results)},
                ):
                    segment_hidden_stash, completion_segment_map = engine.extract_segment_hidden(
                        rollout_results
                    )

            reward_weights = self._get_reward_weights()
            value_head = self._get_or_create_value_head()

            # 3. Compute advantages using CURRENT value head (before update)
            with self._profile_section(
                iteration=iteration,
                phase="EVALUATE",
                stage="compute_advantages",
                gpu_active=False,
            ):
                advantages = compute_segment_advantages_from_rollouts(
                    segment_hidden_stash=segment_hidden_stash,
                    segment_reward_stash=reward_stash,
                    completion_segment_map=completion_segment_map,
                    value_head=value_head,
                    reward_weights=reward_weights,
                )

            # 4. THEN update value head on current data (for next iteration)
            g_obs = compute_value_targets(
                segment_reward_stash=reward_stash,
                reward_weights=reward_weights,
            )
            vh_train_epochs = int(vh_cfg.get("train_epochs", 10))
            vh_log_interval = int(vh_cfg.get("log_interval", 10))
            vh_batch_size = int(vh_cfg.get("batch_size", 64))
            # Wrap value head in DDP for synchronized multi-GPU training
            vh_for_train = value_head
            if dist.is_initialized() and dist.get_world_size() > 1:
                vh_device = torch.device("cuda", dist.get_rank())
                value_head.to(vh_device)
                vh_for_train = torch.nn.parallel.DistributedDataParallel(
                    value_head, device_ids=[vh_device]
                )
            with self._profile_section(
                iteration=iteration,
                phase="EVALUATE",
                stage="train_value_head",
                gpu_active=True,
                extra={"epochs": vh_train_epochs, "batch_size": vh_batch_size},
            ):
                vh_metrics = train_segment_value_head(
                    value_head=vh_for_train,
                    optimizer=self._value_head_optimizer,
                    segment_hidden_stash=segment_hidden_stash,
                    g_obs_list=g_obs,
                    num_epochs=vh_train_epochs,
                    batch_size=vh_batch_size,
                    log_interval=vh_log_interval,
                    tb_writer=self._get_tb_writer(),
                    global_step_offset=self._vh_global_step,
                )
            if isinstance(vh_for_train, torch.nn.parallel.DistributedDataParallel):
                del vh_for_train  # release DDP wrapper
            del segment_hidden_stash, completion_segment_map, reward_stash
            del g_obs
            self._vh_global_step += vh_metrics.get("total_steps", 0)
            logger.info(
                "Evaluate Stage train_value_head metrics: epochs=%d loss=%.4f",
                vh_train_epochs,
                vh_metrics.get("loss", float("nan")),
            )
        else:
            # Fallback: use composite reward as a_obs, no segment-level detail
            advantages = []
            reward_weights = self._get_reward_weights()
            w_traj, w_reason, w_consist = reward_weights
            for rew in reward_stash:
                composite = (
                    w_traj * rew["r_traj"]
                    + w_reason * rew["r_reason"]
                    + w_consist * rew["r_consist"]
                )
                advantages.append({"a_obs": composite, "a_traj": composite})
            del reward_stash

        # 5. Update advantage buffer
        a_obs_list = [a["a_obs"] for a in advantages]
        a_traj_list = [a["a_traj"] for a in advantages]
        self.advantage_buffer.update(a_obs_list, a_traj_list)

        # 6. Binarize
        adv_labels = []
        for adv in advantages:
            i_obs, i_traj = self.advantage_buffer.binarize(adv["a_obs"], adv["a_traj"])
            adv_labels.append({"i_obs": i_obs, "i_traj": i_traj})

        n_pos = sum(1 for lab in adv_labels if lab["i_obs"] and lab["i_traj"])
        logger.info(
            "Evaluate phase: %d/%d all-positive, thresholds=%s",
            n_pos,
            len(adv_labels),
            self.advantage_buffer.compute_thresholds(),
        )
        return adv_labels, advantages

    def _augment_negative_traj_with_gt(
        self,
        rollout_results: list[dict],
        adv_labels: list[dict],
        advantages: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Select bottom-k% rollouts and build GT-augmented positive counterparts.

        The returned training set contains ONLY:
        - The bottom-k% rollouts by trajectory advantage (negative examples)
        - GT-augmented copies of those rollouts, filtered by top-k2% consistency
          (positive trajectory examples)

        All other rollouts are used for buffer/binarization only and are discarded
        from the training set.

        Args:
            rollout_results: All rollout completions from this iteration.
            adv_labels: Binarized advantage labels for all rollouts.
            advantages: Raw advantage dicts ({a_obs, a_traj}) for all rollouts.

        Returns:
            (train_rollouts, train_labels): The filtered training set.
        """
        from alpamayo_r1.training.sft_rollout import RolloutEngine

        adv_cfg = self.cfg.get("advantage_conditioning", {})
        bottom_k = float(adv_cfg.get("gt_augment_bottom_k", 30))
        top_k2 = float(adv_cfg.get("gt_augment_top_k2", 50))

        # 1. Select bottom-k% by trajectory advantage
        n_total = len(rollout_results)
        indexed_a_traj = [(i, advantages[i]["a_traj"]) for i in range(n_total)]
        indexed_a_traj.sort(key=lambda x: x[1])
        n_bottom = max(1, int(n_total * bottom_k / 100))
        bottom_indices = [i for i, _ in indexed_a_traj[:n_bottom]]

        logger.info(
            "GT augmentation: selected bottom %d/%d rollouts (k=%.0f%%, a_traj threshold=%.4f)",
            n_bottom,
            n_total,
            bottom_k,
            indexed_a_traj[n_bottom - 1][1] if n_bottom > 0 else 0.0,
        )

        # 2. Build GT-augmented copies for selected rollouts
        traj_future_start_id = self.full_model.special_token_ids["traj_future_start"]
        traj_future_end_id = self.full_model.special_token_ids["traj_future_end"]
        traj_tokenizer = self.full_model.traj_tokenizer
        traj_token_start_idx = self.full_model.config.traj_token_start_idx
        traj_vocab_size = self.full_model.config.traj_vocab_size
        data_cache = self._get_data_cache()

        augmented = []
        augmented_source_indices = []  # tracks which bottom_indices entry produced each augmented
        for i in bottom_indices:
            r = rollout_results[i]

            if "clip_id" not in r or "hist_xyz" not in r:
                continue

            clip_id = r["clip_id"]
            t0_us = r.get("t0_us", 5_100_000)
            _, ego_future_xyz = data_cache.get(clip_id, t0_us, device="cpu")
            ego_future_rot = data_cache.get_ego_future_rot(clip_id, t0_us)

            hist_xyz = r["hist_xyz"].unsqueeze(0)
            hist_rot = r["hist_rot"].unsqueeze(0)
            fut_xyz = ego_future_xyz[:, 0].cpu()
            fut_rot = ego_future_rot[:, 0].cpu()

            discrete_tokens = traj_tokenizer.encode(hist_xyz, hist_rot, fut_xyz, fut_rot)
            discrete_tokens = discrete_tokens.clamp(0, traj_vocab_size - 1)
            gt_traj_token_ids = (discrete_tokens + traj_token_start_idx).squeeze(0).tolist()

            completion_ids = r["completion_ids"]
            try:
                traj_boundary = completion_ids.index(traj_future_start_id)
            except ValueError:
                continue
            coc_part = completion_ids[:traj_boundary]

            new_completion_ids = (
                coc_part + [traj_future_start_id] + gt_traj_token_ids + [traj_future_end_id]
            )

            pred_xyz, _, _ = traj_tokenizer.decode(hist_xyz, hist_rot, discrete_tokens)
            gt_pred_xyz = pred_xyz[0].cpu().numpy().flatten().tolist()

            augmented.append(
                {
                    **r,
                    "completion_ids": new_completion_ids,
                    "pred_xyz": gt_pred_xyz,
                    "completion_prefix": coc_part + [traj_future_start_id],
                    "expert_fut_xyz": fut_xyz[0],
                    "expert_fut_rot": fut_rot[0],
                    "is_gt_augmented": True,
                }
            )
            augmented_source_indices.append(i)

        if not augmented:
            logger.info("GT augmentation: no augmented completions built, returning negatives only")
            neg_rollouts = [rollout_results[i] for i in bottom_indices]
            neg_labels = [adv_labels[i] for i in bottom_indices]
            return neg_rollouts, neg_labels

        # 3. Score augmented completions to get consistency reward
        rollout_cfg = self.cfg.get("rollout", {})
        engine = RolloutEngine(
            full_model=self.full_model,
            processor=self.processor,
            data_cache=self._get_data_cache(),
            rollout_cfg=rollout_cfg,
            adv_token_ids=self.adv_token_ids,
        )
        aug_rewards = engine.compute_rewards(augmented)

        # Log per-sample reward comparison
        for j, (aug_r, src_i) in enumerate(zip(aug_rewards, augmented_source_indices)):
            clip_id = rollout_results[src_i]["clip_id"]
            logger.info(
                "GT aug [%s] sample %d: "
                "r_traj=%.3f r_reason=%.3f r_consist=%.3f",
                clip_id,
                j,
                aug_r["r_traj"],
                aug_r["r_reason"],
                aug_r["r_consist"],
            )

        # 4. Filter augmented: keep top-k2% by r_consist
        consist_scores = [(j, aug_rewards[j]["r_consist"]) for j in range(len(augmented))]
        consist_scores.sort(key=lambda x: x[1], reverse=True)
        n_keep = max(1, int(len(augmented) * top_k2 / 100))
        kept_indices = [j for j, _ in consist_scores[:n_keep]]
        kept_indices.sort()  # preserve original order

        filtered_augmented = [augmented[j] for j in kept_indices]

        # 5. Assign labels for kept augmented samples:
        #    i_traj = True (GT trajectory, positive by construction)
        #    i_obs = binarize from buffer (preserving independent obs semantics)
        filtered_aug_labels = []
        for j in kept_indices:
            src_i = augmented_source_indices[j]
            a_obs = advantages[src_i]["a_obs"]
            i_obs, _ = self.advantage_buffer.binarize(a_obs, 0.0)
            filtered_aug_labels.append({"i_obs": i_obs, "i_traj": True})

        # 6. Assemble training set: negatives + filtered GT positives
        neg_rollouts = [rollout_results[i] for i in bottom_indices]
        neg_labels = [adv_labels[i] for i in bottom_indices]

        train_rollouts = neg_rollouts + filtered_augmented
        train_labels = neg_labels + filtered_aug_labels

        # 7. Log summary
        n_neg = len(neg_rollouts)
        n_pos = len(filtered_augmented)
        logger.warning(
            "GT augmentation summary: "
            "%d total rollouts → %d negatives (bottom %.0f%%) + %d GT positives "
            "(%.0f%% of %d augmented passed r_consist filter, top %.0f%%) "
            "= %d training samples (pos/neg ratio: %.2f)",
            n_total,
            n_neg,
            bottom_k,
            n_pos,
            n_pos / max(len(augmented), 1) * 100,
            len(augmented),
            top_k2,
            n_neg + n_pos,
            n_pos / max(n_neg, 1),
        )

        return train_rollouts, train_labels

    def _train_phase(
        self,
        rollout_results: list[dict],
        adv_labels: list[dict],
        iteration: int,
    ) -> None:
        """Build dataset, train SFT + expert.

        1. Load model — either pi_0 (reset_to_base) or merge previous LoRA and continue
        2. Apply LoRA to VLM
        3. Build AdvCondDataset from fresh + historical rollouts
        4. Create AdvCondSFTTrainer (handles VLM SFT + expert CFM)
        5. Train for configured epochs
        6. Save as pi_{n+1}
        """
        from peft import LoraConfig

        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        from alpamayo_r1.training.rollout_utils import prepare_vlm_for_training
        from alpamayo_r1.training.sft_trainer import AdvCondSFTTrainer

        train_cfg = self.cfg.get("training", {})
        lora_cfg = self.cfg.get("lora", {})
        expert_cfg = self.cfg.get("expert_finetune", {})
        adv_cfg = self.cfg.get("advantage_conditioning", {})
        reset_to_base = bool(adv_cfg.get("reset_to_base", False))

        # 1. Load model for this iteration
        #    First, free the old model's GPU memory (VLM stays on GPU after rollout/evaluate)
        #    Detach advantage embedding hooks BEFORE freeing the old model so
        #    stale hook handles don't keep the old embedding layer (and thus
        #    the entire old VLM) alive.
        if self.adv_embedding is not None:
            self.adv_embedding.detach()

        if self.full_model is not None:
            self.full_model.vlm.cpu()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if self._profiling_enabled and torch.cuda.is_available():
                snap = self._memory_snapshot()
                logger.info(
                    "PROFILE-TRAIN-PREP it=%d cleanup_after_evaluate alloc/reserved/used=%.2f/%.2f/%.2fGB",
                    iteration,
                    snap["cuda_alloc_gb"],
                    snap["cuda_reserved_gb"],
                    snap["cuda_used_gb"],
                )

        if reset_to_base:
            # RECAP-style: reload base model from scratch every iteration
            logger.info("Loading base model from %s (reset-to-checkpoint)", self.base_model_path)
            full_model = AlpamayoR1.from_pretrained(self.base_model_path, dtype=torch.bfloat16)
        else:
            # Reuse existing model; at iteration 0 the VLM is not a PeftModel
            # so merge_and_unload is a no-op (hasattr check is False).
            full_model = self.full_model
            if hasattr(full_model.vlm, "merge_and_unload"):
                logger.info(
                    "Continuing from iteration %d checkpoint (merge previous LoRA)",
                    iteration - 1,
                )
                full_model.vlm = full_model.vlm.merge_and_unload()
                logger.info("Merged previous VLM LoRA weights into base model")
            else:
                logger.info("Reusing existing model for iteration %d", iteration)

            # Clean up stale peft_config left by merge_and_unload() to prevent
            # PEFT from seeing "multiple adapters" on the next get_peft_model()
            if hasattr(full_model.vlm, "peft_config"):
                delattr(full_model.vlm, "peft_config")

            # Merge expert LoRA if present (prevents double-wrapping when
            # _setup_expert applies fresh LoRA in the next training phase)
            if hasattr(full_model.expert, "merge_and_unload"):
                full_model.expert = full_model.expert.merge_and_unload()
                logger.info("Merged previous expert LoRA weights into base model")
            if hasattr(full_model.expert, "peft_config"):
                delattr(full_model.expert, "peft_config")

        prepare_vlm_for_training(full_model)

        # Freeze non-VLM params
        for name, param in full_model.named_parameters():
            if not name.startswith("vlm."):
                param.requires_grad = False

        # 2. Apply LoRA
        lora_config = None
        if lora_cfg.get("enabled", True):
            lora_config = LoraConfig(
                r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("alpha", 32)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                target_modules=list(
                    lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
                ),
                task_type="CAUSAL_LM",
            )
            from peft import get_peft_model

            full_model.vlm = get_peft_model(full_model.vlm, lora_config)
            full_model.vlm.enable_input_require_grads()
            logger.info("Applied LoRA: r=%d, alpha=%d", lora_config.r, lora_config.lora_alpha)

        # Attach trainable advantage embedding to PeftModel
        if self.adv_embedding is not None:
            self.adv_embedding = self.adv_embedding.to(
                dtype=next(full_model.vlm.parameters()).dtype
            )
            full_model.vlm.adv_embedding = self.adv_embedding  # visible to Trainer optimizer
            self.adv_embedding.attach(full_model.vlm)

        # 3. Build training dataset (fresh + historical replay)
        train_dataset = self._build_training_dataset(rollout_results, adv_labels)

        # 4. Create training args
        output_dir = Path(train_cfg.get("output_dir", "outputs/sft_advcond")) / f"iter_{iteration}"
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        fsdp_kwargs = {}
        if world_size == 1:
            fsdp_kwargs["fsdp"] = ""

        dataloader_num_workers = int(train_cfg.get("dataloader_num_workers", 0))
        dataloader_pin_memory = bool(train_cfg.get("dataloader_pin_memory", True))
        dataloader_persistent_workers = bool(
            train_cfg.get("dataloader_persistent_workers", False)
        )

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=int(train_cfg.get("num_train_epochs", 1)),
            per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 4)),
            gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
            learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
            bf16=bool(train_cfg.get("bf16", True)),
            gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=int(train_cfg.get("logging_steps", 1)),
            save_strategy=train_cfg.get("save_strategy", "no"),
            save_steps=int(train_cfg.get("save_steps", 200)),
            save_total_limit=int(train_cfg.get("save_total_limit", 3)),
            warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            report_to=train_cfg.get("report_to", "tensorboard"),
            remove_unused_columns=False,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=dataloader_pin_memory,
            dataloader_persistent_workers=(
                dataloader_persistent_workers and dataloader_num_workers > 0
            ),
            **fsdp_kwargs,
        )

        # 5. Create trainer
        data_cache = self._get_data_cache()
        adv_cfg = self.cfg.get("advantage_conditioning", {})
        value_head = self._value_head if hasattr(self, "_value_head") else None
        trainer = AdvCondSFTTrainer(
            model=full_model.vlm,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.processor,
            full_model=full_model,
            adv_token_ids=self.adv_token_ids,
            alpha=float(adv_cfg.get("alpha", 1.0)),
            expert_cfg=expert_cfg,
            data_cache=data_cache,
            value_head=value_head,
            profiling_cfg={
                "enabled": bool(train_cfg.get("profile_inner_steps", False)),
                "every_n_steps": int(train_cfg.get("profile_inner_every_n_steps", train_cfg.get("logging_steps", 1))),
                "capture_memory": bool(train_cfg.get("profile_inner_capture_memory", True)),
            },
        )

        # 6. Train
        if self._profiling_enabled and torch.cuda.is_available():
            snap = self._memory_snapshot()
            logger.info(
                "PROFILE-TRAIN-PREP it=%d before_trainer_train alloc/reserved/used=%.2f/%.2f/%.2fGB",
                iteration,
                snap["cuda_alloc_gb"],
                snap["cuda_reserved_gb"],
                snap["cuda_used_gb"],
            )
        logger.info(
            "Starting SFT iteration %d: %d samples (dl_workers=%d pin_memory=%s persistent_workers=%s)",
            iteration,
            len(train_dataset),
            dataloader_num_workers,
            dataloader_pin_memory,
            dataloader_persistent_workers and dataloader_num_workers > 0,
        )
        with self._profile_section(
            iteration=iteration,
            phase="TRAIN",
            stage="trainer_train",
            gpu_active=True,
            extra={"num_train_samples": len(train_dataset)},
        ):
            trainer.train()

        # 7. Save
        advcond_cfg = self.cfg.get("advantage_conditioning", {})
        skip_checkpoint = advcond_cfg.get("skip_checkpoint", False)

        if skip_checkpoint:
            logger.info("Skipping Phase 3 checkpoint save (skip_checkpoint=true)")
        else:
            # Save the adapter first (standard PEFT behavior)
            final_save_dir = output_dir / "final"
            trainer.save_model(str(final_save_dir))

            # ALSO: Save the ENTIRE model to confirm if loading/resizing is the issue.
            logger.info(
                "Merging LoRA weights and saving full AlpamayoR1 model to %s",
                final_save_dir / "full_model",
            )

            # 1. Merge LoRA weights in-place temporarily for saving
            if hasattr(self.full_model.vlm, "merge_and_unload"):
                self.full_model.vlm = self.full_model.vlm.merge_and_unload()
            if hasattr(self.full_model.expert, "merge_and_unload"):
                self.full_model.expert = self.full_model.expert.merge_and_unload()

            # 2. Save the full AlpamayoR1 instance
            # This saves config.json (model_type: alpamayo_r1) and all submodule weights
            self.full_model.save_pretrained(str(final_save_dir / "full_model"))
            self.processor.save_pretrained(str(final_save_dir / "full_model"))

            # 3. Reload LoRA for the next iteration (since SelfPlayLoop continues using this model)
            # Note: In RECAP mode (reset_to_base=True), this doesn't matter as it reloads anyway.
            # If not resetting, we would need to re-apply the adapter here.
            self.current_policy_path = str(output_dir / "final")
            logger.info("Saved pi_%d to %s", iteration + 1, self.current_policy_path)

        # 8. Synchronize NCCL state across ranks before the next iteration.
        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None)
            logger.debug("Post-training barrier complete")

        # Explicitly delete the Trainer (and its DDP wrapper) so the old
        # NCCL state is fully released before the next iteration creates
        # a new DDP.
        del trainer

        # Merge LoRA back into the base model so the next rollout uses a
        # raw HF model, not a PeftModel.  PeftModel.generate() creates
        # many small intermediate tensors (per LoRA adapter, per layer,
        # per token) that fragment PyTorch's caching allocator.  With a
        # raw model the allocation pattern matches iteration-0's rollout
        # and generation succeeds reliably.
        # The checkpoint was already saved above, so LoRA weights are safe.
        if hasattr(full_model.vlm, "merge_and_unload"):
            full_model.vlm = full_model.vlm.merge_and_unload()
            logger.info("Merged VLM LoRA into base model for clean rollout")
        if hasattr(full_model.vlm, "peft_config"):
            delattr(full_model.vlm, "peft_config")
        if hasattr(full_model.expert, "merge_and_unload"):
            full_model.expert = full_model.expert.merge_and_unload()
            logger.info("Merged expert LoRA into base model")
        if hasattr(full_model.expert, "peft_config"):
            delattr(full_model.expert, "peft_config")

        # Move ALL model components to CPU so gc.collect + empty_cache
        # can fully reclaim GPU memory.
        full_model.vlm.cpu()
        full_model.expert.cpu()
        full_model.action_in_proj.cpu()
        full_model.action_out_proj.cpu()
        full_model.action_space.cpu()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self._profiling_enabled:
                snap = self._memory_snapshot()
                logger.info(
                    "PROFILE-TRAIN-CLEANUP it=%d after_training alloc/reserved/used=%.2f/%.2f/%.2fGB",
                    iteration,
                    snap["cuda_alloc_gb"],
                    snap["cuda_reserved_gb"],
                    snap["cuda_used_gb"],
                )

        # Update the full_model reference for next iteration's rollouts
        self.full_model = full_model

    def _build_training_dataset(
        self,
        fresh_rollouts: list[dict],
        fresh_labels: list[dict],
    ) -> AdvCondDataset:
        """Build the advantage-conditioned dataset for SFT training.

        Combines fresh rollouts (current iteration) with historical rollouts
        sampled from the replay buffer at the configured replay_ratio.

        The fresh rollouts are the primary training data. Historical data
        is mixed in to improve stability and prevent catastrophic forgetting.
        """
        # Calculate how many historical samples to add
        n_fresh = len(fresh_rollouts)
        n_historical = int(n_fresh * self.replay_ratio / (1 - self.replay_ratio))

        # Sample from replay buffer
        hist_rollouts, hist_labels = self.replay_buffer.sample(n_historical)

        # Combine fresh + historical
        all_rollouts = fresh_rollouts + hist_rollouts
        all_labels = fresh_labels + hist_labels

        logger.info(
            "Training dataset: %d fresh + %d historical = %d total",
            n_fresh,
            len(hist_rollouts),
            len(all_rollouts),
        )

        adv_cfg = self.cfg.get("advantage_conditioning", {})
        # Get traj_future_start token ID for causal placement of adv_traj
        traj_future_start_id = None
        if hasattr(self.full_model, "special_token_ids"):
            traj_future_start_id = self.full_model.special_token_ids.get("traj_future_start")

        # Pre-compute conditioned sequences to avoid per-sample boundary
        # scanning and list concatenation in the DataLoader hot path.
        precomputed = precompute_conditioned_sequences(
            rollout_results=all_rollouts,
            adv_labels=all_labels,
            adv_token_ids=self.adv_token_ids,
            traj_future_start_id=traj_future_start_id,
        )

        return AdvCondDataset(
            rollout_results=all_rollouts,
            precomputed=precomputed,
            p_drop=float(adv_cfg.get("p_drop", 0.3)),
            data_cache=self._get_data_cache(),
        )

    def _get_data_cache(self):
        """Get or create a ClipDataCache for the current run."""
        if not hasattr(self, "_data_cache"):
            from alpamayo_r1.training.rollout_utils import ClipDataCache

            rollout_cfg = self.cfg.get("rollout", {})
            self._data_cache = ClipDataCache(
                avdi=self.avdi,
                processor=self.processor,
                cache_pil_images=False,
                max_size=int(rollout_cfg.get("data_cache_max_size", 200)),
            )
        return self._data_cache

    def _get_reward_weights(self) -> tuple[float, float, float]:
        """Get reward function weights from config."""
        reward_cfg = self.cfg.get("rewards", {})
        return (
            float(reward_cfg.get("trajectory_weight", 0.5)),
            float(reward_cfg.get("reasoning_weight", 0.25)),
            float(reward_cfg.get("consistency_weight", 0.25)),
        )

    def _get_or_create_value_head(self) -> torch.nn.Module:
        """Get or create the segment-level value head."""
        if not hasattr(self, "_value_head"):
            from alpamayo_r1.training.value_head import SegmentValueHead

            vh_cfg = self.cfg.get("value_head", {})
            hidden_dim = int(vh_cfg.get("hidden_dim", 4096))
            self._value_head = SegmentValueHead(hidden_dim=hidden_dim)

            # Optionally load pretrained weights
            load_path = vh_cfg.get("load_path")
            if load_path:
                state = torch.load(load_path, map_location="cpu", weights_only=False)
                self._value_head.load_state_dict(state)
                logger.info("Loaded value head from %s", load_path)

            self._value_head_optimizer = torch.optim.Adam(
                self._value_head.parameters(), lr=float(vh_cfg.get("lr", 1e-5))
            )
        return self._value_head

    def _save_checkpoint(self, iteration: int) -> None:
        """Save advantage buffer state for the iteration.

        VLM LoRA, expert, and value head are already saved by
        trainer.save_model() → AdvCondSFTTrainer._save() into
        iter_{n}/final/ at the end of Phase 3.
        """
        output_dir = Path(self.cfg.get("training", {}).get("output_dir", "outputs/sft_advcond"))
        iter_dir = output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Save advantage buffer state (loop-level, not managed by trainer)
        adv_buf_path = iter_dir / "advantage_buffer.pt"
        torch.save(self.advantage_buffer.state_dict(), adv_buf_path)
        logger.info("Saved advantage buffer to %s", adv_buf_path)
