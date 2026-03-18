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

import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from physical_ai_av import PhysicalAIAVDatasetInterface
from transformers import TrainingArguments

from alpamayo_r1.training.advantage_conditioning import (
    AdvantageBuffer,
    AdvCondDataset,
    compute_segment_advantages_from_rollouts,
    compute_value_targets,
    register_advantage_tokens,
    train_segment_value_head,
)

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

        # Update labels in buffer
        for entry, adv in zip(self._entries, new_advantages):
            i_obs, i_traj = advantage_buffer.binarize(adv["a_obs"], adv["a_traj"])
            entry["adv_label"] = {"i_obs": i_obs, "i_traj": i_traj}

        logger.info("Recomputed advantage labels for %d buffer entries", len(self._entries))

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
    expert_lora_cfg: dict | None = None,
) -> Any:
    """Load expert + projection weights from a checkpoint.

    Handles both full-state and LoRA-only checkpoints. When the checkpoint
    contains only LoRA weights (``expert_lora: True``), the caller must
    supply ``expert_lora_cfg`` so that LoRA layers can be created before
    the weights are loaded.

    Args:
        full_model: AlpamayoR1 instance (mutated in place).
        path: Path to ``expert_checkpoint.pt``.
        expert_lora_cfg: LoRA config dict (``r``, ``alpha``, ``target_modules``, …).
            Required when loading a LoRA-only checkpoint.

    Returns:
        The same ``full_model`` reference (mutated).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    is_lora = checkpoint.get("expert_lora", False)

    if is_lora:
        if expert_lora_cfg is None:
            raise ValueError(
                "Checkpoint at %s is LoRA-only but no expert_lora_cfg was provided" % path
            )
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=int(expert_lora_cfg.get("r", 4)),
            lora_alpha=int(expert_lora_cfg.get("alpha", 32)),
            lora_dropout=float(expert_lora_cfg.get("dropout", 0.0)),
            target_modules=list(
                expert_lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
            ),
            task_type="FEATURE_EXTRACTION",
        )
        full_model.expert = get_peft_model(full_model.expert, lora_config)
        full_model.expert.load_state_dict(checkpoint["expert"], strict=False)
        logger.info("Loaded expert LoRA weights from %s", path)
    else:
        full_model.expert.load_state_dict(checkpoint["expert"], strict=False)
        logger.info("Loaded full expert weights from %s", path)

    full_model.action_in_proj.load_state_dict(checkpoint["action_in_proj"])
    full_model.action_out_proj.load_state_dict(checkpoint["action_out_proj"])
    logger.info("Loaded projection layers from %s", path)
    return full_model


def load_vlm_from_iterations(
    base_model_name: str,
    output_dir: str | Path,
    up_to_iteration: int,
    expert_lora_cfg: dict | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Reconstruct a model by iteratively merging per-iteration LoRA adapters.

    Starting from the base model, loads and merges VLM LoRA adapters from
    ``iter_0/final/`` through ``iter_{up_to_iteration}/final/``. Also loads
    the latest expert checkpoint and value head from the final iteration.

    Args:
        base_model_name: HuggingFace model name or path for the base AlpamayoR1.
        output_dir: Root output directory containing ``iter_*/final/`` subdirs.
        up_to_iteration: Last iteration index (inclusive) to merge.
        expert_lora_cfg: Expert LoRA config dict (needed if expert checkpoint is LoRA-only).
        dtype: Model dtype.

    Returns:
        Dict with keys ``full_model``, ``value_head`` (or None).
    """
    from peft import PeftModel

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    output_dir = Path(output_dir)

    logger.info("Loading base model from %s", base_model_name)
    full_model = AlpamayoR1.from_pretrained(base_model_name, dtype=dtype)

    # Iteratively apply and merge VLM LoRA adapters
    for i in range(up_to_iteration + 1):
        adapter_path = output_dir / f"iter_{i}" / "final"
        adapter_config = adapter_path / "adapter_config.json"
        if not adapter_config.exists():
            logger.info("No VLM adapter at %s — skipping iteration %d", adapter_path, i)
            continue
        logger.info("Loading VLM LoRA adapter from %s", adapter_path)
        full_model.vlm = PeftModel.from_pretrained(full_model.vlm, str(adapter_path))
        full_model.vlm = full_model.vlm.merge_and_unload()
        logger.info("Merged VLM LoRA from iteration %d", i)

    # Load expert checkpoint from the latest iteration
    latest_expert = output_dir / f"iter_{up_to_iteration}" / "final" / "expert_checkpoint.pt"
    if latest_expert.exists():
        load_expert_checkpoint(full_model, latest_expert, expert_lora_cfg=expert_lora_cfg)

    # Load value head from the latest iteration
    value_head = None
    latest_vh = output_dir / f"iter_{up_to_iteration}" / "final" / "value_head.pt"
    if latest_vh.exists():
        value_head = load_value_head_checkpoint(latest_vh)

    logger.info(
        "Reconstructed model through iteration %d (VLM merged, expert loaded, vh=%s)",
        up_to_iteration,
        value_head is not None,
    )
    return {"full_model": full_model, "value_head": value_head}


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

        adv_cfg = cfg.get("advantage_conditioning", {})
        num_iterations = int(adv_cfg.get("num_iterations", 5))
        self.num_iterations = num_iterations

        self.partitioner = ScenePartitioner(all_clip_ids, num_iterations, seed=cfg.get("seed", 42))
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

        # Register advantage tokens once
        self.adv_token_ids = register_advantage_tokens(processor.tokenizer)

        # Replay ratio: fraction of training data from historical replay buffer
        self.replay_ratio = float(adv_cfg.get("replay_ratio", 0.3))

        # Value head TensorBoard logging
        self._tb_writer = None
        self._vh_global_step = 0

    def _get_tb_writer(self):
        """Lazily create a TensorBoard SummaryWriter for value head metrics."""
        if self._tb_writer is not None:
            return self._tb_writer
        try:
            from torch.utils.tensorboard import SummaryWriter

            output_dir = self.cfg.get("training", {}).get("output_dir", "outputs/sft_advcond")
            log_dir = str(Path(output_dir) / "logs" / "value_head")
            self._tb_writer = SummaryWriter(log_dir=log_dir)
            logger.info("Value head TensorBoard logging at %s", log_dir)
        except ImportError:
            logger.debug("tensorboard not installed — value head TB logging disabled")
        return self._tb_writer

    def run(self) -> None:
        """Run all iterations of the self-play loop.

        If value head pre-training is configured, runs Stage 0 first to
        bootstrap the value head with sensible predictions before the first
        iteration computes advantages.
        """
        vh_cfg = self.cfg.get("value_head", {})
        pretrain_scenes = int(vh_cfg.get("pretrain_scenes", 0))
        if pretrain_scenes > 0 and vh_cfg.get("enabled", False):
            self.pretrain_value_head(num_scenes=pretrain_scenes)

        for i in range(self.num_iterations):
            logger.info("=" * 60)
            logger.info("SELF-PLAY ITERATION %d / %d", i + 1, self.num_iterations)
            logger.info("=" * 60)
            self.run_iteration(i)

    def pretrain_value_head(
        self,
        num_scenes: int = 50,
        num_epochs: int | None = None,
    ) -> None:
        """Stage 0: Pre-train the value head on rollouts from pi_0.

        Generates rollouts from the base policy, extracts segment hidden
        states, and trains the value head for many epochs so that the first
        SFT iteration starts with a sensible baseline instead of random
        predictions.

        This can be called standalone (before run()) or is called automatically
        when value_head.pretrain_scenes > 0 in the config.

        Args:
            num_scenes: Number of scenes to generate rollouts for.
            num_epochs: Training epochs (default: value_head.pretrain_epochs
                from config, or 50).
        """
        from alpamayo_r1.training.sft_rollout import RolloutEngine

        logger.info("=" * 60)
        logger.info("STAGE 0: VALUE HEAD PRE-TRAINING (%d scenes)", num_scenes)
        logger.info("=" * 60)

        vh_cfg = self.cfg.get("value_head", {})
        if num_epochs is None:
            num_epochs = vh_cfg.get("pretrain_epochs", None)
            num_epochs = int(num_epochs) if num_epochs is not None else 50

        adv_cfg = self.cfg.get("advantage_conditioning", {})
        rollout_cfg = self.cfg.get("rollout", {})
        G = int(adv_cfg.get("completions_per_scene", 8))
        t0_us = int(self.cfg.get("data", {}).get("t0_us", 5_100_000))

        # Sample scenes for pre-training from all available clip_ids
        rng = random.Random(self.cfg.get("seed", 42))
        shuffled = list(self.all_clip_ids)
        rng.shuffle(shuffled)
        pretrain_scenes = shuffled[:num_scenes]

        # Generate rollouts from current model (pi_0)
        data_cache = self._get_data_cache()
        engine = RolloutEngine(
            full_model=self.full_model,
            processor=self.processor,
            data_cache=data_cache,
            rollout_cfg=rollout_cfg,
            adv_token_ids=self.adv_token_ids,
        )

        logger.info("Generating rollouts from %d scenes (G=%d)...", len(pretrain_scenes), G)
        rollout_results = engine.generate_completions(pretrain_scenes, t0_us, G)
        logger.info("Generated %d completions", len(rollout_results))

        if not rollout_results:
            logger.warning("No rollouts generated — skipping value head pre-training")
            return

        # Compute rewards
        reward_stash = engine.compute_rewards(rollout_results)

        # Extract segment hidden states
        segment_hidden_stash, completion_segment_map = engine.extract_segment_hidden(
            rollout_results
        )

        # Compute value targets
        reward_weights = self._get_reward_weights()
        g_obs, g_coc, g_traj = compute_value_targets(
            segment_reward_stash=reward_stash,
            completion_segment_map=completion_segment_map,
            reward_weights=reward_weights,
        )

        # Train value head
        vh_log_interval = int(vh_cfg.get("log_interval", 10))
        value_head = self._get_or_create_value_head()
        metrics = train_segment_value_head(
            value_head=value_head,
            optimizer=self._value_head_optimizer,
            segment_hidden_stash=segment_hidden_stash,
            g_obs_list=g_obs,
            g_coc_list=g_coc,
            g_traj_list=g_traj,
            num_epochs=num_epochs,
            log_interval=vh_log_interval,
            tb_writer=self._get_tb_writer(),
            global_step_offset=self._vh_global_step,
        )
        self._vh_global_step += num_epochs

        # Save only the value head after pre-training — the VLM and expert
        # are unchanged during Stage 0, so no need to save them.
        output_dir = Path(self.cfg.get("training", {}).get("output_dir", "outputs/sft_advcond"))
        pretrain_dir = output_dir / "pretrained"
        pretrain_dir.mkdir(parents=True, exist_ok=True)

        torch.save(value_head.state_dict(), pretrain_dir / "value_head.pt")
        logger.info(
            "Saved pretrained value head to %s (vh loss=%.4f)",
            pretrain_dir,
            metrics["loss"],
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

        # ----- Phase 1: ROLLOUT -----
        logger.info("Phase 1: ROLLOUT — generating completions from current policy")
        t0 = time.time()
        rollout_results = self._rollout_phase(fresh_scenes, iteration)
        logger.info(
            "Phase 1 complete: %d completions in %.1fs", len(rollout_results), time.time() - t0
        )

        # ----- Phase 2: EVALUATE -----
        logger.info("Phase 2: EVALUATE — scoring and binarizing advantages")
        t0 = time.time()
        adv_labels = self._evaluate_phase(rollout_results)
        logger.info("Phase 2 complete in %.1fs", time.time() - t0)

        # ----- Phase 3: TRAIN -----
        logger.info("Phase 3: TRAIN — advantage-conditioned SFT + expert CFM")
        t0 = time.time()
        self._train_phase(rollout_results, adv_labels, iteration)
        logger.info("Phase 3 complete in %.1fs", time.time() - t0)

        # ----- Phase 4: BOOKKEEPING -----
        logger.info("Phase 4: BOOKKEEPING — updating replay buffer and checkpoints")
        self.replay_buffer.add(rollout_results, adv_labels, iteration)
        self._save_checkpoint(iteration)

        logger.info(
            "Iteration %d complete: total %.1fs",
            iteration,
            time.time() - iter_start,
        )

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

        # Generate completions
        results = engine.generate_completions(fresh_scenes, t0_us, G)
        logger.info(
            "Rollout phase complete: %d completions from %d scenes",
            len(results),
            len(fresh_scenes),
        )
        return results

    def _evaluate_phase(self, rollout_results: list[dict]) -> list[dict]:
        """Score completions and binarize advantages.

        1. Compute rewards using reward functions
        2. Extract segment hidden states via teacher-forced VLM forward
        3. Compute per-segment advantages using value head
        4. Update the advantage buffer
        5. Binarize advantages into conditioning labels

        Returns:
            List of dicts per completion: {i_obs, i_traj}
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
        reward_stash = engine.compute_rewards(rollout_results)
        if reward_stash:
            r_traj_vals = [r["r_traj"] for r in reward_stash]
            r_reason_vals = [r["r_reason"] for r in reward_stash]
            r_consist_vals = [r["r_consist"] for r in reward_stash]
            logger.info(
                "Rewards (%d completions): "
                "traj=[%.3f, %.3f, %.3f] reason=[%.3f, %.3f, %.3f] consist=[%.3f, %.3f, %.3f]",
                len(reward_stash),
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
        else:
            logger.info("Computed rewards for %d completions", len(reward_stash))

        # 2. Extract segment hidden states
        vh_cfg = self.cfg.get("value_head", {})
        if vh_cfg.get("enabled", False) and vh_cfg.get("segment_level", False):
            segment_hidden_stash, completion_segment_map = engine.extract_segment_hidden(
                rollout_results
            )

            # 3. Train value head to convergence on rollout data
            reward_weights = self._get_reward_weights()
            value_head = self._get_or_create_value_head()
            g_obs, g_coc, g_traj = compute_value_targets(
                segment_reward_stash=reward_stash,
                completion_segment_map=completion_segment_map,
                reward_weights=reward_weights,
            )
            vh_train_epochs = int(vh_cfg.get("train_epochs", 10))
            vh_log_interval = int(vh_cfg.get("log_interval", 10))
            train_segment_value_head(
                value_head=value_head,
                optimizer=self._value_head_optimizer,
                segment_hidden_stash=segment_hidden_stash,
                g_obs_list=g_obs,
                g_coc_list=g_coc,
                g_traj_list=g_traj,
                num_epochs=vh_train_epochs,
                log_interval=vh_log_interval,
                tb_writer=self._get_tb_writer(),
                global_step_offset=self._vh_global_step,
            )
            self._vh_global_step += vh_train_epochs

            # 4. Compute per-segment advantages using converged value head
            advantages = compute_segment_advantages_from_rollouts(
                segment_hidden_stash=segment_hidden_stash,
                segment_reward_stash=reward_stash,
                completion_segment_map=completion_segment_map,
                value_head=value_head,
                reward_weights=reward_weights,
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
        return adv_labels

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
        if self.full_model is not None:
            self.full_model.vlm.cpu()
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                logger.info(
                    "GPU cleanup after evaluate: %.1f/%.1f GB allocated/reserved",
                    torch.cuda.memory_allocated() / 1e9,
                    torch.cuda.memory_reserved() / 1e9,
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

            # Merge expert LoRA if present (prevents double-wrapping when
            # _setup_expert applies fresh LoRA in the next training phase)
            if hasattr(full_model.expert, "merge_and_unload"):
                full_model.expert = full_model.expert.merge_and_unload()
                logger.info("Merged previous expert LoRA weights into base model")

        # Register advantage tokens and resize embeddings
        register_advantage_tokens(self.processor.tokenizer)
        full_model.vlm.resize_token_embeddings(len(self.processor.tokenizer))
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

        # 3. Build training dataset (fresh + historical replay)
        train_dataset = self._build_training_dataset(rollout_results, adv_labels)

        # 4. Create training args
        output_dir = Path(train_cfg.get("output_dir", "outputs/sft_advcond")) / f"iter_{iteration}"
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
        )

        # 6. Train
        if torch.cuda.is_available():
            logger.info(
                "GPU mem before trainer.train(): %.1f/%.1f GB allocated/reserved",
                torch.cuda.memory_allocated() / 1e9,
                torch.cuda.memory_reserved() / 1e9,
            )
        logger.info("Starting SFT iteration %d: %d samples", iteration, len(train_dataset))
        trainer.train()

        # 7. Save
        trainer.save_model(str(output_dir / "final"))
        self.current_policy_path = str(output_dir / "final")
        logger.info("Saved pi_%d to %s", iteration + 1, self.current_policy_path)

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
        return AdvCondDataset(
            rollout_results=all_rollouts,
            adv_labels=all_labels,
            adv_token_ids=self.adv_token_ids,
            p_drop=float(adv_cfg.get("p_drop", 0.3)),
            traj_future_start_id=traj_future_start_id,
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
