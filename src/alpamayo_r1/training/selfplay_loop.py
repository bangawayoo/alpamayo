"""Iterative self-play loop for advantage-conditioned SFT.

Orchestrates three phases per iteration:
1. ROLLOUT: Generate G completions per scene from current policy
2. EVALUATE: Score with rewards + value head, compute and binarize advantages
3. TRAIN: Reset to pi_0, build advantage-conditioned dataset, SFT + expert CFM

Manages strict data partitioning (each scene used for fresh rollouts in exactly
one iteration) and a replay buffer for historical rollouts.

See docs/advantage-conditioning.md for the full design specification.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from physical_ai_av import PhysicalAIAVDatasetInterface

from alpamayo_r1.training.advantage_conditioning import (
    AdvantageBuffer,
    AdvCondDataset,
    compute_segment_advantages_from_rollouts,
    register_advantage_tokens,
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
        chunks = np.array_split(shuffled, num_iterations)
        self.partitions: list[list[str]] = [chunk.tolist() for chunk in chunks]

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
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
    ) -> None:
        """Recompute advantage labels for all buffer entries using current value head.

        This prevents stale advantage labels from degrading training quality
        as the value head improves across iterations.
        """
        if value_head is None or segment_hidden_stash is None:
            logger.debug("No value head or hidden stash — skipping label recomputation")
            return

        # Recompute advantages
        new_advantages = compute_segment_advantages_from_rollouts(
            segment_hidden_stash=segment_hidden_stash,
            segment_reward_stash=segment_reward_stash,
            completion_segment_map=completion_segment_map,
            value_head=value_head,
            reward_weights=reward_weights,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Update labels in buffer
        for entry, adv in zip(self._entries, new_advantages):
            i_obs, i_coc, i_traj = advantage_buffer.binarize(
                adv["a_obs"], adv["a_coc"], adv["a_traj"]
            )
            entry["adv_label"] = {"i_obs": i_obs, "i_coc": i_coc, "i_traj": i_traj}

        logger.info("Recomputed advantage labels for %d buffer entries", len(self._entries))

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Self-play loop
# ---------------------------------------------------------------------------


class SelfPlayLoop:
    """Orchestrate the iterative rollout -> evaluate -> train loop.

    Each iteration:
    1. Generate G completions per fresh scene (never used before)
    2. Score and binarize advantages
    3. Reset to pi_0, build advantage-conditioned dataset, train SFT + expert

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

        adv_cfg = cfg.get("advantage_conditioning", {})
        num_iterations = int(adv_cfg.get("num_iterations", 5))
        self.num_iterations = num_iterations

        self.partitioner = ScenePartitioner(all_clip_ids, num_iterations, seed=cfg.get("seed", 42))
        self.replay_buffer = RolloutReplayBuffer(
            max_size=int(adv_cfg.get("replay_buffer_max_size", 50000))
        )
        self.advantage_buffer = AdvantageBuffer(
            k_obs=float(adv_cfg.get("k_obs", 30)),
            k_coc=float(adv_cfg.get("k_coc", 30)),
            k_traj=float(adv_cfg.get("k_traj", 30)),
            ema_alpha=float(adv_cfg.get("ema_alpha", 0.99)),
        )

        self.base_model_path = cfg.get("model_name", "nvidia/Alpamayo-R1-10B")
        self.current_policy_path = self.base_model_path

        # Register advantage tokens once
        self.adv_token_ids = register_advantage_tokens(processor.tokenizer)

        # Replay ratio: fraction of training data from historical replay buffer
        self.replay_ratio = float(adv_cfg.get("replay_ratio", 0.3))

    def run(self) -> None:
        """Run all iterations of the self-play loop."""
        for i in range(self.num_iterations):
            logger.info("=" * 60)
            logger.info("SELF-PLAY ITERATION %d / %d", i + 1, self.num_iterations)
            logger.info("=" * 60)
            self.run_iteration(i)

    def run_iteration(self, iteration: int) -> None:
        """Run a single iteration of the self-play loop.

        Phase 1: ROLLOUT — Generate completions from current policy
        Phase 2: EVALUATE — Score, compute advantages, binarize
        Phase 3: TRAIN — Reset to pi_0, SFT with advantage conditioning + expert CFM
        Phase 4: BOOKKEEPING — Update replay buffer, save checkpoints
        """
        fresh_scenes = self.partitioner.get_fresh_scenes(iteration)
        logger.info(
            "Iteration %d: %d fresh scenes, %d historical in replay buffer",
            iteration,
            len(fresh_scenes),
            len(self.replay_buffer),
        )

        # ----- Phase 1: ROLLOUT -----
        logger.info("Phase 1: ROLLOUT — generating completions from current policy")
        rollout_results = self._rollout_phase(fresh_scenes, iteration)

        # ----- Phase 2: EVALUATE -----
        logger.info("Phase 2: EVALUATE — scoring and binarizing advantages")
        adv_labels = self._evaluate_phase(rollout_results)

        # ----- Phase 3: TRAIN -----
        logger.info("Phase 3: TRAIN — advantage-conditioned SFT + expert CFM")
        self._train_phase(rollout_results, adv_labels, iteration)

        # ----- Phase 4: BOOKKEEPING -----
        logger.info("Phase 4: BOOKKEEPING — updating replay buffer and checkpoints")
        self.replay_buffer.add(rollout_results, adv_labels, iteration)
        self._save_checkpoint(iteration)

    def _rollout_phase(self, fresh_scenes: list[str], iteration: int) -> list[dict]:
        """Generate G completions per fresh scene from current policy.

        This is a placeholder that should be implemented by the concrete
        subclass or by passing a RolloutEngine. The generation logic is adapted
        from _generate_single_turn() in rollout.py but decoupled from
        GRPOTrainer.

        Returns:
            List of dicts per completion with:
            {prompt_ids, completion_ids, pred_xyz, gt_xyz, coc_text,
             clip_id, t0_us, completion_prefix}
        """
        raise NotImplementedError(
            "Rollout phase not implemented. Override this method or use a RolloutEngine."
        )

    def _evaluate_phase(self, rollout_results: list[dict]) -> list[dict]:
        """Score completions and binarize advantages.

        This is a placeholder that computes rewards and advantages. A concrete
        implementation should:
        1. Compute rewards using trajectory_quality_reward, reasoning_quality_reward,
           consistency_reward
        2. Extract segment hidden states via teacher-forced VLM forward
        3. Compute per-segment advantages
        4. Update the advantage buffer
        5. Binarize advantages

        Returns:
            List of dicts per completion: {i_obs, i_coc, i_traj}
        """
        raise NotImplementedError("Evaluate phase not implemented. Override this method.")

    def _train_phase(
        self,
        rollout_results: list[dict],
        adv_labels: list[dict],
        iteration: int,
    ) -> None:
        """Reset to pi_0, build dataset, train SFT + expert.

        This is a placeholder. A concrete implementation should:
        1. Load pi_0 (base checkpoint)
        2. Register advantage tokens, resize embeddings
        3. Apply LoRA
        4. Build AdvCondDataset from fresh + historical rollouts
        5. Create AdvCondSFTTrainer
        6. Train
        7. Save as pi_{n+1}
        """
        raise NotImplementedError("Train phase not implemented. Override this method.")

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
        return AdvCondDataset(
            rollout_results=all_rollouts,
            adv_labels=all_labels,
            adv_token_ids=self.adv_token_ids,
            p_drop=float(adv_cfg.get("p_drop", 0.3)),
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """Save value head, advantage buffer, and replay buffer state."""
        output_dir = Path(self.cfg.get("training", {}).get("output_dir", "outputs/sft_advcond"))
        iter_dir = output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Save advantage buffer state
        adv_buf_path = iter_dir / "advantage_buffer.pt"
        torch.save(self.advantage_buffer.state_dict(), adv_buf_path)
        logger.info("Saved advantage buffer to %s", adv_buf_path)
