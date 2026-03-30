"""Advantage conditioning for iterative SFT training.

This module implements the two-level advantage conditioning pipeline:
1. Token registration (4 advantage tokens: obs/traj x pos/neg)
2. Advantage binarization via percentile thresholds
3. Conditioned sequence construction with causal placement (adv_obs before
   CoC, adv_traj between CoC and trajectory) and dropout for CFG
4. Dataset wrapper for the SFT trainer

See docs/advantage-conditioning.md for the design specification.
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from alpamayo_r1.models.base_model import ADV_CONDITIONING_TOKENS, IGNORE_INDEX

logger = logging.getLogger(__name__)

# The 4 advantage token string representations, from ADV_CONDITIONING_TOKENS
# in base_model.py. Two levels (obs + traj), two polarities each.
# Separate from SPECIAL_TOKENS to avoid changing vocab_size during model loading.
ADV_TOKEN_NAMES = list(ADV_CONDITIONING_TOKENS.keys())
ADV_TOKEN_STRINGS = dict(ADV_CONDITIONING_TOKENS)


# ---------------------------------------------------------------------------
# 3a. Token ID computation + trainable side embedding
# ---------------------------------------------------------------------------


def compute_advantage_token_ids(vocab_size: int) -> dict[str, int]:
    """Compute sentinel IDs for advantage tokens without modifying the tokenizer.

    Uses IDs just past the VLM's vocab_size boundary. These IDs never hit
    the embedding layer directly — the AdvantageEmbedding pre-hook intercepts
    them before lookup.

    Args:
        vocab_size: The VLM's original vocabulary size.

    Returns:
        Dict mapping token name -> sentinel token ID, e.g.
        {"adv_obs_pos": 155690, "adv_obs_neg": 155691, ...}
    """
    return {name: vocab_size + i for i, name in enumerate(ADV_TOKEN_NAMES)}


# Plain-text labels for text-mode advantage conditioning.
# Each advantage token is replaced by a short phrase that the tokenizer
# encodes into multiple real token IDs (no learned embeddings needed).
ADV_TEXT_LABELS = {
    "adv_obs_pos": "Observation advantage: Positive.",
    "adv_obs_neg": "Observation advantage: Negative.",
    "adv_traj_pos": "Trajectory advantage: Positive.",
    "adv_traj_neg": "Trajectory advantage: Negative.",
}


def compute_text_advantage_token_ids(tokenizer) -> dict[str, list[int]]:
    """Compute real token IDs for text-mode advantage conditioning.

    Instead of out-of-range sentinel IDs with learned embeddings, this
    tokenizes plain-text labels into sequences of real token IDs that
    the VLM already understands.

    Args:
        tokenizer: The VLM tokenizer.

    Returns:
        Dict mapping token name -> list of token IDs, e.g.
        {"adv_obs_pos": [12, 345, 67], "adv_obs_neg": [12, 345, 89], ...}
    """
    return {
        name: tokenizer.encode(text, add_special_tokens=False)
        for name, text in ADV_TEXT_LABELS.items()
    }


class AdvantageEmbedding(torch.nn.Module):
    """Small trainable embedding for advantage conditioning tokens.

    Uses a pre-hook to clamp out-of-range sentinel IDs before embed_tokens
    lookup, then a post-hook to replace those positions with learned embeddings.
    """

    def __init__(self, hidden_size: int, adv_token_ids: dict[str, int]):
        super().__init__()
        self.embedding = torch.nn.Embedding(len(adv_token_ids), hidden_size)
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # token_id -> local index (0..3)
        self._tid_to_idx = {tid: i for i, tid in enumerate(adv_token_ids.values())}
        self._sentinel_ids = set(adv_token_ids.values())
        self._pre_handle = None
        self._post_handle = None
        self._stashed_ids = None  # holds original input_ids across pre→post

    def attach(self, vlm: torch.nn.Module) -> None:
        """Register pre+post hooks on the VLM's input embedding layer."""
        self.detach()
        embed_layer = vlm.get_input_embeddings()
        self._pre_handle = embed_layer.register_forward_pre_hook(self._pre_hook)
        self._post_handle = embed_layer.register_forward_hook(self._post_hook)

    def detach(self) -> None:
        """Remove hooks."""
        if self._pre_handle is not None:
            self._pre_handle.remove()
            self._pre_handle = None
        if self._post_handle is not None:
            self._post_handle.remove()
            self._post_handle = None
        self._stashed_ids = None

    def _pre_hook(self, module, args):
        """Clamp sentinel IDs to 0 so embed_tokens doesn't crash on out-of-range."""
        input_ids = args[0]
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in self._sentinel_ids:
            mask |= (input_ids == tid)
        if mask.any():
            self._stashed_ids = input_ids  # save originals for post-hook
            safe_ids = input_ids.clone()
            safe_ids[mask] = 0
            return (safe_ids,) + args[1:]  # preserve any extra args
        self._stashed_ids = None
        return None  # no modification

    def _post_hook(self, module, args, output):
        """Replace embeddings at sentinel positions with learned embeddings."""
        if self._stashed_ids is None:
            return output
        input_ids = self._stashed_ids
        self._stashed_ids = None
        result = output
        for tid, idx in self._tid_to_idx.items():
            positions = (input_ids == tid)
            if positions.any():
                idx_t = torch.tensor(idx, device=output.device)
                emb = self.embedding(idx_t)  # (hidden_size,)
                result = torch.where(positions.unsqueeze(-1), emb, result)
        return result


# ---------------------------------------------------------------------------
# 3b. AdvantageBuffer
# ---------------------------------------------------------------------------


class AdvantageBuffer:
    """Rolling buffer for percentile-based advantage binarization.

    Uses a single advantage buffer (obs-level) with percentile thresholds
    to binarize continuous advantages into positive/negative labels.

    Args:
        k_obs: Percentile threshold for obs-level (0-100). Values above
            the k-th percentile are labeled positive.
        k_traj: Percentile threshold for traj-level (uses same buffer as obs).
        ema_alpha: EMA decay for observation-level baseline.
        max_size: Maximum number of entries per buffer.
    """

    def __init__(
        self,
        k_obs: float = 30.0,
        k_traj: float = 30.0,
        ema_alpha: float = 0.99,
        max_size: int = 10000,
    ) -> None:
        self.k_obs = k_obs
        self.k_traj = k_traj
        self.ema_alpha = ema_alpha
        self.max_size = max_size

        self._buf_obs: deque[float] = deque(maxlen=max_size)
        self._ema_obs: float | None = None

    def update(
        self,
        a_obs_list: list[float],
        a_traj_list: list[float] | None = None,
    ) -> None:
        """Append new advantages to the rolling buffer.

        Args:
            a_obs_list: Per-completion observation-level advantages.
            a_traj_list: Ignored (kept for backward compat). Traj advantages
                are the same as obs advantages with obs-only value head.
        """
        self._buf_obs.extend(a_obs_list)

        for a in a_obs_list:
            if self._ema_obs is None:
                self._ema_obs = a
            else:
                self._ema_obs = self.ema_alpha * self._ema_obs + (1 - self.ema_alpha) * a

    def compute_thresholds(self) -> tuple[float, float]:
        """Compute current percentile thresholds for binarization.

        Returns:
            (eps_obs, eps_traj) threshold values. Both use the same buffer.
        """
        eps_obs = float(np.percentile(self._buf_obs, self.k_obs)) if self._buf_obs else 0.0
        eps_traj = float(np.percentile(self._buf_obs, self.k_traj)) if self._buf_obs else 0.0
        return eps_obs, eps_traj

    def binarize(self, a_obs: float, a_traj: float) -> tuple[bool, bool]:
        """Binarize advantages using current thresholds.

        Args:
            a_obs: Observation-level advantage.
            a_traj: Trajectory-level advantage (same as a_obs with obs-only VH).

        Returns:
            (i_obs, i_traj): True = positive, False = negative.
        """
        eps_obs, eps_traj = self.compute_thresholds()
        return a_obs > eps_obs, a_traj > eps_traj

    def state_dict(self) -> dict:
        """Serialize buffer state for checkpointing."""
        return {
            "buf_obs": list(self._buf_obs),
            "ema_obs": self._ema_obs,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer state from checkpoint."""
        self._buf_obs = deque(state["buf_obs"], maxlen=self.max_size)
        self._ema_obs = state.get("ema_obs")


# ---------------------------------------------------------------------------
# 3c. Reward normalization
# ---------------------------------------------------------------------------


def normalize_reward_components(
    reward_stash: list[dict],
    eps: float = 1e-8,
) -> list[dict]:
    """Per-component z-score normalization across a batch of completions.

    Normalizes r_traj, r_reason, r_consist independently to zero mean and
    unit variance so that no single component dominates the combined return
    due to scale differences.

    Args:
        reward_stash: List of {r_traj, r_reason, r_consist, ...} per completion.
        eps: Small constant to avoid division by zero.

    Returns:
        New list of dicts with normalized r_traj, r_reason, r_consist.
        Other keys are preserved unchanged.
    """
    if len(reward_stash) < 2:
        return reward_stash

    keys = ("r_traj", "r_reason", "r_consist")
    arrays = {}
    for k in keys:
        vals = np.array([s.get(k, 0.0) for s in reward_stash])
        std = vals.std()
        if std > eps:
            arrays[k] = (vals - vals.mean()) / std
        else:
            arrays[k] = np.zeros_like(vals)

    result = []
    for i, s in enumerate(reward_stash):
        new_s = dict(s)
        for k in keys:
            new_s[k] = float(arrays[k][i])
        result.append(new_s)
    return result


def compute_obs_return(
    r_traj: float,
    r_reason: float,
    r_consist: float,
    w_traj: float,
    w_reason: float,
    w_consist: float,
) -> float:
    """Compute observation-level return G(s_obs) = weighted sum of rewards."""
    return w_traj * r_traj + w_reason * r_reason + w_consist * r_consist


# ---------------------------------------------------------------------------
# 3d. Segment advantage computation (generalized from rollout.py)
# ---------------------------------------------------------------------------


def compute_segment_advantages_from_rollouts(
    segment_hidden_stash: list[dict],
    segment_reward_stash: list[dict],
    completion_segment_map: list[dict],
    value_head: torch.nn.Module,
    reward_weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> list[dict]:
    """Compute per-completion advantages using V(obs) as the sole baseline.

    Rewards are normalized per-component before weighting, so no single
    reward dominates due to variance differences.

    Args:
        segment_hidden_stash: List of {h_obs} per completion.
        segment_reward_stash: List of {r_traj, r_reason, r_consist} per completion.
        completion_segment_map: List of {coc_len, traj_len, traj_positions}.
        value_head: SegmentValueHead instance.
        reward_weights: (w_traj, w_reason, w_consist).

    Returns:
        List of dicts per completion: {a_obs, a_traj}.
        Both use V(obs) as baseline: A = G(s_obs) - V(s_obs).
    """
    from alpamayo_r1.training.value_head import SegmentValueHead

    B = len(segment_hidden_stash)
    if B == 0:
        return []

    vh_device = next(value_head.parameters()).device

    g_obs_list = compute_value_targets(segment_reward_stash, reward_weights)

    # Batched V(obs) inference
    h_obs_all = torch.cat(
        [segment_hidden_stash[i]["h_obs"] for i in range(B)], dim=0
    ).to(vh_device)

    with torch.no_grad():
        v_obs_all = value_head(h_obs_all, level=SegmentValueHead.LEVEL_OBS)

    # Assemble advantages: A = G(obs) - V(obs) for both obs and traj levels
    results = []
    for i in range(B):
        a = g_obs_list[i] - v_obs_all[i].item()
        results.append({"a_obs": a, "a_traj": a})

    return results


# ---------------------------------------------------------------------------
# 3c-bis. Value head training
# ---------------------------------------------------------------------------


def compute_value_targets(
    segment_reward_stash: list[dict],
    reward_weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> list[float]:
    """Compute obs-level return targets G(s_obs) for value head training.

    Rewards are normalized per-component before weighting.

    Args:
        segment_reward_stash: Per-completion {r_traj, r_reason, r_consist}.
        reward_weights: (w_traj, w_reason, w_consist).

    Returns:
        g_obs_list: per-completion total return G(s_obs).
    """
    w_traj, w_reason, w_consist = reward_weights
    normed_stash = normalize_reward_components(segment_reward_stash)

    g_obs_list = []
    for seg_rew in normed_stash:
        g_obs = compute_obs_return(
            seg_rew.get("r_traj", 0.0), seg_rew.get("r_reason", 0.0),
            seg_rew.get("r_consist", 0.0), w_traj, w_reason, w_consist,
        )
        g_obs_list.append(g_obs)

    return g_obs_list


def train_segment_value_head(
    value_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    segment_hidden_stash: list[dict],
    g_obs_list: list[float],
    num_epochs: int = 10,
    batch_size: int = 64,
    log_interval: int = 10,
    tb_writer: Any | None = None,
    global_step_offset: int = 0,
) -> dict[str, float]:
    """Train the obs-level value head on G(s_obs) targets.

    Uses mini-batch SGD with shuffled indices each epoch. Under DDP,
    each rank processes a different shard; gradients are synchronized
    automatically if ``value_head`` is DDP-wrapped.

    Args:
        value_head: SegmentValueHead instance (optionally DDP-wrapped).
        optimizer: Optimizer for the value head parameters.
        segment_hidden_stash: Per-completion {h_obs}.
        g_obs_list: Per-completion G(s_obs) targets.
        num_epochs: Number of training epochs over the data.
        batch_size: Mini-batch size (default 64, **per rank** under DDP).
        log_interval: Log to display every N steps (default 10).
        tb_writer: Optional TensorBoard SummaryWriter for scalar logging.
        global_step_offset: Starting global step for TensorBoard.

    Returns:
        Dict of final-step metrics: {loss, pred_obs_mean, target_obs_mean,
        total_steps}.
    """
    import torch.distributed as dist
    from alpamayo_r1.training.value_head import SegmentValueHead

    B = len(g_obs_list)
    if B == 0 or num_epochs <= 0:
        return {"loss": 0.0, "pred_obs_mean": 0.0, "target_obs_mean": 0.0, "total_steps": 0}

    vh_device = next(value_head.parameters()).device

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    h_obs_all = torch.cat([segment_hidden_stash[i]["h_obs"] for i in range(B)], dim=0)
    g_obs_all = torch.tensor(g_obs_list, dtype=torch.float32)

    from tqdm import tqdm

    metrics = {"loss": 0.0, "pred_obs_mean": 0.0, "target_obs_mean": 0.0}
    global_step = 0
    total_batches = num_epochs * math.ceil(math.ceil(B / world_size) / batch_size)
    pbar = tqdm(total=total_batches, desc="VH train", unit="step", disable=rank != 0)

    for epoch in range(num_epochs):
        g = torch.Generator().manual_seed(epoch * 31337)
        perm = torch.randperm(B, generator=g)

        rank_indices = perm[rank::world_size]
        max_per_rank = math.ceil(B / world_size)
        if len(rank_indices) < max_per_rank:
            extra = perm[: max_per_rank - len(rank_indices)]
            rank_indices = torch.cat([rank_indices, extra])

        for batch_start in range(0, len(rank_indices), batch_size):
            idx = rank_indices[batch_start : batch_start + batch_size]

            h_obs_batch = h_obs_all[idx].to(vh_device)
            g_obs_batch = g_obs_all[idx].to(vh_device)
            obs_pred = value_head(h_obs_batch, level=SegmentValueHead.LEVEL_OBS)
            loss = F.mse_loss(obs_pred, g_obs_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = {
                "loss": loss.item(),
                "pred_obs_mean": obs_pred.detach().mean().item(),
                "target_obs_mean": g_obs_batch.mean().item(),
            }

            if tb_writer is not None:
                step = global_step_offset + global_step
                for key, val in metrics.items():
                    tb_writer.add_scalar(f"value_head/{key}", val, step)

            at_epoch_start = (batch_start == 0)
            at_log_interval = (global_step + 1) % max(1, log_interval) == 0
            if global_step == 0 or at_log_interval or at_epoch_start:
                logger.debug(
                    "  Value head step %d (epoch %d/%d): loss=%.4f",
                    global_step + 1, epoch + 1, num_epochs, metrics["loss"],
                )

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

    pbar.close()
    metrics["total_steps"] = global_step

    if tb_writer is not None:
        tb_writer.flush()

    logger.debug(
        "Value head training: %d epochs, %d steps, %d samples (batch_size=%d) | "
        "loss=%.4f | pred_obs=%.3f target_obs=%.3f",
        num_epochs, global_step, B, batch_size,
        metrics["loss"], metrics["pred_obs_mean"], metrics["target_obs_mean"],
    )
    return metrics


# ---------------------------------------------------------------------------
# 3d. Conditioned sequence construction
# ---------------------------------------------------------------------------


def _find_completion_prefix(
    completion_ids: list[int], traj_future_start_id: int | None
) -> list[int]:
    """Return completion tokens up to and including <traj_future_start>."""
    if traj_future_start_id is not None:
        for idx, tid in enumerate(completion_ids):
            if tid == traj_future_start_id:
                return completion_ids[: idx + 1]
    return list(completion_ids)


def build_conditioned_sequence(
    prompt_ids: list[int],
    completion_ids: list[int],
    i_obs: bool,
    i_traj: bool,
    adv_token_ids: dict[str, int | list[int]],
    traj_future_start_id: int | None = None,
    p_drop: float = 0.3,
) -> dict[str, Any]:
    """Construct a training sequence with causally-placed advantage tokens.

    Each conditioning token is placed immediately before the segment it
    conditions: adv_obs before CoC (conditions entire completion), adv_traj
    between CoC and trajectory (conditions only trajectory generation).

    Implements conditioning dropout for classifier-free guidance:
    all-positive completions have a p_drop probability of being treated as
    unconditional (no conditioning tokens).

    Args:
        prompt_ids: Tokenized prompt (with fused history trajectory).
        completion_ids: Tokenized completion (CoC + trajectory tokens).
        i_obs: Binarized observation advantage (True=positive).
        i_traj: Binarized trajectory advantage (True=positive).
        adv_token_ids: Dict mapping token name -> ID (embedding mode) or
            list of IDs (text mode).
        traj_future_start_id: Token ID of <|traj_future_start|> used to find
            the CoC/trajectory boundary. If None, adv_traj is placed at the
            end of the completion (fallback).
        p_drop: Conditioning dropout probability for all-positive completions.

    Returns:
        Dict with:
        - input_ids: list[int] — full sequence
        - labels: list[int] — IGNORE_INDEX on prompt+conditioning, real IDs on completion
        - attention_mask: list[int] — all 1s
        - is_unconditional: bool — True if this is an unconditional example
    """
    # When adv_token_ids is None, skip all conditioning (plain SFT)
    if adv_token_ids is None:
        input_ids = prompt_ids + completion_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
        # completion_prefix: CoC tokens up to and including <traj_future_start>
        comp_prefix = _find_completion_prefix(completion_ids, traj_future_start_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "is_unconditional": True,
            "completion_prefix": comp_prefix,
        }

    is_all_positive = i_obs and i_traj
    is_unconditional = is_all_positive and random.random() < p_drop

    if is_unconditional:
        # Unconditional path: no conditioning tokens
        input_ids = prompt_ids + completion_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
        # No advantage tokens — prefix is just CoC + <traj_future_start>
        comp_prefix = _find_completion_prefix(completion_ids, traj_future_start_id)
    else:
        # Conditional path: split placement
        # adv_obs goes after prompt (before CoC)
        adv_obs_token = adv_token_ids["adv_obs_pos" if i_obs else "adv_obs_neg"]
        adv_traj_token = adv_token_ids["adv_traj_pos" if i_traj else "adv_traj_neg"]

        # Normalize to list for uniform handling (int for embedding mode,
        # list[int] for text mode)
        adv_obs_token = [adv_obs_token] if isinstance(adv_obs_token, int) else list(adv_obs_token)
        adv_traj_token = [adv_traj_token] if isinstance(adv_traj_token, int) else list(adv_traj_token)

        # Find CoC/trajectory boundary in completion_ids
        traj_boundary = len(completion_ids)  # fallback: end of completion
        if traj_future_start_id is not None:
            for idx, tid in enumerate(completion_ids):
                if tid == traj_future_start_id:
                    traj_boundary = idx
                    break

        coc_part = completion_ids[:traj_boundary]
        traj_part = completion_ids[traj_boundary:]

        # [prompt] [adv_obs...] [CoC tokens...] [adv_traj...] [trajectory tokens...]
        input_ids = prompt_ids + adv_obs_token + coc_part + adv_traj_token + traj_part
        labels = (
            [IGNORE_INDEX] * (len(prompt_ids) + len(adv_obs_token))  # prompt + adv_obs
            + coc_part  # CoC tokens (supervised)
            + [IGNORE_INDEX] * len(adv_traj_token)  # adv_traj
            + traj_part  # trajectory tokens (supervised)
        )

        # Conditioned prefix for expert KV cache: includes advantage tokens
        # [adv_obs...] [CoC tokens...] [adv_traj...] [<traj_future_start>]
        traj_start = traj_part[:1] if traj_part else []
        comp_prefix = adv_obs_token + coc_part + adv_traj_token + traj_start

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "is_unconditional": is_unconditional,
        "completion_prefix": comp_prefix,
    }


# ---------------------------------------------------------------------------
# 3e. Pre-computation + AdvCondDataset
# ---------------------------------------------------------------------------


def precompute_conditioned_sequences(
    rollout_results: list[dict],
    adv_labels: list[dict],
    adv_token_ids: dict[str, int | list[int]] | None,
    traj_future_start_id: int | None = None,
) -> list[dict]:
    """Pre-compute both conditional and unconditional sequences for all samples.

    Moves the per-sample scanning (finding traj_future_start boundary) and
    list concatenation out of the DataLoader hot path and into a single
    batch pre-computation step. The returned list is consumed by
    AdvCondDataset, which only needs a cheap random check for p_drop at
    __getitem__ time.

    Args:
        rollout_results: List of rollout dicts with prompt_ids, completion_ids.
        adv_labels: List of dicts with i_obs, i_traj booleans.
        adv_token_ids: Advantage token IDs (or None for plain SFT).
        traj_future_start_id: Token ID of <|traj_future_start|>.

    Returns:
        List of dicts, one per sample, each containing:
        - "cond": dict with input_ids, labels, attention_mask, completion_prefix
        - "uncond": dict with input_ids, labels, attention_mask, completion_prefix
        - "is_all_positive": bool — whether dropout can apply
    """
    precomputed = []
    for rollout, label in zip(rollout_results, adv_labels):
        prompt_ids = rollout["prompt_ids"]
        completion_ids = rollout["completion_ids"]
        i_obs = label["i_obs"]
        i_traj = label["i_traj"]

        # --- Unconditional version (always needed for dropout fallback) ---
        uncond_input_ids = prompt_ids + completion_ids
        uncond_labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
        uncond_prefix = _find_completion_prefix(completion_ids, traj_future_start_id)
        uncond = {
            "input_ids": uncond_input_ids,
            "labels": uncond_labels,
            "attention_mask": [1] * len(uncond_input_ids),
            "completion_prefix": uncond_prefix,
        }

        # --- Conditional version ---
        if adv_token_ids is None:
            # Plain SFT: conditional == unconditional
            cond = uncond
        else:
            adv_obs_token = adv_token_ids["adv_obs_pos" if i_obs else "adv_obs_neg"]
            adv_traj_token = adv_token_ids["adv_traj_pos" if i_traj else "adv_traj_neg"]

            adv_obs_token = (
                [adv_obs_token] if isinstance(adv_obs_token, int) else list(adv_obs_token)
            )
            adv_traj_token = (
                [adv_traj_token] if isinstance(adv_traj_token, int) else list(adv_traj_token)
            )

            # Find CoC/trajectory boundary
            traj_boundary = len(completion_ids)
            if traj_future_start_id is not None:
                for idx, tid in enumerate(completion_ids):
                    if tid == traj_future_start_id:
                        traj_boundary = idx
                        break

            coc_part = completion_ids[:traj_boundary]
            traj_part = completion_ids[traj_boundary:]

            cond_input_ids = prompt_ids + adv_obs_token + coc_part + adv_traj_token + traj_part
            cond_labels = (
                [IGNORE_INDEX] * (len(prompt_ids) + len(adv_obs_token))
                + coc_part
                + [IGNORE_INDEX] * len(adv_traj_token)
                + traj_part
            )
            traj_start = traj_part[:1] if traj_part else []
            cond_prefix = adv_obs_token + coc_part + adv_traj_token + traj_start

            cond = {
                "input_ids": cond_input_ids,
                "labels": cond_labels,
                "attention_mask": [1] * len(cond_input_ids),
                "completion_prefix": cond_prefix,
            }

        precomputed.append(
            {
                "cond": cond,
                "uncond": uncond,
                "is_all_positive": i_obs and i_traj,
            }
        )

    return precomputed


class AdvCondDataset(Dataset):
    """Dataset wrapping rollout results with pre-computed advantage conditioning.

    Sequences (input_ids, labels, attention_mask, completion_prefix) are
    pre-built by precompute_conditioned_sequences() before training starts.
    At __getitem__ time the only CPU work is a single random check for
    conditioning dropout, plus vision data retrieval from the cache.

    Args:
        rollout_results: List of dicts per completion (metadata for vision
            data lookup and expert CFM: clip_id, t0_us, etc.).
        precomputed: Pre-computed sequences from precompute_conditioned_sequences().
        p_drop: Conditioning dropout probability.
        data_cache: ClipDataCache for loading vision data.
    """

    def __init__(
        self,
        rollout_results: list[dict],
        precomputed: list[dict],
        p_drop: float = 0.3,
        data_cache: Any | None = None,
    ) -> None:
        if len(rollout_results) != len(precomputed):
            raise ValueError(
                f"Mismatched lengths: {len(rollout_results)} rollouts vs "
                f"{len(precomputed)} precomputed"
            )
        self._rollouts = rollout_results
        self._precomputed = precomputed
        self._p_drop = p_drop
        self._data_cache = data_cache

    def __len__(self) -> int:
        return len(self._rollouts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rollout = self._rollouts[idx]
        pc = self._precomputed[idx]
        is_unconditional = pc["is_all_positive"] and random.random() < self._p_drop
        seq = pc["uncond"] if is_unconditional else pc["cond"]
        item = {
            "input_ids": seq["input_ids"],
            "labels": seq["labels"],
            "attention_mask": seq["attention_mask"],
            "is_unconditional": is_unconditional,
            "completion_prefix": seq["completion_prefix"],
        }

        # Propagate metadata for expert CFM step.
        if "clip_id" in rollout:
            item["clip_id"] = rollout["clip_id"]
        if "t0_us" in rollout:
            item["t0_us"] = rollout["t0_us"]
        if "expert_fut_xyz" in rollout:
            item["expert_fut_xyz"] = rollout["expert_fut_xyz"]
        if "expert_fut_rot" in rollout:
            item["expert_fut_rot"] = rollout["expert_fut_rot"]
        if "hist_xyz" in rollout:
            item["hist_xyz"] = rollout["hist_xyz"]
        if "hist_rot" in rollout:
            item["hist_rot"] = rollout["hist_rot"]

        # Fetch vision data from cached model_inputs (already processor-encoded)
        if self._data_cache is not None and "clip_id" in rollout:
            clip_id = rollout["clip_id"]
            t0_us = rollout.get("t0_us", 5_100_000)
            model_inputs, _ = self._data_cache.get(clip_id, t0_us, device="cpu")
            tokenized = model_inputs["tokenized_data"]
            if "pixel_values" in tokenized:
                item["pixel_values"] = tokenized["pixel_values"].squeeze(0)
            if "image_grid_thw" in tokenized:
                item["image_grid_thw"] = tokenized["image_grid_thw"].squeeze(0)

        return item
