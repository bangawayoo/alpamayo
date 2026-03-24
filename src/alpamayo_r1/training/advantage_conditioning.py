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

    Maintains separate deques for each advantage level (obs, traj).
    Uses percentile thresholds to binarize continuous advantages into
    positive/negative labels for conditioning.

    For the observation level, an EMA baseline is used to compute
    scene-difficulty-adjusted advantages: A_obs = G(s_obs) - ema.

    Args:
        k_obs: Percentile threshold for obs-level (0-100). Values above
            the k-th percentile are labeled positive.
        k_traj: Percentile threshold for traj-level.
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
        self._buf_traj: deque[float] = deque(maxlen=max_size)
        self._ema_obs: float | None = None

    def update(
        self,
        a_obs_list: list[float],
        a_traj_list: list[float],
    ) -> None:
        """Append new advantages to the rolling buffers.

        Args:
            a_obs_list: Per-completion observation-level advantages.
            a_traj_list: Per-completion trajectory-level advantages
                (mean across timesteps for each completion).
        """
        self._buf_obs.extend(a_obs_list)
        self._buf_traj.extend(a_traj_list)

        # Update EMA for observation level
        for a in a_obs_list:
            if self._ema_obs is None:
                self._ema_obs = a
            else:
                self._ema_obs = self.ema_alpha * self._ema_obs + (1 - self.ema_alpha) * a

    def compute_thresholds(self) -> tuple[float, float]:
        """Compute current percentile thresholds for binarization.

        Returns:
            (eps_obs, eps_traj) threshold values. Advantages above
            the threshold are labeled positive.
        """
        eps_obs = float(np.percentile(self._buf_obs, self.k_obs)) if self._buf_obs else 0.0
        eps_traj = float(np.percentile(self._buf_traj, self.k_traj)) if self._buf_traj else 0.0
        return eps_obs, eps_traj

    def binarize(self, a_obs: float, a_traj: float) -> tuple[bool, bool]:
        """Binarize per-level advantages using current thresholds.

        Args:
            a_obs: Observation-level advantage.
            a_traj: Trajectory-level advantage (mean over timesteps).

        Returns:
            (i_obs, i_traj): True = positive, False = negative.
        """
        eps_obs, eps_traj = self.compute_thresholds()
        return a_obs > eps_obs, a_traj > eps_traj

    def state_dict(self) -> dict:
        """Serialize buffer state for checkpointing."""
        return {
            "buf_obs": list(self._buf_obs),
            "buf_traj": list(self._buf_traj),
            "ema_obs": self._ema_obs,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer state from checkpoint."""
        self._buf_obs = deque(state["buf_obs"], maxlen=self.max_size)
        self._buf_traj = deque(state["buf_traj"], maxlen=self.max_size)
        self._ema_obs = state.get("ema_obs")


# ---------------------------------------------------------------------------
# 3c. Segment advantage computation (generalized from rollout.py)
# ---------------------------------------------------------------------------


def compute_segment_advantages_from_rollouts(
    segment_hidden_stash: list[dict],
    segment_reward_stash: list[dict],
    completion_segment_map: list[dict],
    value_head: torch.nn.Module,
    reward_weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> list[dict]:
    """Compute per-completion segment-level advantages (return minus baseline).

    Uses the return-minus-baseline formulation: at each information level,
    the advantage is the actual return from that state minus the value head's
    prediction. See docs/value-head.md "Approach A" for details.

    Args:
        segment_hidden_stash: List of {h_obs, h_coc, h_traj} per completion.
            h_obs: (1, D), h_coc: (1, D), h_traj: (T_traj, D).
        segment_reward_stash: List of {r_traj, r_reason, r_consist,
            r_traj_per_step} per completion.
        completion_segment_map: List of {coc_len, traj_len, traj_positions}
            per completion.
        value_head: SegmentValueHead instance.
        reward_weights: (w_traj, w_reason, w_consist).

    Returns:
        List of dicts per completion: {a_obs, a_traj, a_traj_per_step}.
        a_obs: observation-level advantage — G(s_obs) - V(s_obs)
        a_traj: mean trajectory advantage (over timesteps)
        a_traj_per_step: per-timestep trajectory advantages
    """
    from alpamayo_r1.training.value_head import SegmentValueHead

    w_traj, w_reason, w_consist = reward_weights
    vh_device = next(value_head.parameters()).device
    results = []

    for i in range(len(segment_hidden_stash)):
        seg = segment_hidden_stash[i]
        seg_map = completion_segment_map[i]
        seg_rew = segment_reward_stash[i]

        h_obs = seg["h_obs"].to(vh_device)
        h_traj = seg["h_traj"].to(vh_device)

        # Per-function rewards
        r_traj_scalar = seg_rew.get("r_traj", 0.0)
        r_reason = seg_rew.get("r_reason", 0.0)
        r_consist = seg_rew.get("r_consist", 0.0)

        # Per-token trajectory rewards and returns-to-go (computed over all tokens)
        T_traj = seg_map["traj_len"]
        if T_traj > 0:
            r_per_step = seg_rew.get("r_traj_per_step")
            if r_per_step is not None and len(r_per_step) == T_traj:
                r_per_step_t = torch.tensor(r_per_step, dtype=torch.float32, device=vh_device)
            else:
                r_per_step_t = torch.full(
                    (T_traj,), r_traj_scalar / max(T_traj, 1), device=vh_device
                )
            r_traj_weighted = w_traj * r_per_step_t
            r_traj_weighted[-1] = r_traj_weighted[-1] + w_consist * r_consist
            g_traj_all = torch.flip(torch.cumsum(torch.flip(r_traj_weighted, [0]), dim=0), [0])
        else:
            r_traj_weighted = torch.zeros(0, device=vh_device)
            g_traj_all = torch.zeros(0, device=vh_device)

        # Returns-to-go at observation state
        traj_total = r_traj_weighted.sum().item() if T_traj > 0 else 0.0
        g_obs = w_reason * r_reason + traj_total  # total return from s_obs

        # Value predictions at curvature positions only (idx % 2 == 1).
        # Trajectory tokens are (acceleration, curvature) pairs; the curvature
        # token has seen both components and is the natural V evaluation point.
        with torch.no_grad():
            v_obs = value_head(h_obs, level=SegmentValueHead.LEVEL_OBS).item()
            if T_traj > 0:
                curv_idx = torch.arange(1, T_traj, 2, device=vh_device)
                h_traj_curv = h_traj[curv_idx]
                traj_pos = (curv_idx // 2 + SegmentValueHead.POS_TRAJ_START).unsqueeze(0)
                v_traj = value_head(
                    h_traj_curv.unsqueeze(0), level=SegmentValueHead.LEVEL_TRAJ,
                    positions=traj_pos,
                ).squeeze(0)
                g_traj = g_traj_all[curv_idx]
            else:
                v_traj = torch.zeros(0, device=vh_device)
                g_traj = torch.zeros(0, device=vh_device)

        # A_obs: completion quality relative to scene baseline
        a_obs = g_obs - v_obs

        # A_traj: per-timestep trajectory advantages (curvature positions only)
        if T_traj > 0:
            a_traj_per_step = g_traj - v_traj
            a_traj_mean = a_traj_per_step.mean().item()
            a_traj_list = a_traj_per_step.cpu().tolist()
        else:
            a_traj_mean = 0.0
            a_traj_list = []

        results.append(
            {
                "a_obs": a_obs,
                "a_traj": a_traj_mean,
                "a_traj_per_step": a_traj_list,
            }
        )

    return results


# ---------------------------------------------------------------------------
# 3c-bis. Value head training
# ---------------------------------------------------------------------------


def compute_value_targets(
    segment_reward_stash: list[dict],
    completion_segment_map: list[dict],
    reward_weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> tuple[list[float], list[torch.Tensor]]:
    """Compute returns-to-go targets for value head training at obs and traj levels.

    Uses the same reward decomposition as compute_segment_advantages_from_rollouts,
    but returns the raw targets (not advantages) for MSE training.

    Args:
        segment_reward_stash: Per-completion {r_traj, r_reason, r_consist, r_traj_per_step}.
        completion_segment_map: Per-completion {coc_len, traj_len, traj_positions}.
        reward_weights: (w_traj, w_reason, w_consist).

    Returns:
        (g_obs_list, g_traj_list):
        - g_obs_list: per-completion total return G(s_obs)
        - g_traj_list: per-completion return-to-go tensors G(s_traj_j), shape (T_traj,) each
    """
    w_traj, w_reason, w_consist = reward_weights
    g_obs_list = []
    g_traj_list = []

    for i in range(len(segment_reward_stash)):
        seg_rew = segment_reward_stash[i]
        seg_map = completion_segment_map[i]

        r_traj_scalar = seg_rew.get("r_traj", 0.0)
        r_reason = seg_rew.get("r_reason", 0.0)
        r_consist = seg_rew.get("r_consist", 0.0)

        T_traj = seg_map["traj_len"]
        if T_traj > 0:
            r_per_step = seg_rew.get("r_traj_per_step")
            if r_per_step is not None and len(r_per_step) == T_traj:
                r_per_step_t = torch.tensor(r_per_step, dtype=torch.float32)
            else:
                r_per_step_t = torch.full((T_traj,), r_traj_scalar / max(T_traj, 1))
            r_traj_weighted = w_traj * r_per_step_t
            r_traj_weighted[-1] = r_traj_weighted[-1] + w_consist * r_consist
        else:
            r_traj_weighted = torch.zeros(0)

        traj_total = r_traj_weighted.sum().item() if T_traj > 0 else 0.0
        g_obs = w_reason * r_reason + traj_total

        g_obs_list.append(g_obs)

        if T_traj > 0:
            g_traj = torch.flip(torch.cumsum(torch.flip(r_traj_weighted, [0]), dim=0), [0])
            g_traj_list.append(g_traj)
        else:
            g_traj_list.append(torch.zeros(0))

    return g_obs_list, g_traj_list


def train_segment_value_head(
    value_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    segment_hidden_stash: list[dict],
    g_obs_list: list[float],
    g_traj_list: list[torch.Tensor],
    num_epochs: int = 10,
    batch_size: int = 64,
    log_interval: int = 10,
    tb_writer: Any | None = None,
    global_step_offset: int = 0,
) -> dict[str, float]:
    """Train the two-level value head on returns-to-go targets.

    Uses mini-batch SGD with shuffled indices each epoch. Obs hidden states
    are pre-stacked for efficient batched forward passes; traj tokens are
    processed per-completion within each mini-batch (variable-length sequences
    cannot be stacked without padding).

    Both obs and traj losses are per-completion averages: obs uses MSE over
    the batch, traj computes per-completion MSE across all curvature timesteps
    and then averages across completions. This keeps the two loss terms at
    comparable scales.

    Args:
        value_head: SegmentValueHead instance.
        optimizer: Optimizer for the value head parameters.
        segment_hidden_stash: Per-completion {h_obs, h_traj}.
        g_obs_list: Per-completion G(s_obs) targets.
        g_traj_list: Per-completion G(s_traj_j) tensors.
        num_epochs: Number of training epochs over the data.
        batch_size: Mini-batch size (default 64).
        log_interval: Log to display every N steps (default 10).
        tb_writer: Optional TensorBoard SummaryWriter for scalar logging.
        global_step_offset: Starting global step for TensorBoard (allows
            monotonically increasing steps across pretrain + iterations).

    Returns:
        Dict of final-step metrics: {loss, loss_obs, loss_traj,
        pred_obs_mean, target_obs_mean, total_steps}.
    """
    from alpamayo_r1.training.value_head import SegmentValueHead

    B = len(g_obs_list)
    if B == 0 or num_epochs <= 0:
        return {"loss": 0.0, "loss_obs": 0.0, "loss_traj": 0.0,
                "pred_obs_mean": 0.0, "target_obs_mean": 0.0, "total_steps": 0}

    vh_device = next(value_head.parameters()).device

    # Pre-stack obs into (B, D) / (B,) tensors on CPU for efficient indexing
    h_obs_all = torch.cat([segment_hidden_stash[i]["h_obs"] for i in range(B)], dim=0)
    g_obs_all = torch.tensor(g_obs_list, dtype=torch.float32)

    metrics = {}
    global_step = 0

    for epoch in range(num_epochs):
        perm = torch.randperm(B)

        for batch_start in range(0, B, batch_size):
            idx = perm[batch_start : batch_start + batch_size]

            # --- Obs-level loss ---
            h_obs_batch = h_obs_all[idx].to(vh_device)
            g_obs_batch = g_obs_all[idx].to(vh_device)
            obs_pred = value_head(h_obs_batch, level=SegmentValueHead.LEVEL_OBS)
            loss_obs = F.mse_loss(obs_pred, g_obs_batch)

            # --- Traj-level loss (all curvature positions) ---
            # Trajectory tokens come in (acceleration, curvature) pairs per timestep.
            # We evaluate V only at curvature positions (idx % 2 == 1) because:
            # (1) the curvature token has seen both components of the timestep,
            # (2) rewards are per-timestep, so the value function is naturally
            #     timestep-level rather than token-level.
            # Per-completion MSE is averaged across completions so that each
            # completion contributes equally regardless of trajectory length.
            traj_losses = []
            for i_local in idx.tolist():
                h_traj = segment_hidden_stash[i_local]["h_traj"]
                g_traj = g_traj_list[i_local]
                if h_traj.shape[0] == 0:
                    continue
                T_t = h_traj.shape[0]
                curvature_idx = torch.arange(1, T_t, 2)
                h_sub = h_traj[curvature_idx].to(vh_device)
                g_sub = g_traj[curvature_idx].to(vh_device)
                traj_pos = (curvature_idx // 2 + SegmentValueHead.POS_TRAJ_START).unsqueeze(0).to(vh_device)
                v = value_head(
                    h_sub.unsqueeze(0), level=SegmentValueHead.LEVEL_TRAJ, positions=traj_pos
                ).squeeze(0)
                traj_losses.append(F.mse_loss(v, g_sub))

            if traj_losses:
                loss_traj = torch.stack(traj_losses).mean()
            else:
                loss_traj = torch.tensor(0.0, device=vh_device)

            # --- Combined loss ---
            total_loss = (loss_obs + loss_traj) / 2.0

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            metrics = {
                "loss": total_loss.item(),
                "loss_obs": loss_obs.item(),
                "loss_traj": loss_traj.item(),
                "pred_obs_mean": obs_pred.detach().mean().item(),
                "target_obs_mean": g_obs_batch.mean().item(),
            }

            # TensorBoard: log every step
            if tb_writer is not None:
                step = global_step_offset + global_step
                for key, val in metrics.items():
                    tb_writer.add_scalar(f"value_head/{key}", val, step)

            # Display: log every log_interval steps or at epoch boundaries
            at_epoch_start = (batch_start == 0)
            at_log_interval = (global_step + 1) % max(1, log_interval) == 0
            if global_step == 0 or at_log_interval or at_epoch_start:
                logger.info(
                    "  Value head step %d (epoch %d/%d): "
                    "loss=%.4f (obs=%.4f traj=%.4f)",
                    global_step + 1,
                    epoch + 1,
                    num_epochs,
                    metrics["loss"],
                    metrics["loss_obs"],
                    metrics["loss_traj"],
                )

            global_step += 1

    total_steps = global_step
    metrics["total_steps"] = total_steps

    if tb_writer is not None:
        tb_writer.flush()

    logger.info(
        "Value head training: %d epochs, %d steps, %d samples (batch_size=%d) | "
        "loss=%.4f (obs=%.4f, traj=%.4f) | "
        "pred_obs=%.3f target_obs=%.3f",
        num_epochs,
        total_steps,
        B,
        batch_size,
        metrics["loss"],
        metrics["loss_obs"],
        metrics["loss_traj"],
        metrics["pred_obs_mean"],
        metrics["target_obs_mean"],
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
# 3e. AdvCondDataset
# ---------------------------------------------------------------------------


class AdvCondDataset(Dataset):
    """Dataset wrapping rollout results with advantage conditioning labels.

    Each item returns a training example with conditioning tokens inserted
    and labels masked appropriately for cross-entropy loss. Vision data
    (pixel_values, image_grid_thw) is pulled from the ClipDataCache's
    already-processed model_inputs, avoiding redundant re-processing.

    Args:
        rollout_results: List of dicts per completion with at least:
            {prompt_ids: list[int], completion_ids: list[int]}.
        adv_labels: List of dicts per completion with:
            {i_obs: bool, i_traj: bool}.
        adv_token_ids: Dict mapping token name -> ID.
        p_drop: Conditioning dropout probability.
        traj_future_start_id: Token ID of <|traj_future_start|> for finding
            CoC/trajectory boundary. If None, adv_traj is placed at end.
        data_cache: ClipDataCache for loading vision data.
    """

    def __init__(
        self,
        rollout_results: list[dict],
        adv_labels: list[dict],
        adv_token_ids: dict[str, int],
        p_drop: float = 0.3,
        traj_future_start_id: int | None = None,
        data_cache: Any | None = None,
    ) -> None:
        if len(rollout_results) != len(adv_labels):
            raise ValueError(
                f"Mismatched lengths: {len(rollout_results)} rollouts vs {len(adv_labels)} labels"
            )
        self._rollouts = rollout_results
        self._labels = adv_labels
        self._adv_token_ids = adv_token_ids
        self._p_drop = p_drop
        self._traj_future_start_id = traj_future_start_id
        self._data_cache = data_cache

    def __len__(self) -> int:
        return len(self._rollouts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rollout = self._rollouts[idx]
        label = self._labels[idx]

        item = build_conditioned_sequence(
            prompt_ids=rollout["prompt_ids"],
            completion_ids=rollout["completion_ids"],
            i_obs=label["i_obs"],
            i_traj=label["i_traj"],
            adv_token_ids=self._adv_token_ids,
            traj_future_start_id=self._traj_future_start_id,
            p_drop=self._p_drop,
        )

        # Propagate metadata for expert CFM step.
        # completion_prefix is set by build_conditioned_sequence and includes
        # advantage tokens when conditioning is active.
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

    def set_p_drop(self, p_drop: float) -> None:
        """Update dropout probability (e.g., per epoch)."""
        self._p_drop = p_drop
