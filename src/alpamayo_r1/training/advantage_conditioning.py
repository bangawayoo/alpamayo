"""Advantage conditioning for iterative SFT training.

This module implements the three-level advantage conditioning pipeline:
1. Token registration (6 advantage tokens: obs/coc/traj x pos/neg)
2. Advantage binarization via percentile thresholds
3. Conditioned sequence construction (with dropout for CFG)
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
from torch.utils.data import Dataset

from alpamayo_r1.models.base_model import IGNORE_INDEX, SPECIAL_TOKENS

logger = logging.getLogger(__name__)

# The 6 advantage token string representations, derived from SPECIAL_TOKENS_KEYS
# in base_model.py. The keys must match those added in Step 2.
ADV_TOKEN_NAMES = [
    "adv_obs_pos",
    "adv_obs_neg",
    "adv_coc_pos",
    "adv_coc_neg",
    "adv_traj_pos",
    "adv_traj_neg",
]
ADV_TOKEN_STRINGS = {name: SPECIAL_TOKENS[name] for name in ADV_TOKEN_NAMES}


# ---------------------------------------------------------------------------
# 3a. Token registration
# ---------------------------------------------------------------------------


def register_advantage_tokens(tokenizer) -> dict[str, int]:
    """Add 6 advantage conditioning tokens to the tokenizer.

    Should be called once during SFT model setup. After calling this, the
    caller must resize model embeddings:
        model.vlm.resize_token_embeddings(len(tokenizer))

    Args:
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        Dict mapping token name -> token ID, e.g.
        {"adv_obs_pos": 155690, "adv_obs_neg": 155691, ...}
    """
    tokens_to_add = list(ADV_TOKEN_STRINGS.values())

    # Check if already registered
    existing = tokenizer.convert_tokens_to_ids(tokens_to_add[0])
    if existing != tokenizer.unk_token_id:
        logger.info("Advantage tokens already registered (first token ID=%d)", existing)
    else:
        num_added = tokenizer.add_tokens(tokens_to_add, special_tokens=True)
        logger.info("Registered %d advantage conditioning tokens", num_added)

    # Build name -> id mapping
    adv_token_ids = {}
    for name in ADV_TOKEN_NAMES:
        token_str = ADV_TOKEN_STRINGS[name]
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid == tokenizer.unk_token_id:
            raise ValueError(f"Failed to register advantage token {token_str!r}")
        adv_token_ids[name] = tid
    return adv_token_ids


# ---------------------------------------------------------------------------
# 3b. AdvantageBuffer
# ---------------------------------------------------------------------------


class AdvantageBuffer:
    """Rolling buffer for percentile-based advantage binarization.

    Maintains separate deques for each advantage level (obs, coc, traj).
    Uses percentile thresholds to binarize continuous advantages into
    positive/negative labels for conditioning.

    For the observation level, an EMA baseline is used to compute
    scene-difficulty-adjusted advantages: A_obs = G(s_obs) - ema.

    Args:
        k_obs: Percentile threshold for obs-level (0-100). Values above
            the k-th percentile are labeled positive.
        k_coc: Percentile threshold for coc-level.
        k_traj: Percentile threshold for traj-level.
        ema_alpha: EMA decay for observation-level baseline.
        max_size: Maximum number of entries per buffer.
    """

    def __init__(
        self,
        k_obs: float = 30.0,
        k_coc: float = 30.0,
        k_traj: float = 30.0,
        ema_alpha: float = 0.99,
        max_size: int = 10000,
    ) -> None:
        self.k_obs = k_obs
        self.k_coc = k_coc
        self.k_traj = k_traj
        self.ema_alpha = ema_alpha
        self.max_size = max_size

        self._buf_obs: deque[float] = deque(maxlen=max_size)
        self._buf_coc: deque[float] = deque(maxlen=max_size)
        self._buf_traj: deque[float] = deque(maxlen=max_size)
        self._ema_obs: float | None = None

    def update(
        self,
        a_obs_list: list[float],
        a_coc_list: list[float],
        a_traj_list: list[float],
    ) -> None:
        """Append new advantages to the rolling buffers.

        Args:
            a_obs_list: Per-completion observation-level advantages.
            a_coc_list: Per-completion CoC-level advantages.
            a_traj_list: Per-completion trajectory-level advantages
                (mean across timesteps for each completion).
        """
        self._buf_obs.extend(a_obs_list)
        self._buf_coc.extend(a_coc_list)
        self._buf_traj.extend(a_traj_list)

        # Update EMA for observation level
        for a in a_obs_list:
            if self._ema_obs is None:
                self._ema_obs = a
            else:
                self._ema_obs = self.ema_alpha * self._ema_obs + (1 - self.ema_alpha) * a

    def compute_thresholds(self) -> tuple[float, float, float]:
        """Compute current percentile thresholds for binarization.

        Returns:
            (eps_obs, eps_coc, eps_traj) threshold values. Advantages above
            the threshold are labeled positive.
        """
        eps_obs = float(np.percentile(self._buf_obs, self.k_obs)) if self._buf_obs else 0.0
        eps_coc = float(np.percentile(self._buf_coc, self.k_coc)) if self._buf_coc else 0.0
        eps_traj = float(np.percentile(self._buf_traj, self.k_traj)) if self._buf_traj else 0.0
        return eps_obs, eps_coc, eps_traj

    def binarize(self, a_obs: float, a_coc: float, a_traj: float) -> tuple[bool, bool, bool]:
        """Binarize per-level advantages using current thresholds.

        Args:
            a_obs: Observation-level advantage.
            a_coc: CoC-level advantage.
            a_traj: Trajectory-level advantage (mean over timesteps).

        Returns:
            (i_obs, i_coc, i_traj): True = positive, False = negative.
        """
        eps_obs, eps_coc, eps_traj = self.compute_thresholds()
        return a_obs >= eps_obs, a_coc >= eps_coc, a_traj >= eps_traj

    def state_dict(self) -> dict:
        """Serialize buffer state for checkpointing."""
        return {
            "buf_obs": list(self._buf_obs),
            "buf_coc": list(self._buf_coc),
            "buf_traj": list(self._buf_traj),
            "ema_obs": self._ema_obs,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer state from checkpoint."""
        self._buf_obs = deque(state["buf_obs"], maxlen=self.max_size)
        self._buf_coc = deque(state["buf_coc"], maxlen=self.max_size)
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
        List of dicts per completion: {a_obs, a_coc, a_traj, a_traj_per_step}.
        a_obs: observation-level advantage — G(s_obs) - V(s_obs)
        a_coc: CoC-level advantage — G(s_coc) - V(s_coc)
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
        h_coc = seg["h_coc"].to(vh_device)
        h_traj = seg["h_traj"].to(vh_device)

        # Value predictions
        with torch.no_grad():
            v_obs = value_head(h_obs, level=SegmentValueHead.LEVEL_OBS).item()
            v_coc = value_head(h_coc, level=SegmentValueHead.LEVEL_COC).item()
            if h_traj.shape[0] > 0:
                v_traj = value_head(h_traj.unsqueeze(0), level=SegmentValueHead.LEVEL_TRAJ).squeeze(
                    0
                )
            else:
                v_traj = torch.zeros(0, device=vh_device)

        # Per-function rewards
        r_traj_scalar = seg_rew.get("r_traj", 0.0)
        r_reason = seg_rew.get("r_reason", 0.0)
        r_consist = seg_rew.get("r_consist", 0.0)

        # Per-timestep trajectory rewards
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
        else:
            r_traj_weighted = torch.zeros(0, device=vh_device)

        # Returns-to-go at each state
        traj_total = r_traj_weighted.sum().item() if T_traj > 0 else 0.0
        g_obs = w_reason * r_reason + traj_total  # total return from s_obs
        g_coc = traj_total  # remaining return from s_coc (R_reasoning already collected)

        # Advantages: actual return minus value baseline at each information level
        # A_obs: completion quality relative to scene baseline
        a_obs = g_obs - v_obs

        # A_coc: trajectory quality relative to CoC-conditioned baseline
        a_coc = g_coc - v_coc

        # A_traj_j: remaining trajectory quality at each step
        if T_traj > 0:
            # G(s_traj_j) = w_traj * Σ_{t=j}^{T} r_t + w_consist * R_consistency
            g_traj = torch.flip(torch.cumsum(torch.flip(r_traj_weighted, [0]), dim=0), [0])
            a_traj_per_step = g_traj - v_traj
            a_traj_mean = a_traj_per_step.mean().item()
            a_traj_list = a_traj_per_step.cpu().tolist()
        else:
            a_traj_mean = 0.0
            a_traj_list = []

        results.append(
            {
                "a_obs": a_obs,
                "a_coc": a_coc,
                "a_traj": a_traj_mean,
                "a_traj_per_step": a_traj_list,
            }
        )

    return results


# ---------------------------------------------------------------------------
# 3d. Conditioned sequence construction
# ---------------------------------------------------------------------------


def build_conditioned_sequence(
    prompt_ids: list[int],
    completion_ids: list[int],
    i_obs: bool,
    i_coc: bool,
    i_traj: bool,
    adv_token_ids: dict[str, int],
    p_drop: float = 0.3,
) -> dict[str, Any]:
    """Construct a training sequence with advantage conditioning tokens.

    Inserts 3 conditioning tokens (one per level) between the prompt and
    completion. Implements conditioning dropout for classifier-free guidance:
    all-positive completions have a p_drop probability of being treated as
    unconditional (no conditioning tokens).

    Args:
        prompt_ids: Tokenized prompt (with fused history trajectory).
        completion_ids: Tokenized completion (CoC + trajectory tokens).
        i_obs: Binarized observation advantage (True=positive).
        i_coc: Binarized CoC advantage (True=positive).
        i_traj: Binarized trajectory advantage (True=positive).
        adv_token_ids: Dict mapping token name -> ID.
        p_drop: Conditioning dropout probability for all-positive completions.

    Returns:
        Dict with:
        - input_ids: list[int] — full sequence
        - labels: list[int] — IGNORE_INDEX on prompt+conditioning, real IDs on completion
        - attention_mask: list[int] — all 1s
        - is_unconditional: bool — True if this is an unconditional example
    """
    is_all_positive = i_obs and i_coc and i_traj
    is_unconditional = is_all_positive and random.random() < p_drop

    if is_unconditional:
        # Unconditional path: no conditioning tokens
        input_ids = prompt_ids + completion_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    else:
        # Conditional path: insert 3 advantage tokens after prompt
        cond_tokens = [
            adv_token_ids["adv_obs_pos" if i_obs else "adv_obs_neg"],
            adv_token_ids["adv_coc_pos" if i_coc else "adv_coc_neg"],
            adv_token_ids["adv_traj_pos" if i_traj else "adv_traj_neg"],
        ]
        input_ids = prompt_ids + cond_tokens + completion_ids
        labels = [IGNORE_INDEX] * (len(prompt_ids) + len(cond_tokens)) + completion_ids

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "is_unconditional": is_unconditional,
    }


# ---------------------------------------------------------------------------
# 3e. AdvCondDataset
# ---------------------------------------------------------------------------


class AdvCondDataset(Dataset):
    """Dataset wrapping rollout results with advantage conditioning labels.

    Each item returns a training example with conditioning tokens inserted
    and labels masked appropriately for cross-entropy loss.

    Args:
        rollout_results: List of dicts per completion with at least:
            {prompt_ids: list[int], completion_ids: list[int]}.
        adv_labels: List of dicts per completion with:
            {i_obs: bool, i_coc: bool, i_traj: bool}.
        adv_token_ids: Dict mapping token name -> ID.
        p_drop: Conditioning dropout probability.
    """

    def __init__(
        self,
        rollout_results: list[dict],
        adv_labels: list[dict],
        adv_token_ids: dict[str, int],
        p_drop: float = 0.3,
    ) -> None:
        if len(rollout_results) != len(adv_labels):
            raise ValueError(
                f"Mismatched lengths: {len(rollout_results)} rollouts vs {len(adv_labels)} labels"
            )
        self._rollouts = rollout_results
        self._labels = adv_labels
        self._adv_token_ids = adv_token_ids
        self._p_drop = p_drop

    def __len__(self) -> int:
        return len(self._rollouts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rollout = self._rollouts[idx]
        label = self._labels[idx]

        return build_conditioned_sequence(
            prompt_ids=rollout["prompt_ids"],
            completion_ids=rollout["completion_ids"],
            i_obs=label["i_obs"],
            i_coc=label["i_coc"],
            i_traj=label["i_traj"],
            adv_token_ids=self._adv_token_ids,
            p_drop=self._p_drop,
        )

    def set_p_drop(self, p_drop: float) -> None:
        """Update dropout probability (e.g., per epoch)."""
        self._p_drop = p_drop
