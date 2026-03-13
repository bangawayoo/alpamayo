# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO post-training.

Three reward signals:
1. trajectory_quality_reward  – L2 displacement-based trajectory quality
2. reasoning_quality_reward   – rule-based chain-of-causation quality
3. consistency_reward         – agreement between CoC text and predicted trajectory
"""

from __future__ import annotations

import re

import numpy as np

from alpamayo_r1.training.meta_actions import (
    _META_ACTION_KEYWORDS,
    trajectory_to_meta_action_sets,
)


# ---------------------------------------------------------------------------
# 1. Trajectory quality reward
# ---------------------------------------------------------------------------

def trajectory_quality_reward(
    completions: list[str],
    pred_xyz: list[list[float]] | None = None,
    gt_xyz: list[list[float]] | None = None,
    **kwargs,
) -> list[float]:
    """Reward based on L2 displacement between predicted and ground-truth trajectories.

    Each GRPO rollout produces a single independent trajectory. The reward is
    computed as the mean per-timestep L2 distance (ADE) between that trajectory
    and the ground truth, mapped to [0, 1] via a soft threshold.

    Args:
        completions: Generated CoC text (unused for trajectory scoring but
            required by the TRL reward function interface).
        pred_xyz: Per-sample predicted trajectories as flattened lists.
            Original shape per sample: (T, 3).
        gt_xyz: Per-sample ground-truth trajectories as flattened lists.
            Original shape per sample: (T, 3).
        **kwargs: Additional fields forwarded by TRL (ignored).

    Returns:
        List of float rewards in [0, 1], one per completion.
    """
    ade_threshold: float = kwargs.get("ade_threshold", 5.0)

    if pred_xyz is None or gt_xyz is None:
        return [0.0] * len(completions)

    rewards: list[float] = []
    for pred_flat, gt_flat in zip(pred_xyz, gt_xyz):
        try:
            pred = np.array(pred_flat, dtype=np.float32)
            gt = np.array(gt_flat, dtype=np.float32)

            # Reshape to (T, 3)
            if gt.ndim == 1:
                gt = gt.reshape(-1, 3)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 3)

            # Per-timestep L2 distance on xy plane, then average
            pred_xy = pred[:, :2]
            gt_xy = gt[:, :2]
            l2_per_step = np.linalg.norm(pred_xy - gt_xy, axis=-1)  # (T,)
            ade = float(l2_per_step.mean())

            reward = max(0.0, 1.0 - ade / ade_threshold)
            rewards.append(reward)
        except Exception:
            rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# 2. Reasoning quality reward
# ---------------------------------------------------------------------------

# Patterns for rule-based scoring
_CAUSAL_CONNECTORS = re.compile(
    r"\b(because|therefore|since|thus|hence|as a result|so that|due to|"
    r"in order to|consequently|leads to|causing|results in)\b",
    re.IGNORECASE,
)
_DRIVING_TERMS = re.compile(
    r"\b(vehicle|car|truck|pedestrian|cyclist|lane|intersection|traffic|"
    r"signal|light|stop|yield|merge|speed|brake|accelerat|steer|turn|"
    r"left|right|straight|ahead|behind|front|rear|lateral|oncoming|"
    r"highway|road|path|obstacle|distance|gap|follow|approach|slow|fast)\b",
    re.IGNORECASE,
)
_REPETITION_PATTERN = re.compile(r"(.{20,}?)\1{2,}")

# Length bounds (in characters)
_MIN_LENGTH = 40
_MAX_LENGTH = 2000


def reasoning_quality_reward(
    completions: list[str],
    **kwargs,
) -> list[float]:
    """Rule-based reward for chain-of-causation reasoning quality.

    Scores completions on four criteria:
    - Presence of causal connectors (because, therefore, ...)
    - Driving-domain vocabulary usage
    - Appropriate length (not too short / too long)
    - Absence of degenerate repetition

    Args:
        completions: Generated CoC reasoning strings.
        **kwargs: Ignored.

    Returns:
        List of float rewards in [0, 1], one per completion.
    """
    rewards: list[float] = []
    for text in completions:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        score = 0.0
        total_criteria = 4.0

        # 1. Causal connectors
        n_causal = len(_CAUSAL_CONNECTORS.findall(text))
        if n_causal >= 2:
            score += 1.0
        elif n_causal == 1:
            score += 0.5

        # 2. Driving-domain terms
        n_driving = len(_DRIVING_TERMS.findall(text))
        if n_driving >= 4:
            score += 1.0
        elif n_driving >= 2:
            score += 0.5

        # 3. Appropriate length
        length = len(text)
        if _MIN_LENGTH <= length <= _MAX_LENGTH:
            score += 1.0
        elif length > 0:
            score += 0.25

        # 4. No degenerate repetition
        if not _REPETITION_PATTERN.search(text):
            score += 1.0

        rewards.append(score / total_criteria)

    return rewards


# ---------------------------------------------------------------------------
# 3. Consistency reward
# ---------------------------------------------------------------------------

# Deprecated: use meta_actions module instead. Kept for backward compatibility with tests.
_BEHAVIOR_KEYWORDS = {
    "turning_left": ["turn left", "turning left", "left turn", "veer left"],
    "turning_right": ["turn right", "turning right", "right turn", "veer right"],
    "going_straight": ["straight", "continue ahead", "go straight", "maintain lane"],
    "accelerating": ["accelerat", "speed up", "faster", "increase speed"],
    "decelerating": ["decelerat", "slow down", "brake", "braking", "reduce speed"],
    "stopping": ["stop", "halt", "come to a stop", "standstill"],
}


# Deprecated: use meta_actions.trajectory_to_meta_actions instead.
def _trajectory_to_behaviors(pred_flat: list[float]) -> set[str]:
    """Infer coarse driving behaviors from a predicted trajectory.

    Analyzes the trajectory's lateral displacement (turning) and longitudinal
    velocity changes (acceleration/braking) to produce behavior labels.

    Args:
        pred_flat: Flattened trajectory, reshaped to (num_samples, T, 3).

    Returns:
        Set of behavior keys (e.g. {"turning_left", "accelerating"}).
    """
    behaviors: set[str] = set()
    try:
        pred = np.array(pred_flat, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 3)
        if pred.ndim == 3:
            # Use the first sample trajectory
            pred = pred[0]

        if pred.shape[0] < 3:
            return behaviors

        # Lateral displacement (y-axis in ego frame)
        lateral_displacement = pred[-1, 1] - pred[0, 1]
        if lateral_displacement > 1.0:
            behaviors.add("turning_left")
        elif lateral_displacement < -1.0:
            behaviors.add("turning_right")
        else:
            behaviors.add("going_straight")

        # Longitudinal velocity change (x-axis in ego frame)
        dx = np.diff(pred[:, 0])
        speed_start = abs(float(dx[:3].mean())) if len(dx) >= 3 else 0.0
        speed_end = abs(float(dx[-3:].mean())) if len(dx) >= 3 else 0.0

        if speed_end < 0.05:
            behaviors.add("stopping")
        elif speed_end > speed_start * 1.2:
            behaviors.add("accelerating")
        elif speed_end < speed_start * 0.8:
            behaviors.add("decelerating")
    except Exception:
        pass

    return behaviors


def consistency_reward(
    completions: list[str],
    pred_xyz: list[list[float]] | None = None,
    **kwargs,
) -> list[float]:
    """Reward for consistency between CoC text and predicted trajectory.

    Converts the trajectory into coarse behavior descriptions (turning, braking,
    etc.) and checks whether the CoC text mentions corresponding keywords.

    Args:
        completions: Generated CoC reasoning strings.
        pred_xyz: Per-sample predicted trajectories as flattened lists.
        **kwargs: Ignored.

    Returns:
        List of float rewards in [0, 1], one per completion.
    """
    if pred_xyz is None:
        return [0.0] * len(completions)

    rewards: list[float] = []
    for text, pred_flat in zip(completions, pred_xyz):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        result = trajectory_to_meta_action_sets(pred_flat, min_fraction=0.25)
        if result is None:
            rewards.append(0.0)
            continue

        lon_set, lat_set = result
        text_lower = text.lower()

        # Match if any meta-action (≥25% of timesteps) has a keyword in the text
        lon_match = any(
            kw in text_lower
            for action in lon_set
            for kw in _META_ACTION_KEYWORDS.get(action, [])
        )
        lat_match = any(
            kw in text_lower
            for action in lat_set
            for kw in _META_ACTION_KEYWORDS.get(action, [])
        )

        # Going straight is the implicit default — if the trajectory only
        # goes straight laterally, not mentioning it in the CoC text is
        # consistent (humans omit redundant lateral descriptions).
        if lat_set == {"go_straight"}:
            lat_match = True

        # Binary: both axes must match for reward (sharper GRPO signal)
        if lon_match and lat_match:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards
