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

"""Rule-based per-timestep meta-action extractor from ego-centric trajectories.

Classifies each timestep of a trajectory into one longitudinal and one lateral
atomic meta-action using the 13-label taxonomy from docs/meta-action.md.

Meta-actions are instantaneous (10Hz), not scene-level summaries.  A trajectory
is converted to a *sequence* of (longitudinal, lateral) label pairs.  For
reward computation, a summary (mode) is also provided.

Trajectory format: (T, 3) cumulative ego-centric positions,
X = forward, Y = left, Z = vertical, dt = 0.1s.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

# ---------------------------------------------------------------------------
# Meta-action string constants (7 longitudinal + 7 lateral)
# ---------------------------------------------------------------------------

LON_GENTLE_ACCEL = "gentle_accelerate"
LON_STRONG_ACCEL = "strong_accelerate"
LON_GENTLE_DECEL = "gentle_decelerate"
LON_STRONG_DECEL = "strong_decelerate"
LON_MAINTAIN = "maintain_speed"
LON_REVERSE = "reverse"
LON_STOP = "stop"

LAT_GO_STRAIGHT = "go_straight"
LAT_STEER_LEFT = "steer_left"
LAT_SHARP_LEFT = "sharp_steer_left"
LAT_STEER_RIGHT = "steer_right"
LAT_SHARP_RIGHT = "sharp_steer_right"
LAT_REVERSE_LEFT = "reverse_left"
LAT_REVERSE_RIGHT = "reverse_right"

# ---------------------------------------------------------------------------
# Threshold constants (same values, now applied per-timestep)
# ---------------------------------------------------------------------------

# Longitudinal (applied to instantaneous velocity / acceleration)
_STOP_SPEED_THRESHOLD = 0.5  # m/s — near standstill
_REVERSE_SPEED_THRESHOLD = -0.3  # m/s — negative longitudinal velocity
_GENTLE_ACCEL_THRESHOLD = 0.5  # m/s²
_STRONG_ACCEL_THRESHOLD = 2.0  # m/s²
_GENTLE_DECEL_THRESHOLD = -0.5  # m/s²
_STRONG_DECEL_THRESHOLD = -2.0  # m/s²

# Lateral (applied to instantaneous lateral rate)
_STRAIGHT_RATE_THRESHOLD = 0.3  # m/s — below this = go_straight
_STEER_RATE_THRESHOLD = 1.0  # m/s — above this = sharp steer

# Smoothing kernel size for finite-difference noise reduction
_SMOOTH_KERNEL = 5


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetaActions:
    """Per-timestep meta-action sequences for a trajectory.

    Each list has length T-2 (aligned with acceleration, which requires two
    finite differences). For a 64-step trajectory this gives 62 labels.
    """

    longitudinal: list[str]
    lateral: list[str]


@dataclass
class MetaActionsSummary:
    """Dominant (mode) meta-actions for a trajectory — one label per axis."""

    longitudinal: str
    lateral: str


# ---------------------------------------------------------------------------
# Core per-timestep extractor
# ---------------------------------------------------------------------------

_DT = 0.1  # seconds per timestep


def extract_meta_actions(traj: np.ndarray) -> MetaActions:
    """Extract per-timestep longitudinal and lateral meta-actions.

    Args:
        traj: Array of shape (T, 3), cumulative ego-centric positions.

    Returns:
        MetaActions with per-timestep label lists (length T-2).
    """
    T = traj.shape[0]
    if T < 3:
        return MetaActions([LON_STOP], [LAT_GO_STRAIGHT])

    # --- Kinematic quantities ---
    velocity = np.diff(traj[:, 0]) / _DT  # (T-1,) m/s longitudinal
    accel = np.diff(velocity) / _DT  # (T-2,) m/s²
    lateral_rate = np.diff(traj[:, 1]) / _DT  # (T-1,) m/s

    # Smooth to reduce finite-difference noise from discrete tokenization
    if len(velocity) >= _SMOOTH_KERNEL:
        velocity = median_filter(velocity, size=_SMOOTH_KERNEL, mode="nearest")
    if len(accel) >= _SMOOTH_KERNEL:
        accel = median_filter(accel, size=_SMOOTH_KERNEL, mode="nearest")
    if len(lateral_rate) >= _SMOOTH_KERNEL:
        lateral_rate = median_filter(lateral_rate, size=_SMOOTH_KERNEL, mode="nearest")

    # Align to length of accel (T-2): velocity and lateral_rate are (T-1),
    # so we take [1:] to align with accel indices (centered differencing)
    vel_aligned = velocity[1:]  # (T-2,)
    lat_aligned = lateral_rate[1:]  # (T-2,)
    N = len(accel)

    lon_labels: list[str] = []
    lat_labels: list[str] = []

    for i in range(N):
        v = float(vel_aligned[i])
        a = float(accel[i])
        lr = float(lat_aligned[i])

        # --- Longitudinal (first match wins) ---
        if v < _REVERSE_SPEED_THRESHOLD:
            lon = LON_REVERSE
        elif v < _STOP_SPEED_THRESHOLD:
            lon = LON_STOP
        elif a > _STRONG_ACCEL_THRESHOLD:
            lon = LON_STRONG_ACCEL
        elif a > _GENTLE_ACCEL_THRESHOLD:
            lon = LON_GENTLE_ACCEL
        elif a < _STRONG_DECEL_THRESHOLD:
            lon = LON_STRONG_DECEL
        elif a < _GENTLE_DECEL_THRESHOLD:
            lon = LON_GENTLE_DECEL
        else:
            lon = LON_MAINTAIN

        # --- Lateral (first match wins) ---
        is_reversing = lon == LON_REVERSE
        if is_reversing and lr > _STRAIGHT_RATE_THRESHOLD:
            lat = LAT_REVERSE_LEFT
        elif is_reversing and lr < -_STRAIGHT_RATE_THRESHOLD:
            lat = LAT_REVERSE_RIGHT
        elif lr > _STEER_RATE_THRESHOLD:
            lat = LAT_SHARP_LEFT
        elif lr < -_STEER_RATE_THRESHOLD:
            lat = LAT_SHARP_RIGHT
        elif lr > _STRAIGHT_RATE_THRESHOLD:
            lat = LAT_STEER_LEFT
        elif lr < -_STRAIGHT_RATE_THRESHOLD:
            lat = LAT_STEER_RIGHT
        else:
            lat = LAT_GO_STRAIGHT

        lon_labels.append(lon)
        lat_labels.append(lat)

    return MetaActions(lon_labels, lat_labels)


# ---------------------------------------------------------------------------
# Summary extractor (dominant label = mode)
# ---------------------------------------------------------------------------


def extract_meta_actions_summary(traj: np.ndarray) -> MetaActionsSummary:
    """Extract the dominant meta-action per axis (mode of per-timestep labels).

    Args:
        traj: Array of shape (T, 3), cumulative ego-centric positions.

    Returns:
        MetaActionsSummary with one longitudinal and one lateral label.
    """
    meta = extract_meta_actions(traj)
    lon_mode = Counter(meta.longitudinal).most_common(1)[0][0]
    lat_mode = Counter(meta.lateral).most_common(1)[0][0]
    return MetaActionsSummary(lon_mode, lat_mode)


# ---------------------------------------------------------------------------
# Convenience wrapper for flattened trajectory lists (used by rewards.py)
# ---------------------------------------------------------------------------


def _reshape_pred_flat(pred_flat: list[float], num_steps: int = 64) -> np.ndarray:
    """Reshape a flattened trajectory list to (T, 3), taking first sample if multi-sample."""
    arr = np.array(pred_flat, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    if arr.shape[0] > num_steps:
        arr = arr[:num_steps]
    return arr


def trajectory_to_meta_actions(
    pred_flat: list[float], num_steps: int = 64,
) -> MetaActionsSummary | None:
    """Convert a flattened trajectory list to summary (mode) meta-actions.

    If the flat list contains multiple trajectory samples (S*T*3 elements),
    only the first sample (T steps) is used.  Returns None on parse failure.

    Args:
        pred_flat: Flattened trajectory, either (T*3,) or (S*T*3,).
        num_steps: Expected number of timesteps per trajectory (default: 64).
    """
    try:
        return extract_meta_actions_summary(_reshape_pred_flat(pred_flat, num_steps))
    except Exception:
        return None


def trajectory_to_meta_action_sets(
    pred_flat: list[float], num_steps: int = 64, min_fraction: float = 0.0,
) -> tuple[set[str], set[str]] | None:
    """Convert a flattened trajectory to the sets of meta-actions present.

    Only actions that occupy at least ``min_fraction`` of all timesteps are
    included.  With ``min_fraction=0`` (default) every action that appears
    at least once is returned.

    Returns:
        (lon_set, lat_set) or None on parse failure.
    """
    try:
        meta = extract_meta_actions(_reshape_pred_flat(pred_flat, num_steps))
        n = len(meta.longitudinal)
        if n == 0:
            return set(), set()

        lon_counts = Counter(meta.longitudinal)
        lat_counts = Counter(meta.lateral)
        threshold = n * min_fraction

        lon_set = {a for a, c in lon_counts.items() if c >= threshold}
        lat_set = {a for a, c in lat_counts.items() if c >= threshold}
        return lon_set, lat_set
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Keyword mapping for CoC text matching
# ---------------------------------------------------------------------------

_META_ACTION_KEYWORDS: dict[str, list[str]] = {
    # Longitudinal
    LON_GENTLE_ACCEL: [
        "gently accelerat",
        "gradually speed up",
        "mild acceleration",
        "slight acceleration",
        "slowly accelerat",
        "gentle acceleration",
        "accelerat",
        "speed up",
        "resume speed",
    ],
    LON_STRONG_ACCEL: [
        "rapidly accelerat",
        "hard acceleration",
        "strong acceleration",
        "aggressive acceleration",
        "quickly accelerat",
        "sharp acceleration",
    ],
    LON_GENTLE_DECEL: [
        "gently decelerat",
        "gradually slow",
        "mild deceleration",
        "slight deceleration",
        "ease off",
        "gentle deceleration",
        "lightly brak",
        "slow down",
        "slowing down",
        "reduce speed",
        "adapt speed",
        "adjust speed",
        "keep distance",
        "decelerat",
        "yield",
    ],
    LON_STRONG_DECEL: [
        "hard brak",
        "heavy brak",
        "strong deceleration",
        "emergency brak",
        "slam.*brak",
        "aggressive deceleration",
        "rapidly decelerat",
    ],
    LON_MAINTAIN: [
        "maintain speed",
        "constant speed",
        "cruise",
        "keep speed",
        "steady speed",
        "same speed",
        "maintain velocity",
        "continue driving",
        "keep driving",
        "keep going",
        "driving forward",
        "keep distance",
    ],
    LON_REVERSE: [
        "reverse",
        "backing up",
        "back up",
        "moving backward",
        "drive backward",
    ],
    LON_STOP: [
        "stop",
        "halt",
        "come to a stop",
        "standstill",
        "stationary",
        "pulled over",
        "wait",
        "idle",
    ],
    # Lateral
    LAT_GO_STRAIGHT: [
        "go straight",
        "straight ahead",
        "maintain lane",
        "stay in lane",
        "continue forward",
        "keep lane",
        "drive straight",
    ],
    LAT_STEER_LEFT: [
        "steer left",
        "turn left",
        "turning left",
        "left turn",
        "veer left",
        "bear left",
        "move left",
        "shift left",
        "drift left",
        "nudge to the left",
        "nudge left",
        "left curve",
    ],
    LAT_SHARP_LEFT: [
        "sharp left",
        "hard left",
        "sharp turn left",
        "sharp steer left",
        "abrupt left",
        "turn left",
        "turning left",
        "left turn",
        "lane change left",
        "change lane left",
        "change to the left",
        "nudge to the left",
        "nudge left",
        "left curve",
    ],
    LAT_STEER_RIGHT: [
        "steer right",
        "turn right",
        "turning right",
        "right turn",
        "veer right",
        "bear right",
        "move right",
        "shift right",
        "drift right",
        "nudge to the right",
        "nudge right",
        "right curve",
    ],
    LAT_SHARP_RIGHT: [
        "sharp right",
        "hard right",
        "sharp turn right",
        "sharp steer right",
        "abrupt right",
        "turn right",
        "turning right",
        "right turn",
        "lane change right",
        "change lane right",
        "change to the right",
        "nudge to the right",
        "nudge right",
        "right curve",
    ],
    LAT_REVERSE_LEFT: [
        "reverse left",
        "back left",
        "backing left",
    ],
    LAT_REVERSE_RIGHT: [
        "reverse right",
        "back right",
        "backing right",
    ],
}
