"""Segment-level value head for GRPO baseline estimation.

SegmentValueHead maps VLM hidden states at sequence positions to scalar
expected-reward estimates, providing a learned baseline for advantage
computation.  Currently used at the scene level (h_obs); designed to
accept batched token-level hidden states for future segment-level
advantages (CoC, per-trajectory-token).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SegmentValueHead(nn.Module):
    """Shared MLP: h → V(s) at any sequence position.

    Accepts both single-position hidden states ``(B, D)`` and
    multi-position hidden states ``(B, T, D)``, returning ``(B,)``
    or ``(B, T)`` respectively.

    Args:
        hidden_dim: VLM hidden state dimension (4096 for Qwen3-VL-7B/10B).
    """

    def __init__(self, hidden_dim: int = 4096) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict value at one or more sequence positions.

        Args:
            h: Hidden state tensor, shape ``(B, D)`` or ``(B, T, D)``.

        Returns:
            Value estimates, shape ``(B,)`` or ``(B, T)``.
        """
        return self.net(h).squeeze(-1)


# Backward-compatible alias
SceneValueHead = SegmentValueHead
