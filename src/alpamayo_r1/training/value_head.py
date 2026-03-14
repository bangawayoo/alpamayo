"""Segment-level value head for GRPO baseline estimation.

SegmentValueHead maps VLM hidden states to scalar value estimates at three
semantic levels: observation (scene), CoC reasoning, and trajectory tokens.
A shared MLP with additive level embeddings predicts V(s) at each level.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SegmentValueHead(nn.Module):
    """Shared MLP with level embedding: (h, level) -> V(s).

    Three levels correspond to the three semantic segments of a VLA generation:
      - Level 0 (obs): last prompt token — scene understanding before generation
      - Level 1 (coc): <cot_end> token — scene + quality of reasoning produced
      - Level 2 (traj): each trajectory token — scene + reasoning + trajectory-so-far

    The level embedding is additive: h' = h + level_embed[level], giving the
    MLP an explicit signal about which stage of generation it's evaluating.

    Args:
        hidden_dim: VLM hidden state dimension (4096 for Qwen3-VL-7B/10B).
        num_levels: Number of distinct levels (default 3).
    """

    LEVEL_OBS = 0
    LEVEL_COC = 1
    LEVEL_TRAJ = 2

    def __init__(self, hidden_dim: int = 4096, num_levels: int = 3) -> None:
        super().__init__()
        self.level_embed = nn.Embedding(num_levels, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, h: torch.Tensor, level: int = 0) -> torch.Tensor:
        """Predict value at one or more sequence positions.

        Args:
            h: Hidden state, shape (B, D) or (B, T, D).
            level: 0=obs, 1=coc, 2=traj. Additive embedding.

        Returns:
            Value estimates, shape (B,) or (B, T).
        """
        h = h + self.level_embed.weight[level]
        return self.net(h).squeeze(-1)


# Backward compatibility alias
SceneValueHead = SegmentValueHead
