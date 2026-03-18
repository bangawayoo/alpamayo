"""Segment-level value head for GRPO baseline estimation.

SegmentValueHead maps VLM hidden states to scalar value estimates at two
semantic levels: observation (scene) and trajectory tokens.
A shared MLP with additive level embeddings and rotary positional encoding
predicts V(s) at each level.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SegmentValueHead(nn.Module):
    """Shared MLP with level embedding and rotary positional encoding.

    Two levels correspond to the two semantic segments of a VLA generation:
      - Level 0 (obs): last prompt token — scene understanding before generation
      - Level 1 (traj): each trajectory token — scene + reasoning + trajectory-so-far

    The level embedding is additive: h' = h + level_embed[level], giving the
    MLP an explicit signal about which stage of generation it's evaluating.

    Rotary positional encoding (RoPE) provides a parameter-free sinusoidal
    position signal. Default positions place obs at 0 and trajectory tokens at
    1, 2, ..., T — enabling the MLP to learn that earlier positions have higher
    returns-to-go.

    Args:
        hidden_dim: VLM hidden state dimension (4096 for Qwen3-VL-7B/10B).
        num_levels: Number of distinct levels (default 2).
        rope_base: Base frequency for RoPE (default 10000.0).
    """

    LEVEL_OBS = 0
    LEVEL_TRAJ = 1

    # Default position offsets for each level
    POS_OBS = 0
    POS_TRAJ_START = 1

    def __init__(
        self, hidden_dim: int = 4096, num_levels: int = 2, rope_base: float = 10000.0
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.level_embed = nn.Embedding(num_levels, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # RoPE inverse frequencies — deterministic, not learned
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional encoding to hidden states.

        Args:
            h: Hidden states, shape (..., D). D must be even.
            positions: Position indices, shape matching h's leading dims
                (i.e., all dims of h except the last).

        Returns:
            Rotated hidden states, same shape as h.
        """
        # Compute rotation angles: (..., D//2)
        freqs = torch.einsum("..., d -> ...d", positions.float(), self.inv_freq)
        cos_f = freqs.cos()
        sin_f = freqs.sin()

        # Split into even/odd dimension pairs and rotate
        h1 = h[..., 0::2]
        h2 = h[..., 1::2]
        out = torch.stack(
            [
                h1 * cos_f - h2 * sin_f,
                h1 * sin_f + h2 * cos_f,
            ],
            dim=-1,
        ).flatten(-2)

        return out

    def _default_positions(self, h: torch.Tensor, level: int) -> torch.Tensor:
        """Generate default position indices based on level and input shape.

        Default mapping:
          - obs:  position 0
          - traj: positions 1, 2, ..., T

        Args:
            h: Input hidden state, (B, D) or (B, T, D).
            level: Segment level index.

        Returns:
            Position tensor broadcastable with h's non-feature dims.
        """
        device = h.device
        if level == self.LEVEL_OBS:
            pos_val = self.POS_OBS
        else:
            pos_val = self.POS_TRAJ_START

        if h.dim() == 2:
            # (B, D) — single position per sample
            return torch.full((h.shape[0],), pos_val, device=device)
        elif h.dim() == 3:
            B, T, _ = h.shape
            if level == self.LEVEL_TRAJ:
                # Sequential positions: [POS_TRAJ_START, POS_TRAJ_START+1, ...]
                pos = torch.arange(T, device=device) + self.POS_TRAJ_START
                return pos.unsqueeze(0).expand(B, -1)
            else:
                return torch.full((B, T), pos_val, device=device)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {h.dim()}D")

    def forward(
        self,
        h: torch.Tensor,
        level: int = 0,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict value at one or more sequence positions.

        Args:
            h: Hidden state, shape (B, D) or (B, T, D).
            level: 0=obs, 1=traj. Additive embedding.
            positions: Position indices for RoPE encoding. Shape (B,) for
                2D h or (B, T) for 3D h. If None, uses default positions
                based on level (obs=0, coc=1, traj=2..T+1).

        Returns:
            Value estimates, shape (B,) or (B, T).
        """
        h = h + self.level_embed.weight[level]

        if positions is None:
            positions = self._default_positions(h, level)
        h = self._apply_rope(h, positions)

        return self.net(h).squeeze(-1)


# Backward compatibility alias
SceneValueHead = SegmentValueHead
