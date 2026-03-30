"""Segment-level value head for GRPO baseline estimation.

SegmentValueHead maps a VLM hidden state to a scalar value estimate at the
observation (scene) level. A shared MLP with additive level embedding and
rotary positional encoding predicts V(s_obs).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SegmentValueHead(nn.Module):
    """MLP with level embedding and rotary positional encoding.

    Predicts V(s_obs) from the last-prompt-token hidden state. The level
    embedding is additive: h' = h + level_embed[level].

    Args:
        hidden_dim: VLM hidden state dimension (4096 for Qwen3-VL-7B/10B).
        num_levels: Number of distinct levels (default 1, obs-only).
        rope_base: Base frequency for RoPE (default 10000.0).
    """

    LEVEL_OBS = 0

    # Default position offset
    POS_OBS = 0

    def __init__(
        self, hidden_dim: int = 4096, num_levels: int = 1, rope_base: float = 10000.0
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
        """Generate default position indices (always POS_OBS=0).

        Args:
            h: Input hidden state, (B, D) or (B, T, D).
            level: Segment level index (unused, kept for API compat).

        Returns:
            Position tensor broadcastable with h's non-feature dims.
        """
        device = h.device
        if h.dim() == 2:
            return torch.full((h.shape[0],), self.POS_OBS, device=device)
        elif h.dim() == 3:
            B, T, _ = h.shape
            return torch.full((B, T), self.POS_OBS, device=device)
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
            level: Level index (default 0=obs). Additive embedding.
            positions: Position indices for RoPE encoding. Shape (B,) for
                2D h or (B, T) for 3D h. If None, uses default position 0.

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
