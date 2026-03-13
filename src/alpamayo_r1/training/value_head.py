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

"""Scene-level value head for GRPO baseline estimation.

SceneValueHead maps the VLM's last-prompt hidden state to a scalar
expected reward, providing a learned baseline for advantage computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SceneValueHead(nn.Module):
    """3-layer MLP: h_0 → E[composite_reward | scene].

    Input h_0 is the VLM's last hidden state at the final prompt token,
    encoding the model's scene understanding before generation begins.

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

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        """Predict scene value.

        Args:
            h0: Hidden state tensor, shape (B, hidden_dim).

        Returns:
            Scalar value estimates, shape (B,).
        """
        return self.net(h0).squeeze(-1)
