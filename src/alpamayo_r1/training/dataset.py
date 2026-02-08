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

"""Dataset builder for GRPO training on PhysicalAI-AV data.

Creates an HF Dataset with conversational prompts and metadata needed
for the custom rollout function to load driving data and run inference.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from datasets import Dataset
from physical_ai_av import PhysicalAIAVDatasetInterface

logger = logging.getLogger(__name__)


def _build_prompt_text(clip_id: str, t0_us: int) -> list[dict[str, Any]]:
    """Build a conversational prompt that encodes clip_id and t0_us.

    The actual images and trajectory history are loaded at rollout time.
    We embed clip_id/t0_us in the system message so the rollout function
    can retrieve the full driving data.

    Args:
        clip_id: Unique clip identifier from the dataset.
        t0_us: Timestamp in microseconds for the prediction origin.

    Returns:
        Chat-format message list suitable for TRL GRPO.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a driving assistant that generates safe and accurate actions. "
                f"[clip_id={clip_id}] [t0_us={t0_us}]"
            ),
        },
        {
            "role": "user",
            "content": (
                "Given the driving scene, output the chain-of-thought reasoning "
                "of the driving process, then output the future trajectory."
            ),
        },
    ]


def build_alpamayo_dataset(
    split: str = "train",
    t0_us: int = 5_100_000,
    max_samples: int | None = None,
    clip_ids_file: str | None = None,
    avdi: PhysicalAIAVDatasetInterface | None = None,
) -> Dataset:
    """Build an HF Dataset from PhysicalAI-AV for GRPO training.

    Each row contains:
    - ``prompt``: Chat-format messages with embedded clip_id/t0_us
    - ``clip_id``: Clip identifier for data loading at rollout time
    - ``t0_us``: Timestamp for prediction origin

    Args:
        split: Dataset split to use ("train", "val", "test").
        t0_us: Default prediction timestamp in microseconds (5.1s into clip).
        max_samples: Optional cap on dataset size for debugging.
        clip_ids_file: Optional path to a parquet file with a ``clip_id`` column.
            If provided, overrides the split-based selection.
        avdi: Pre-initialized dataset interface. Created if None.

    Returns:
        HF Dataset ready for GRPOTrainer.
    """
    if avdi is None:
        avdi = PhysicalAIAVDatasetInterface()

    if clip_ids_file is not None:
        logger.info("Loading clip IDs from %s", clip_ids_file)
        clip_ids_df = pd.read_parquet(clip_ids_file)
        clip_ids = clip_ids_df["clip_id"].tolist()
    else:
        clip_index = avdi.clip_index
        split_df = clip_index[
            (clip_index["split"] == split) & clip_index["clip_is_valid"]
        ]
        clip_ids = split_df.index.tolist()
        logger.info("Found %d valid clips for split '%s'", len(clip_ids), split)

    if max_samples is not None:
        clip_ids = clip_ids[:max_samples]

    records = []
    for clip_id in clip_ids:
        records.append(
            {
                "prompt": _build_prompt_text(clip_id, t0_us),
                "clip_id": clip_id,
                "t0_us": t0_us,
            }
        )

    dataset = Dataset.from_list(records)
    logger.info("Built dataset with %d samples", len(dataset))
    return dataset
