#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract the VLM backbone from a full AlpamayoR1 checkpoint.

vLLM cannot load the full AlpamayoR1 model (custom architecture with
Expert/Diffusion heads). This script extracts just the VLM
(Qwen3VLForConditionalGeneration) so vLLM can serve it directly.

Usage:
    python scripts/extract_vlm.py --model nvidia/Alpamayo-R1-10B --output /tmp/alpamayo_vlm
    python scripts/extract_vlm.py --model /path/to/checkpoint --output /tmp/alpamayo_vlm
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_vlm(model_name: str, output_dir: str, dtype: str = "bfloat16") -> None:
    """Load full AlpamayoR1 and save just the VLM + processor."""
    from alpamayo_r1 import helper
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    torch_dtype = getattr(torch, dtype)
    output_path = Path(output_dir)

    logger.info("Loading full model: %s", model_name)
    full_model = AlpamayoR1.from_pretrained(model_name, dtype=torch_dtype)

    logger.info("Saving VLM to: %s", output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    full_model.vlm.save_pretrained(output_path)

    # Save the processor (tokenizer with trajectory tokens + image processor)
    processor = helper.get_processor(full_model.tokenizer)
    processor.save_pretrained(output_path)

    logger.info("Done. VLM saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract VLM from AlpamayoR1 checkpoint")
    parser.add_argument("--model", required=True, help="Full AlpamayoR1 model name or path")
    parser.add_argument("--output", required=True, help="Output directory for extracted VLM")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype (default: bfloat16)")
    args = parser.parse_args()

    extract_vlm(args.model, args.output, args.dtype)


if __name__ == "__main__":
    main()
