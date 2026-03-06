#!/usr/bin/env python3
"""Phase 1: Build prompt using the full AlpamayoR1 pipeline (CPU only).

Loads AlpamayoR1 model, tokenizes with the processor, fuses trajectory tokens,
strips image pads, extracts PIL images, and saves everything to a pickle file.

This script must run in a SEPARATE process from vLLM to avoid CUDA conflicts.

Usage:
    python scripts/test_vllm_build_prompt.py [--output PATH] [--n-images N]
"""
from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "src")


def build_prompt(n_images: int = 1) -> dict:
    """Build a vLLM-ready prompt dict using the full AlpamayoR1 pipeline.

    Follows the exact same flow as the inference notebook:
      1. AlpamayoR1.from_pretrained (CPU)
      2. helper.get_processor / helper.create_message
      3. processor.apply_chat_template (tokenize=True)
      4. full_model.fuse_traj_tokens
      5. Strip <|image_pad|> tokens (vLLM re-inserts them)
      6. Extract PIL images
    """
    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from physical_ai_av import PhysicalAIAVDatasetInterface

    # 1. Load model on CPU
    print("  Loading AlpamayoR1 model (CPU)...")
    full_model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16
    )
    processor = helper.get_processor(full_model.tokenizer)

    # 2. Pick a clip
    avdi = PhysicalAIAVDatasetInterface()
    clip_index = avdi.clip_index
    valid_clips = clip_index[
        (clip_index["split"] == "train") & clip_index["clip_is_valid"]
    ]
    clip_id = valid_clips.index[0]
    t0_us = 5_100_000
    print(f"  clip_id={clip_id}, t0_us={t0_us}")

    # 3. Load driving data
    data = load_physical_aiavdataset(
        clip_id=clip_id, t0_us=t0_us, avdi=avdi, maybe_stream=True
    )

    # 4. Select subset of frames
    all_frames = data["image_frames"].flatten(0, 1)
    subset_frames = all_frames[:n_images]

    # 5. Tokenize (same as inference notebook)
    messages = helper.create_message(subset_frames)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]

    # 6. Fuse trajectory tokens
    traj_data = {
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    input_ids = full_model.fuse_traj_tokens(input_ids, traj_data)
    prompt_token_ids = input_ids[0].cpu().tolist()

    # 7. Collapse image pads: keep 1 per image (vLLM needs a marker to expand)
    IMAGE_PAD = 151655
    VISION_START = 151652
    VISION_END = 151653
    n_pads = prompt_token_ids.count(IMAGE_PAD)

    collapsed = []
    in_vision = False
    pad_emitted = False
    for t in prompt_token_ids:
        if t == VISION_START:
            in_vision = True
            pad_emitted = False
            collapsed.append(t)
        elif t == VISION_END:
            in_vision = False
            collapsed.append(t)
        elif t == IMAGE_PAD and in_vision:
            if not pad_emitted:
                collapsed.append(t)  # keep first pad
                pad_emitted = True
            # skip subsequent pads
        else:
            collapsed.append(t)

    prompt_token_ids_stripped = collapsed
    print(f"  Token IDs: {len(prompt_token_ids)} (with {n_pads} image pads)")
    print(f"  Collapsed: {len(prompt_token_ids_stripped)} tokens "
          f"(1 image pad kept per image)")

    # 8. Extract PIL images
    pil_images = [
        Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
        for frame in subset_frames
    ]
    print(f"  Images: {len(pil_images)}, size={pil_images[0].size}")

    return {
        "prompt_token_ids": prompt_token_ids_stripped,
        "prompt_token_ids_with_pads": prompt_token_ids,
        "pil_images": pil_images,
        "clip_id": clip_id,
        "t0_us": t0_us,
        "n_images": n_images,
        "min_pixels": helper.MIN_PIXELS,
        "max_pixels": helper.MAX_PIXELS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default=".cache/test_vllm_prompt.pkl",
        help="Output pickle path (default: .cache/test_vllm_prompt.pkl)",
    )
    parser.add_argument(
        "--n-images", type=int, default=1,
        help="Number of images to include (default: 1)",
    )
    args = parser.parse_args()

    print(f"Building prompt with {args.n_images} image(s)...")
    result = build_prompt(n_images=args.n_images)

    with open(args.output, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
