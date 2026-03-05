#!/usr/bin/env python3
"""Diagnostic: send test requests to a running vLLM server.

Exercises the same data-loading and tokenization pipeline used by the GRPO
training rollout (``alpamayo_r1.training.rollout``), so a passing test here
means the server will work during training.

Usage (while server is running on port 8000):
    python scripts/test_vllm_server.py --model .cache/vlm_extracted
    python scripts/test_vllm_server.py --model .cache/vlm_extracted --with-images
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from io import BytesIO
from typing import Any

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_tokenizer = None


def _get_tokenizer(model_path: str | None):
    global _tokenizer
    if _tokenizer is None and model_path is not None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return _tokenizer


def _decode_output(data: dict, tokenizer) -> None:
    """Print decoded prompt and completion text for each sample."""
    for i in range(len(data["completion_ids"])):
        comp_ids = data["completion_ids"][i]
        comp_text = (
            tokenizer.decode(comp_ids, skip_special_tokens=False)
            if tokenizer
            else str(comp_ids[:20]) + "..."
        )
        prompt_ids = (
            data.get("prompt_ids", [None])[i]
            if i < len(data.get("prompt_ids", []))
            else None
        )
        prompt_len = len(prompt_ids) if prompt_ids else "?"
        print(f"  [sample {i}] prompt_len={prompt_len}, completion_len={len(comp_ids)}")
        print(f"    completion: {comp_text!r}")


def _image_to_b64(img: Image.Image) -> str:
    """Encode a PIL Image to base64 PNG string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _send_generate(base_url: str, payload: dict, timeout: int = 180) -> dict | None:
    """POST to /generate/ and return parsed JSON, or None on failure."""
    try:
        start = time.time()
        r = requests.post(f"{base_url}/generate/", json=payload, timeout=timeout)
        elapsed = time.time() - start
        if r.status_code == 200:
            print(f"  OK ({elapsed:.1f}s)")
            return r.json()
        else:
            print(f"  FAILED: {r.status_code} — {r.text[:300]}")
            return None
    except requests.exceptions.Timeout:
        print(f"  TIMEOUT (>{timeout}s)")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ---------------------------------------------------------------------------
# Load a real clip using the training pipeline
# ---------------------------------------------------------------------------


def _load_clip_for_server(
    full_model_name: str = "nvidia/Alpamayo-R1-10B",
) -> dict[str, Any]:
    """Load a real driving clip and prepare it exactly like the GRPO rollout.

    Returns dict with:
        prompt_token_ids: list[int]  — full prompt with all 16 images
        pil_images: list[PIL.Image]  — all 16 camera frames
        clip_id: str
        t0_us: int
        full_model: AlpamayoR1      — for building sub-prompts
        processor: AutoProcessor     — for building sub-prompts
        data: dict                   — raw driving data
    """
    import numpy as np
    import torch

    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    # 1. Load the full model (CPU only — we just need tokenizer + fuse_traj_tokens)
    print("  Loading AlpamayoR1 model (CPU)...")
    full_model = AlpamayoR1.from_pretrained(full_model_name, dtype=torch.bfloat16)
    processor = helper.get_processor(full_model.tokenizer)

    # 2. Pick a clip from the dataset
    from physical_ai_av import PhysicalAIAVDatasetInterface

    avdi = PhysicalAIAVDatasetInterface()
    clip_index = avdi.clip_index
    valid_clips = clip_index[
        (clip_index["split"] == "train") & clip_index["clip_is_valid"]
    ]
    clip_id = valid_clips.index[0]
    t0_us = 5_100_000
    print(f"  clip_id={clip_id}, t0_us={t0_us}")

    # 3. Load raw driving data (same as ClipDataCache._load_and_cache)
    data = load_physical_aiavdataset(
        clip_id=clip_id, t0_us=t0_us, avdi=avdi, maybe_stream=True
    )

    # 4. Tokenize via chat template (same as helper.prepare_model_inputs)
    model_inputs = helper.prepare_model_inputs(data, processor, device="cpu")

    # 5. Fuse history trajectory tokens (same as rollout_func)
    tokenized = model_inputs["tokenized_data"]
    input_ids = tokenized.pop("input_ids")
    traj_data = {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }
    input_ids = full_model.fuse_traj_tokens(input_ids, traj_data)
    prompt_token_ids = input_ids[0].cpu().tolist()

    # 6. Extract PIL images (same as ClipDataCache._load_and_cache)
    frames = data["image_frames"].flatten(0, 1)  # (N*F, 3, H, W)
    pil_images = [
        Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
        for frame in frames
    ]

    # 7. Diagnostic: count image placeholder groups in token IDs
    VISION_START = 151652
    VISION_END = 151653
    IMAGE_PAD = 151655
    n_vision_start = prompt_token_ids.count(VISION_START)
    n_vision_end = prompt_token_ids.count(VISION_END)
    n_image_pad = prompt_token_ids.count(IMAGE_PAD)
    print(f"  prompt_token_ids length: {len(prompt_token_ids)}")
    print(f"  num images: {len(pil_images)}")
    print(f"  vision placeholders: {n_vision_start} starts, {n_vision_end} ends, "
          f"{n_image_pad} pad tokens")
    if pil_images:
        print(f"  image size: {pil_images[0].size}, mode={pil_images[0].mode}")
    if n_vision_start != len(pil_images):
        print(f"  WARNING: placeholder count ({n_vision_start}) != image count "
              f"({len(pil_images)})")

    return {
        "prompt_token_ids": prompt_token_ids,
        "pil_images": pil_images,
        "clip_id": clip_id,
        "t0_us": t0_us,
        "full_model": full_model,
        "processor": processor,
        "data": data,
    }


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def check_health(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health/", timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"  Health check failed: {e}")
        return False


def test_text_only(base_url: str, tokenizer=None) -> bool:
    """Send a minimal text-only prompt."""
    print("\n[Test 1] Text-only generate (no images)...")
    payload = {
        "prompts": ["Hello, how are you?"],
        "max_tokens": 10,
        "temperature": 0.1,
    }
    data = _send_generate(base_url, payload, timeout=60)
    if data:
        _decode_output(data, tokenizer)
        return True
    return False


def test_token_ids_only(base_url: str, tokenizer=None) -> bool:
    """Send prompt_token_ids instead of text."""
    print("\n[Test 2] prompt_token_ids (no images)...")
    payload = {
        "prompt_token_ids": [[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13]],
        "max_tokens": 10,
        "temperature": 0.1,
    }
    data = _send_generate(base_url, payload, timeout=60)
    if data:
        _decode_output(data, tokenizer)
        return True
    return False


def _build_prompt_for_n_images(
    clip_data: dict, n_images: int,
) -> tuple[list[int], list[Image.Image]]:
    """Build prompt_token_ids and PIL images for a subset of the clip's frames.

    Re-runs the chat template with only the first ``n_images`` frames so the
    placeholder count matches exactly.
    """
    import torch

    from alpamayo_r1 import helper

    full_model = clip_data["full_model"]
    processor = clip_data["processor"]
    data = clip_data["data"]

    # Take first n_images frames from the flattened (N*F, 3, H, W) tensor
    all_frames = data["image_frames"].flatten(0, 1)
    subset_frames = all_frames[:n_images]

    # Re-tokenize with the subset (chat template inserts correct placeholders)
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

    # Fuse history trajectory tokens
    traj_data = {
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    input_ids = full_model.fuse_traj_tokens(input_ids, traj_data)
    prompt_token_ids = input_ids[0].cpu().tolist()

    import numpy as np

    pil_images = [
        Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
        for frame in subset_frames
    ]

    return prompt_token_ids, pil_images


def test_real_clip_single_image(base_url: str, clip_data: dict, tokenizer=None) -> bool:
    """Send 1 real camera frame through the training rollout path."""
    print(f"\n[Test 3a] Real clip, 1 image (clip={clip_data['clip_id']})...")

    prompt_token_ids, pil_images = _build_prompt_for_n_images(clip_data, 1)
    images_b64 = [_image_to_b64(img) for img in pil_images]
    print(f"  prompt_len={len(prompt_token_ids)}, images={len(pil_images)}, "
          f"size={pil_images[0].size}")

    from alpamayo_r1 import helper

    payload = {
        "prompt_token_ids": [prompt_token_ids],
        "images": [images_b64],
        "mm_processor_kwargs": {
            "min_pixels": helper.MIN_PIXELS,
            "max_pixels": helper.MAX_PIXELS,
        },
        "max_tokens": 20,
        "temperature": 0.6,
    }
    data = _send_generate(base_url, payload, timeout=120)
    if data:
        _decode_output(data, tokenizer)
        return True
    return False


def test_real_clip_four_images(base_url: str, clip_data: dict, tokenizer=None) -> bool:
    """Send 4 real camera frames (1 camera × 4 frames)."""
    print(f"\n[Test 3b] Real clip, 4 images (clip={clip_data['clip_id']})...")

    prompt_token_ids, pil_images = _build_prompt_for_n_images(clip_data, 4)
    images_b64 = [_image_to_b64(img) for img in pil_images]
    print(f"  prompt_len={len(prompt_token_ids)}, images={len(pil_images)}, "
          f"size={pil_images[0].size}")

    from alpamayo_r1 import helper

    payload = {
        "prompt_token_ids": [prompt_token_ids],
        "images": [images_b64],
        "mm_processor_kwargs": {
            "min_pixels": helper.MIN_PIXELS,
            "max_pixels": helper.MAX_PIXELS,
        },
        "max_tokens": 20,
        "temperature": 0.6,
    }
    data = _send_generate(base_url, payload, timeout=180)
    if data:
        _decode_output(data, tokenizer)
        return True
    return False


def test_real_clip_eight_images(base_url: str, clip_data: dict, tokenizer=None) -> bool:
    """Send 8 real camera frames (2 cameras × 4 frames)."""
    print(f"\n[Test 3c] Real clip, 8 images (clip={clip_data['clip_id']})...")

    prompt_token_ids, pil_images = _build_prompt_for_n_images(clip_data, 8)
    images_b64 = [_image_to_b64(img) for img in pil_images]
    print(f"  prompt_len={len(prompt_token_ids)}, images={len(pil_images)}, "
          f"size={pil_images[0].size}")

    from alpamayo_r1 import helper

    payload = {
        "prompt_token_ids": [prompt_token_ids],
        "images": [images_b64],
        "mm_processor_kwargs": {
            "min_pixels": helper.MIN_PIXELS,
            "max_pixels": helper.MAX_PIXELS,
        },
        "max_tokens": 20,
        "temperature": 0.6,
    }
    data = _send_generate(base_url, payload, timeout=180)
    if data:
        _decode_output(data, tokenizer)
        return True
    return False


def test_real_clip_all_images(base_url: str, clip_data: dict, tokenizer=None) -> bool:
    """Send all 16 real camera frames (4 cameras × 4 frames) — full rollout path."""
    n_images = len(clip_data["pil_images"])
    print(f"\n[Test 3d] Real clip, all {n_images} images (clip={clip_data['clip_id']})...")

    images_b64 = [_image_to_b64(img) for img in clip_data["pil_images"]]
    print(f"  prompt_len={len(clip_data['prompt_token_ids'])}, images={n_images}, "
          f"size={clip_data['pil_images'][0].size}")

    from alpamayo_r1 import helper

    payload = {
        "prompt_token_ids": [clip_data["prompt_token_ids"]],
        "images": [images_b64],
        "mm_processor_kwargs": {
            "min_pixels": helper.MIN_PIXELS,
            "max_pixels": helper.MAX_PIXELS,
        },
        "max_tokens": 50,
        "temperature": 0.6,
    }
    data = _send_generate(base_url, payload, timeout=300)
    if data:
        _decode_output(data, tokenizer)
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="localhost")
    parser.add_argument(
        "--with-images",
        action="store_true",
        help="Run real-clip image test (loads a driving clip from PhysicalAI-AV)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path for tokenizer (to decode output tokens). "
        "E.g. .cache/vlm_extracted or nvidia/Alpamayo-R1-10B",
    )
    parser.add_argument(
        "--full-model",
        default="nvidia/Alpamayo-R1-10B",
        help="Full AlpamayoR1 model name/path (for loading clip data). "
        "Only used with --with-images.",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    tokenizer = _get_tokenizer(args.model)

    print(f"Testing vLLM server at {base_url}")
    if tokenizer:
        print(f"Tokenizer: {args.model} (vocab_size={tokenizer.vocab_size})")
    else:
        print("Tokenizer: none (pass --model to decode output text)")
    print("=" * 60)

    # Health check
    print("\n[Health] Checking server...")
    if not check_health(base_url):
        print("Server not reachable. Is it running?")
        sys.exit(1)
    print("  Server is healthy.")

    results = {}
    results["text_only"] = test_text_only(base_url, tokenizer)
    results["token_ids"] = test_token_ids_only(base_url, tokenizer)

    if args.with_images:
        print("\n" + "=" * 60)
        print("Loading real driving clip (same pipeline as GRPO rollout)...")
        print("=" * 60)
        clip_data = _load_clip_for_server(full_model_name=args.full_model)

        # Progressive tests: 1 → 4 → 8 → 16 images
        # Each image adds ~179 tokens after vLLM vision expansion.
        # With max_model_len=4096, the expanded prompt may exceed the limit.
        results["real_1img"] = test_real_clip_single_image(base_url, clip_data, tokenizer)
        if results["real_1img"]:
            results["real_4img"] = test_real_clip_four_images(base_url, clip_data, tokenizer)
        if results.get("real_4img"):
            results["real_8img"] = test_real_clip_eight_images(base_url, clip_data, tokenizer)
        if results.get("real_8img"):
            results["real_all"] = test_real_clip_all_images(base_url, clip_data, tokenizer)

    print("\n" + "=" * 60)
    print("Results:")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
