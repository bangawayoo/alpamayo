#!/usr/bin/env python3
"""Phase 2: Call vLLM directly with the preprocessed prompt from Phase 1.

Loads the pickle produced by test_vllm_build_prompt.py and calls
vLLM's llm.generate() with the native Qwen3-VL backend.

Usage:
    # Phase 1 (run once, or when changing n_images):
    python scripts/vllm/test_vllm_build_prompt.py --n-images 1

    # Phase 2 (run on GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/vllm/test_vllm_generate.py
"""
from __future__ import annotations

import argparse
import pickle
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=".cache/test_vllm_prompt.pkl",
        help="Pickle from test_vllm_build_prompt.py",
    )
    parser.add_argument(
        "--model", default=".cache/vlm_extracted",
        help="Path to extracted VLM",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    # 1. Load preprocessed prompt (no CUDA touched yet)
    print(f"[1/3] Loading prompt from {args.input}...")
    with open(args.input, "rb") as f:
        data = pickle.load(f)

    prompt_token_ids = data["prompt_token_ids"]
    pil_images = data["pil_images"]
    min_pixels = data["min_pixels"]
    max_pixels = data["max_pixels"]

    print(f"  Token IDs: {len(prompt_token_ids)}")
    print(f"  Images: {len(pil_images)}, size={pil_images[0].size}")
    print(f"  mm_processor_kwargs: min_pixels={min_pixels}, max_pixels={max_pixels}")

    # 2. Create vLLM (CUDA init happens here — clean state)
    print(f"[2/3] Creating vLLM LLM instance ({args.model})...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # 3. Call generate
    print("[3/3] Calling llm.generate()...")
    print(f"  prompt_token_ids: {len(prompt_token_ids)} tokens, first 10: {prompt_token_ids[:10]}")
    print(f"  multi_modal_data: {len(pil_images)} image(s)")
    print("  (If this hangs, the issue is in vLLM's native multimodal preprocessing)")

    vllm_input = {
        "prompt_token_ids": prompt_token_ids,
        "multi_modal_data": {
            "image": pil_images[0] if len(pil_images) == 1 else pil_images,
        },
        "mm_processor_kwargs": {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
    }

    outputs = llm.generate([vllm_input], sampling_params=sampling_params, use_tqdm=True)

    print("\nSUCCESS!")
    tokenizer = llm.get_tokenizer()
    for out in outputs:
        print(f"  Prompt tokens (after vLLM expansion): {len(out.prompt_token_ids)}")
        for i, o in enumerate(out.outputs):
            print(f"  Completion [{i}]: {len(o.token_ids)} tokens")
            print(f"    IDs: {list(o.token_ids)[:30]}")
            text = tokenizer.decode(list(o.token_ids), skip_special_tokens=False)
            print(f"    Text: {text!r}")


if __name__ == "__main__":
    main()
