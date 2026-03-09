#!/usr/bin/env python3
"""Minimal test: call vLLM directly (no TRL, no AlpamayoR1) with a single image.

Tests whether vLLM's native Qwen3-VL backend can handle multimodal inputs.
No CUDA is touched before vLLM starts, avoiding the fork/spawn issue.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/vllm/test_vllm_direct.py
"""
import numpy as np
from PIL import Image

# 1. Build a minimal prompt with 1 image — CPU only, no CUDA, no AlpamayoR1
print("[1/3] Building prompt with 1 dummy image (no CUDA, no model load)...")

# Create a small test image (no dataset needed)
test_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
print(f"  Image: {test_img.size}")

# Use the standard Qwen3-VL chat format with an image content block.
# vLLM's processor will handle tokenization and image placeholder insertion.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": test_img},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]

# 2. Create vLLM (this is where CUDA gets initialized — clean, no prior init)
print("[2/3] Creating vLLM LLM instance...")
from vllm import LLM, SamplingParams

VLM_PATH = ".cache/vlm_extracted"
llm = LLM(
    model=VLM_PATH,
    max_model_len=4096,
    enforce_eager=True,
)

sampling_params = SamplingParams(temperature=0.6, max_tokens=20)

# Apply chat template to get the text prompt
tokenizer = llm.get_tokenizer()
prompt_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
print(f"  Prompt text length: {len(prompt_text)} chars")

print("[3/3] Calling llm.generate() with text prompt + image...")
print("  (If this hangs, the bug is in vLLM's native Qwen3-VL multimodal processing)")

vllm_input = {
    "prompt": prompt_text,
    "multi_modal_data": {"image": test_img},
}

outputs = llm.generate([vllm_input], sampling_params=sampling_params, use_tqdm=True)

print("\nSUCCESS!")
for out in outputs:
    print(f"  Prompt tokens: {len(out.prompt_token_ids)}")
    for o in out.outputs:
        print(f"  Completion tokens: {len(o.token_ids)}")
        text = tokenizer.decode(list(o.token_ids))
        print(f"  Text: {text!r}")
