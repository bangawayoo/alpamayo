# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke test for the GRPO training module.

Loads the full model, runs 1 rollout step with a real clip, computes rewards,
and verifies the full pipeline works end-to-end.

Usage:
    python tests/test_training_gpu.py
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np
import torch
from physical_ai_av import PhysicalAIAVDatasetInterface

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.training.dataset import build_alpamayo_dataset, _build_prompt_text
from alpamayo_r1.training.rewards import (
    consistency_reward,
    reasoning_quality_reward,
    trajectory_quality_reward,
)
from alpamayo_r1.training.rollout import make_rollout_func, _parse_clip_metadata


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    device = "cuda"
    model_name = "nvidia/Alpamayo-R1-10B"
    num_traj_samples = 2  # small for speed

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print_section("1. Loading model")
    t0 = time.time()
    model = AlpamayoR1.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    avdi = PhysicalAIAVDatasetInterface()
    print(f"   Model loaded in {time.time() - t0:.1f}s")

    mem = torch.cuda.memory_allocated() / 1e9
    print(f"   GPU memory used: {mem:.1f} GB")

    # ------------------------------------------------------------------
    # 2. Get a test clip
    # ------------------------------------------------------------------
    print_section("2. Getting test clip")
    clip_index = avdi.clip_index
    valid_clips = clip_index[clip_index["clip_is_valid"]].index.tolist()
    clip_id = valid_clips[0]
    t0_us = 5_100_000
    print(f"   Using clip: {clip_id}")

    # ------------------------------------------------------------------
    # 3. Test dataset builder
    # ------------------------------------------------------------------
    print_section("3. Testing dataset builder")
    prompt = _build_prompt_text(clip_id, t0_us)
    print(f"   Prompt has {len(prompt)} messages")
    print(f"   System: {prompt[0]['content'][:80]}...")

    # Verify roundtrip
    prompt_str = " ".join(m["content"] for m in prompt)
    parsed_clip, parsed_t0 = _parse_clip_metadata(prompt_str)
    assert parsed_clip == clip_id, f"Roundtrip failed: {parsed_clip} != {clip_id}"
    assert parsed_t0 == t0_us
    print("   Metadata roundtrip: OK")

    # ------------------------------------------------------------------
    # 4. Test prepare_model_inputs
    # ------------------------------------------------------------------
    print_section("4. Testing prepare_model_inputs")
    data = load_physical_aiavdataset(clip_id=clip_id, t0_us=t0_us, avdi=avdi)
    model_inputs = helper.prepare_model_inputs(data, processor, device)
    print(f"   tokenized_data keys: {list(model_inputs['tokenized_data'].keys())}")
    print(f"   input_ids shape: {model_inputs['tokenized_data']['input_ids'].shape}")
    print(f"   ego_history_xyz shape: {model_inputs['ego_history_xyz'].shape}")

    # ------------------------------------------------------------------
    # 5. Test full inference pipeline
    # ------------------------------------------------------------------
    print_section("5. Running inference")
    t0 = time.time()

    # Save input_ids before model pops them
    prompt_input_ids = model_inputs["tokenized_data"]["input_ids"].clone()

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=num_traj_samples,
            max_generation_length=128,
            return_extra=True,
        )

    elapsed = time.time() - t0
    print(f"   Inference done in {elapsed:.1f}s")
    print(f"   pred_xyz shape: {pred_xyz.shape}")
    print(f"   pred_rot shape: {pred_rot.shape}")

    # Verify extra has CoC text
    coc_texts = extra.get("cot", None)
    assert coc_texts is not None, "No CoC text in extra"
    coc_flat = coc_texts.flatten().tolist()
    print(f"   Got {len(coc_flat)} CoC texts")
    for i, txt in enumerate(coc_flat[:2]):
        preview = txt[:100] if txt else "(empty)"
        print(f"   CoC[{i}]: {preview}...")

    # Verify input_ids was popped (the bug we fixed)
    assert "input_ids" not in model_inputs["tokenized_data"], \
        "Expected input_ids to be popped by model"
    print("   Confirmed: input_ids was popped from model_inputs (as expected)")
    print(f"   Saved prompt_input_ids shape: {prompt_input_ids.shape} (still available)")

    # ------------------------------------------------------------------
    # 6. Test reward functions with real outputs
    # ------------------------------------------------------------------
    print_section("6. Testing reward functions")

    # Prepare data in the format rewards expect (flattened lists)
    sample_pred = pred_xyz[0, 0, 0].cpu().numpy().flatten().tolist()
    sample_gt = data["ego_future_xyz"][0, 0].numpy().flatten().tolist()
    sample_coc = coc_flat[0] if coc_flat[0] else ""

    traj_rewards = trajectory_quality_reward(
        [sample_coc], pred_xyz=[sample_pred], gt_xyz=[sample_gt]
    )
    reason_rewards = reasoning_quality_reward([sample_coc])
    consist_rewards = consistency_reward(
        [sample_coc], pred_xyz=[sample_pred]
    )

    print(f"   trajectory_quality: {traj_rewards[0]:.4f}")
    print(f"   reasoning_quality:  {reason_rewards[0]:.4f}")
    print(f"   consistency:        {consist_rewards[0]:.4f}")

    weighted = (
        0.5 * traj_rewards[0]
        + 0.25 * reason_rewards[0]
        + 0.25 * consist_rewards[0]
    )
    print(f"   weighted total:     {weighted:.4f}")

    # ------------------------------------------------------------------
    # 7. Test rollout function
    # ------------------------------------------------------------------
    print_section("7. Testing rollout function")
    rollout_fn = make_rollout_func(
        full_model=model,
        processor=processor,
        avdi=avdi,
        num_traj_samples=num_traj_samples,
        temperature=0.6,
        top_p=0.98,
        max_generation_length=128,
        device=device,
    )

    # Build a prompt string matching what TRL would pass
    prompt_messages = _build_prompt_text(clip_id, t0_us)
    prompt_str = " ".join(m["content"] for m in prompt_messages)

    t0 = time.time()
    result = rollout_fn([prompt_str], trainer=None)
    elapsed = time.time() - t0
    print(f"   Rollout done in {elapsed:.1f}s")

    print(f"   prompt_ids shape:     {result['prompt_ids'].shape}")
    print(f"   completion_ids shape: {result['completion_ids'].shape}")
    print(f"   logprobs shape:       {result['logprobs'].shape}")
    print(f"   num pred_xyz:         {len(result['pred_xyz'])}")
    print(f"   num gt_xyz:           {len(result['gt_xyz'])}")
    print(f"   num completions:      {len(result['completions'])}")

    # Verify shapes match num_traj_samples
    B_G = num_traj_samples  # 1 prompt * G samples
    assert result["prompt_ids"].shape[0] == B_G, \
        f"Expected {B_G} rows, got {result['prompt_ids'].shape[0]}"
    assert result["completion_ids"].shape[0] == B_G
    assert result["logprobs"].shape[0] == B_G
    assert len(result["pred_xyz"]) == B_G
    assert len(result["gt_xyz"]) == B_G
    assert len(result["completions"]) == B_G
    print(f"   All shapes match num_traj_samples={num_traj_samples}: OK")

    # Verify logprobs are negative (valid log-probs)
    nonzero_logprobs = result["logprobs"][result["logprobs"] != 0.0]
    if len(nonzero_logprobs) > 0:
        assert (nonzero_logprobs <= 0).all(), "Log-probs should be <= 0"
        print(f"   Logprobs range: [{nonzero_logprobs.min():.2f}, {nonzero_logprobs.max():.2f}]")
    else:
        print("   WARNING: All logprobs are zero (empty completions?)")

    # ------------------------------------------------------------------
    # 8. Test rewards on rollout output
    # ------------------------------------------------------------------
    print_section("8. Testing rewards on rollout output")
    completions = result["completions"]
    traj_r = trajectory_quality_reward(
        completions, pred_xyz=result["pred_xyz"], gt_xyz=result["gt_xyz"]
    )
    reason_r = reasoning_quality_reward(completions)
    consist_r = consistency_reward(completions, pred_xyz=result["pred_xyz"])

    for i in range(min(2, len(completions))):
        print(f"   Sample {i}: traj={traj_r[i]:.3f} reason={reason_r[i]:.3f} consist={consist_r[i]:.3f}")

    # ------------------------------------------------------------------
    # 9. Test freeze function
    # ------------------------------------------------------------------
    print_section("9. Testing parameter freezing")
    from alpamayo_r1.training.train_grpo import _freeze_non_vlm_params

    # Count params before
    total_params = sum(p.numel() for p in model.parameters())
    vlm_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("vlm."))
    non_vlm_params = total_params - vlm_params

    _freeze_non_vlm_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_params - trainable

    print(f"   Total params:     {total_params:,}")
    print(f"   VLM params:       {vlm_params:,}")
    print(f"   Non-VLM params:   {non_vlm_params:,}")
    print(f"   Trainable:        {trainable:,}")
    print(f"   Frozen:           {frozen:,}")
    assert frozen >= non_vlm_params, "All non-VLM params should be frozen"
    print("   Freeze check: OK")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_section("ALL TESTS PASSED")
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"   Peak GPU memory: {mem:.1f} GB")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
