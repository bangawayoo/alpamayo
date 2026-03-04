#!/usr/bin/env python3
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

"""Evaluate reward signal quality on a training-set subset.

Runs inference over N randomly-sampled training clips, computes all three
GRPO reward signals, and prints summary statistics so we can assess whether
the rewards are well-calibrated and non-trivial.

Automatically uses all available GPUs in parallel (one process per GPU).

Usage:
    python scripts/evaluate_reward_signal.py --num-samples 50
    python scripts/evaluate_reward_signal.py --num-samples 20 --num-traj-samples 4
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Must be set before any CUDA import/init.  In MIG environments the default
# CUDA caching allocator queries NVML for per-device free memory, which fails
# with "NVML_SUCCESS == r INTERNAL ASSERT FAILED".  expandable_segments uses
# a different allocation strategy that skips those NVML calls entirely.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.multiprocessing as mp
from physical_ai_av import PhysicalAIAVDatasetInterface
from tqdm import tqdm

# Ensure src/ is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.training.rewards import (
    consistency_reward,
    reasoning_quality_reward,
    trajectory_quality_reward,
)


# Reward weights from grpo_default.yaml
TRAJ_WEIGHT = 0.50
REASONING_WEIGHT = 0.25
CONSISTENCY_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Per-sample inference + reward computation
# ---------------------------------------------------------------------------

def run_inference(model, processor, avdi, clip_id: str, t0_us: int,
                  num_traj_samples: int, temperature: float, top_p: float,
                  device: str) -> dict:
    """Load one clip, run inference, return trajectories and CoC text."""
    try:
        data = load_physical_aiavdataset(
            clip_id=clip_id, t0_us=t0_us, avdi=avdi, maybe_stream=True
        )
    except Exception as e:
        return {"error": f"data load failed: {e}"}

    try:
        model_inputs = helper.prepare_model_inputs(data, processor, device)

        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=top_p,
                temperature=temperature,
                num_traj_samples=num_traj_samples,
                max_generation_length=256,
                return_extra=True,
            )
    except Exception as e:
        return {"error": f"inference failed: {e}"}

    # pred_xyz: (1, 1, S, T, 3)  →  flatten (S, T, 3) for reward functions
    pred_flat = pred_xyz.cpu().numpy()[0, 0].flatten().tolist()
    # gt_xyz:   (1, 1, T, 3)    →  flatten (T, 3)
    gt_flat = data["ego_future_xyz"].cpu().numpy()[0, 0].flatten().tolist()

    # CoC texts: extra["cot"] shape is (1, S) or nested list
    coc_texts = []
    if "cot" in extra and extra["cot"] is not None:
        raw = extra["cot"]
        if hasattr(raw, "flatten"):
            coc_texts = raw.flatten().tolist()
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, (list, tuple)):
                    coc_texts.extend(item)
                else:
                    coc_texts.append(item)

    return {
        "clip_id": clip_id,
        "pred_flat": pred_flat,
        "gt_flat": gt_flat,
        "coc_texts": coc_texts,
        "error": None,
    }


def compute_rewards_for_sample(result: dict, ade_threshold: float) -> dict:
    """Compute all three rewards for a single inference result."""
    pred_flat = result["pred_flat"]
    gt_flat = result["gt_flat"]
    coc_texts = result["coc_texts"]
    n_texts = len(coc_texts) if coc_texts else 1

    traj_r = trajectory_quality_reward(
        completions=[""] * n_texts,
        pred_xyz=[pred_flat] * n_texts,
        gt_xyz=[gt_flat] * n_texts,
        ade_threshold=ade_threshold,
    )

    if coc_texts:
        reason_r = reasoning_quality_reward(completions=coc_texts)
        consist_r = consistency_reward(
            completions=coc_texts,
            pred_xyz=[pred_flat] * n_texts,
        )
    else:
        reason_r = [0.0]
        consist_r = [0.0]

    traj_mean    = float(np.mean(traj_r))
    reason_mean  = float(np.mean(reason_r))
    consist_mean = float(np.mean(consist_r))
    weighted = (TRAJ_WEIGHT * traj_mean
                + REASONING_WEIGHT * reason_mean
                + CONSISTENCY_WEIGHT * consist_mean)

    return {
        "traj_reward":        traj_mean,
        "reasoning_reward":   reason_mean,
        "consistency_reward": consist_mean,
        "weighted_reward":    weighted,
        "first_coc":          coc_texts[0] if coc_texts else "",
    }


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------

def _worker(rank: int, world_size: int, clip_ids: list, args_dict: dict, tmp_dir: str):
    """Run inference on a single GPU (rank) over a disjoint clip subset.

    Uses device_map={"": f"cuda:{rank}"} to pin all model layers to one GPU
    without touching CUDA_VISIBLE_DEVICES, so all workers can coexist.
    Results are written to ``tmp_dir/rank_{rank}.json``.
    """
    device = f"cuda:{rank}"

    # Load model pinned to this GPU
    model = AlpamayoR1.from_pretrained(
        args_dict["model_name"],
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # One AVDI instance per worker (not safe to share across processes)
    avdi = PhysicalAIAVDatasetInterface()

    # Each rank processes every world_size-th clip starting at rank
    my_clips = clip_ids[rank::world_size]

    rewards = []
    failed = []

    for clip_id in tqdm(my_clips, desc=f"GPU {rank}", position=rank, leave=True):
        result = run_inference(
            model=model,
            processor=processor,
            avdi=avdi,
            clip_id=clip_id,
            t0_us=args_dict["t0_us"],
            num_traj_samples=args_dict["num_traj_samples"],
            temperature=args_dict["temperature"],
            top_p=args_dict["top_p"],
            device=device,
        )
        if result.get("error"):
            failed.append({"clip_id": clip_id, "error": result["error"]})
            tqdm.write(f"  [GPU {rank}] FAILED {clip_id}: {result['error']}")
            continue

        r = compute_rewards_for_sample(result, ade_threshold=args_dict["ade_threshold"])
        r["clip_id"] = clip_id
        rewards.append(r)

    out_path = Path(tmp_dir) / f"rank_{rank}.json"
    with open(out_path, "w") as f:
        json.dump({"rewards": rewards, "failed": failed}, f)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def print_stats(name: str, values: list[float]) -> None:
    arr = np.array(values)
    print(f"  {name}:")
    print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  "
          f"median={np.median(arr):.4f}")
    buckets = [0.0, 0.25, 0.5, 0.75, 1.0]
    counts = np.histogram(arr, bins=buckets)[0]
    total = len(arr)
    bucket_str = "  ".join(
        f"[{buckets[i]:.2f},{buckets[i+1]:.2f}): {counts[i]}/{total}"
        for i in range(len(counts))
    )
    print(f"    dist: {bucket_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO reward signals on training data")
    parser.add_argument("--model-name", default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of training clips to evaluate")
    parser.add_argument("--num-traj-samples", type=int, default=8,
                        help="Trajectory samples per clip (match GRPO num_generations)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--ade-threshold", type=float, default=5.0,
                        help="ADE threshold (meters) for trajectory reward normalization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reward_signal_eval.json",
                        help="Path to save per-sample results as JSON")
    parser.add_argument("--show-coc-samples", type=int, default=3,
                        help="Number of sample CoC texts to print")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("ERROR: no CUDA GPUs found.")
        sys.exit(1)

    print("=" * 70)
    print("  Alpamayo-R1 Reward Signal Evaluation")
    print("=" * 70)
    print(f"  Model:            {args.model_name}")
    print(f"  GPUs:             {world_size}")
    print(f"  Training samples: {args.num_samples}")
    print(f"  Traj samples:     {args.num_traj_samples}  (GRPO group size G)")
    print(f"  ADE threshold:    {args.ade_threshold} m")
    print(f"  Reward weights:   traj={TRAJ_WEIGHT}  reason={REASONING_WEIGHT}  consist={CONSISTENCY_WEIGHT}")
    print("=" * 70)

    # Sample clip IDs from training split
    print("\nLoading training clip index...")
    avdi = PhysicalAIAVDatasetInterface()
    clip_index = avdi.clip_index
    train_df = clip_index[(clip_index["split"] == "train") & clip_index["clip_is_valid"]]
    all_train_clips = train_df.index.tolist()
    print(f"  Found {len(all_train_clips)} valid training clips")

    rng = np.random.default_rng(args.seed)
    sampled_clips = rng.choice(
        all_train_clips,
        size=min(args.num_samples, len(all_train_clips)),
        replace=False,
    ).tolist()
    print(f"  Sampled {len(sampled_clips)} clips  →  "
          f"~{len(sampled_clips) // world_size} per GPU")

    # Pack args into a plain dict for pickling across processes
    args_dict = {
        "model_name":       args.model_name,
        "num_traj_samples": args.num_traj_samples,
        "temperature":      args.temperature,
        "top_p":            args.top_p,
        "t0_us":            args.t0_us,
        "ade_threshold":    args.ade_threshold,
    }

    # Spawn one worker per GPU; results written to per-rank JSON files
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"\nLaunching {world_size} worker(s)...")
        mp.spawn(
            _worker,
            args=(world_size, sampled_clips, args_dict, tmp_dir),
            nprocs=world_size,
            join=True,
        )

        # Aggregate results from all ranks
        all_rewards = []
        failed = []
        for rank in range(world_size):
            rank_path = Path(tmp_dir) / f"rank_{rank}.json"
            with open(rank_path) as f:
                data = json.load(f)
            all_rewards.extend(data["rewards"])
            failed.extend(data["failed"])

    print(f"\n  Successful: {len(all_rewards)} / {len(sampled_clips)}")
    if failed:
        print(f"  Failed:     {len(failed)}")

    if not all_rewards:
        print("ERROR: no successful evaluations.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Aggregate statistics
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  REWARD SIGNAL STATISTICS")
    print("=" * 70)

    traj_vals     = [r["traj_reward"]        for r in all_rewards]
    reason_vals   = [r["reasoning_reward"]   for r in all_rewards]
    consist_vals  = [r["consistency_reward"] for r in all_rewards]
    weighted_vals = [r["weighted_reward"]    for r in all_rewards]

    print_stats(f"Trajectory quality  (w={TRAJ_WEIGHT})", traj_vals)
    print_stats(f"Reasoning quality   (w={REASONING_WEIGHT})", reason_vals)
    print_stats(f"Consistency         (w={CONSISTENCY_WEIGHT})", consist_vals)
    print_stats("Weighted total", weighted_vals)

    print("\n  Reward discriminability check (std > 0 = good signal):")
    for name, vals in [
        ("traj", traj_vals), ("reasoning", reason_vals),
        ("consistency", consist_vals), ("weighted", weighted_vals),
    ]:
        std = float(np.std(vals))
        flag = "OK" if std > 0.01 else "FLAT — consider re-tuning"
        print(f"    {name:15s}: std={std:.4f}  [{flag}]")

    # ----------------------------------------------------------------
    # Sample CoC texts
    # ----------------------------------------------------------------
    if args.show_coc_samples > 0:
        print(f"\n{'=' * 70}")
        print(f"  SAMPLE CoC TEXTS (first {args.show_coc_samples} successful clips)")
        print("=" * 70)
        for r in all_rewards[:args.show_coc_samples]:
            print(f"\n  --- Clip: {r['clip_id']} ---")
            print(f"  traj={r['traj_reward']:.3f}  "
                  f"reason={r['reasoning_reward']:.3f}  "
                  f"consist={r['consistency_reward']:.3f}  "
                  f"weighted={r['weighted_reward']:.3f}")
            coc = r["first_coc"]
            print(f"  CoC: {coc[:400]!r}" if coc else "  CoC: (empty)")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    output = {
        "config": vars(args),
        "summary": {
            "num_evaluated": len(all_rewards),
            "num_failed":    len(failed),
            "num_gpus":      world_size,
            "traj_reward":        {"mean": float(np.mean(traj_vals)),     "std": float(np.std(traj_vals)),     "min": float(np.min(traj_vals)),     "max": float(np.max(traj_vals))},
            "reasoning_reward":   {"mean": float(np.mean(reason_vals)),   "std": float(np.std(reason_vals)),   "min": float(np.min(reason_vals)),   "max": float(np.max(reason_vals))},
            "consistency_reward": {"mean": float(np.mean(consist_vals)),  "std": float(np.std(consist_vals)),  "min": float(np.min(consist_vals)),  "max": float(np.max(consist_vals))},
            "weighted_reward":    {"mean": float(np.mean(weighted_vals)), "std": float(np.std(weighted_vals)), "min": float(np.min(weighted_vals)), "max": float(np.max(weighted_vals))},
        },
        "per_sample": all_rewards,
        "failed":     failed,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
