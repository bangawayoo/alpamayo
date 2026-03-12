#!/usr/bin/env python3
"""Inspect per-timestep meta-action labels on real ground-truth trajectories.

Loads N samples from PhysicalAI-AV, extracts the ground-truth future trajectory,
and classifies each timestep into longitudinal + lateral meta-actions at 10Hz.
Shows the dominant (summary) label plus the unique action sequence transitions.

No model or GPU needed.

Usage:
    python scripts/evaluate/inspect_meta_actions.py --num-samples 20
    python scripts/evaluate/inspect_meta_actions.py --num-samples 50 --seed 123
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from physical_ai_av import PhysicalAIAVDatasetInterface

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.training.meta_actions import extract_meta_actions, extract_meta_actions_summary


def _ordered_unique(labels: list[str]) -> list[str]:
    """Return unique labels in order of first appearance (action sequence)."""
    seen: set[str] = set()
    result: list[str] = []
    for l in labels:
        if l not in seen:
            seen.add(l)
            result.append(l)
    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect meta-actions on real trajectories")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading clip index...")
    avdi = PhysicalAIAVDatasetInterface()
    clip_index = avdi.clip_index
    split_df = clip_index[(clip_index["split"] == args.split) & clip_index["clip_is_valid"]]
    all_clips = split_df.index.tolist()
    print(f"  {len(all_clips)} valid {args.split} clips")

    sampled = rng.choice(all_clips, size=min(args.num_samples, len(all_clips)), replace=False)
    print(f"  Sampling {len(sampled)} clips\n")

    # Aggregate counters for summary distribution
    lon_counter: Counter = Counter()
    lat_counter: Counter = Counter()
    # Per-timestep counters (across all clips)
    lon_timestep_counter: Counter = Counter()
    lat_timestep_counter: Counter = Counter()
    failed = 0

    print(f"{'Clip ID':>45s}  {'LON summary':>20s}  {'LAT summary':>20s}  "
          f"{'v_start':>8s}  {'v_end':>8s}  {'net_y':>8s}")
    print("-" * 130)

    for clip_id in sampled:
        try:
            data = load_physical_aiavdataset(
                clip_id=clip_id, t0_us=args.t0_us, avdi=avdi, maybe_stream=True,
            )
            traj = data["ego_future_xyz"].cpu().numpy()[0, 0]  # (T, 3)

            meta = extract_meta_actions(traj)
            summary = extract_meta_actions_summary(traj)
            lon_counter[summary.longitudinal] += 1
            lat_counter[summary.lateral] += 1
            lon_timestep_counter.update(meta.longitudinal)
            lat_timestep_counter.update(meta.lateral)

            # Kinematics
            velocity = np.diff(traj[:, 0]) / 0.1
            v_start = float(np.median(velocity[:10]))
            v_end = float(np.median(velocity[-10:]))
            net_y = float(traj[-1, 1] - traj[0, 1])

            # Action sequence (ordered unique transitions)
            lon_seq = " -> ".join(_ordered_unique(meta.longitudinal))
            lat_seq = " -> ".join(_ordered_unique(meta.lateral))

            print(f"{clip_id:>45s}  {summary.longitudinal:>20s}  {summary.lateral:>20s}  "
                  f"{v_start:>8.2f}  {v_end:>8.2f}  {net_y:>8.2f}")
            print(f"{'':>45s}    LON: {lon_seq}")
            print(f"{'':>45s}    LAT: {lat_seq}")

        except Exception as e:
            failed += 1
            print(f"{clip_id:>45s}  ERROR: {e}")

    # Summary statistics
    n = len(sampled) - failed
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY DISTRIBUTION  ({n} clips, {failed} failed)")
    print(f"{'=' * 70}")

    print("\n  Longitudinal (dominant per clip):")
    for label, count in lon_counter.most_common():
        bar = "#" * int(40 * count / n)
        print(f"    {label:>20s}: {count:>4d} ({100*count/n:5.1f}%)  {bar}")

    print("\n  Lateral (dominant per clip):")
    for label, count in lat_counter.most_common():
        bar = "#" * int(40 * count / n)
        print(f"    {label:>20s}: {count:>4d} ({100*count/n:5.1f}%)  {bar}")

    # Per-timestep distribution
    total_lon = sum(lon_timestep_counter.values())
    total_lat = sum(lat_timestep_counter.values())

    print(f"\n{'=' * 70}")
    print(f"  PER-TIMESTEP DISTRIBUTION  ({total_lon} total timesteps)")
    print(f"{'=' * 70}")

    print("\n  Longitudinal (per timestep):")
    for label, count in lon_timestep_counter.most_common():
        bar = "#" * int(40 * count / total_lon)
        print(f"    {label:>20s}: {count:>5d} ({100*count/total_lon:5.1f}%)  {bar}")

    print("\n  Lateral (per timestep):")
    for label, count in lat_timestep_counter.most_common():
        bar = "#" * int(40 * count / total_lat)
        print(f"    {label:>20s}: {count:>5d} ({100*count/total_lat:5.1f}%)  {bar}")

    print()


if __name__ == "__main__":
    main()
