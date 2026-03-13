"""Recompute consistency reward from a reward_signal JSON file without re-running inference.

Reads coc_text and trajectory meta-action data from the JSON, applies the current
consistency_reward algorithm, and prints a comparison with the original scores.

Usage:
    python scripts/evaluate/recompute_consistency.py reward_signal-100.json
    python scripts/evaluate/recompute_consistency.py reward_signal-100.json --output recomputed.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

from alpamayo_r1.training.meta_actions import _META_ACTION_KEYWORDS


def recompute_entry(text: str, lon_set: set[str], lat_set: set[str]) -> dict:
    """Recompute consistency score for a single entry using the current algorithm."""
    text_lower = text.lower()

    lon_matched = []
    for action in lon_set:
        for kw in _META_ACTION_KEYWORDS.get(action, []):
            if kw in text_lower:
                lon_matched.append(f"{action}: '{kw}'")

    lat_matched = []
    for action in lat_set:
        for kw in _META_ACTION_KEYWORDS.get(action, []):
            if kw in text_lower:
                lat_matched.append(f"{action}: '{kw}'")

    lon_match = len(lon_matched) > 0
    lat_match = len(lat_matched) > 0
    implicit_straight = lat_set == {"go_straight"}
    if implicit_straight:
        lat_match = True

    # Binary scoring: both axes must match
    if lon_match and lat_match:
        new_score = 1.0
    else:
        new_score = 0.0

    return {
        "new_score": new_score,
        "lon_match": lon_match,
        "lat_match": lat_match,
        "implicit_straight": implicit_straight,
        "lon_matched": lon_matched,
        "lat_matched": lat_matched,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to reward_signal JSON file")
    parser.add_argument("--output", "-o", help="Write recomputed JSON to this path")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    samples = data["per_sample"]
    all_old = []
    all_new = []
    per_sample_results = []

    for sample in samples:
        clip_id = sample.get("clip_id", "unknown")
        sample_old = []
        sample_new = []
        entries_out = []

        for entry in sample["meta_debug"]:
            coc_text = entry.get("coc_text", "")
            if not coc_text:
                continue

            # Derive lon_set / lat_set from the entry
            if "lon_set" in entry:
                lon_set = set(entry["lon_set"])
                lat_set = set(entry["lat_set"])
            else:
                # Older format: derive from lon_seq / lat_seq
                lon_set = set(entry.get("lon_seq", []))
                lat_set = set(entry.get("lat_seq", []))

            old_score = entry["consist_score"]
            result = recompute_entry(coc_text, lon_set, lat_set)
            new_score = result["new_score"]

            all_old.append(old_score)
            all_new.append(new_score)
            sample_old.append(old_score)
            sample_new.append(new_score)

            entries_out.append({
                "coc_text": coc_text[:100],
                "old_score": old_score,
                "new_score": new_score,
                "lon_set": sorted(lon_set),
                "lat_set": sorted(lat_set),
                "lon_matched": result["lon_matched"],
                "lat_matched": result["lat_matched"],
                "implicit_straight": result["implicit_straight"],
            })

        n = len(sample_old)
        per_sample_results.append({
            "clip_id": clip_id,
            "old_mean": sum(sample_old) / n if n else 0,
            "new_mean": sum(sample_new) / n if n else 0,
            "entries": entries_out,
        })

    # Print summary
    n = len(all_old)
    old_dist = Counter(all_old)
    new_dist = Counter(all_new)

    print(f"{'':=<60}")
    print(f"Recomputed consistency reward: {args.input}")
    print(f"  {n} entries across {len(samples)} samples")
    print(f"{'':=<60}")

    print(f"\n{'Score':<8} {'Old':>8} {'New':>8} {'Delta':>8}")
    print(f"{'':─<35}")
    for s in [0.0, 0.5, 1.0]:
        o, nw = old_dist.get(s, 0), new_dist.get(s, 0)
        print(f"{s:<8.1f} {o:>8d} {nw:>8d} {nw - o:>+8d}")

    print(f"\n{'Mean':<8} {sum(all_old)/n:>8.3f} {sum(all_new)/n:>8.3f}")
    if n > 1:
        print(f"{'Std':<8} {statistics.stdev(all_old):>8.3f} {statistics.stdev(all_new):>8.3f}")

    # Transition matrix
    transitions = Counter((o, nw) for o, nw in zip(all_old, all_new))
    print(f"\nTransitions:")
    for (o, nw), count in sorted(transitions.items()):
        print(f"  {o:.1f} → {nw:.1f}: {count:>4d}")

    # Sample-level
    sample_old_means = [r["old_mean"] for r in per_sample_results]
    sample_new_means = [r["new_mean"] for r in per_sample_results]
    print(f"\nSample-level mean: {sum(sample_old_means)/len(sample_old_means):.3f} → {sum(sample_new_means)/len(sample_new_means):.3f}")

    # Write output if requested
    if args.output:
        output_data = {
            "source": args.input,
            "algorithm": "binary + implicit go_straight",
            "summary": {
                "num_entries": n,
                "num_samples": len(samples),
                "old_mean": sum(all_old) / n,
                "new_mean": sum(all_new) / n,
                "distribution": {str(s): new_dist.get(s, 0) for s in [0.0, 0.5, 1.0]},
            },
            "per_sample": per_sample_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nWrote recomputed results to {args.output}")


if __name__ == "__main__":
    main()
