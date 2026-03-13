#!/usr/bin/env python3
"""Pre-download all chunk files needed for GRPO training.

Downloads egomotion + 4 camera features for every chunk that contains at least
one training clip.  Once cached, the trainer's ``maybe_stream=True`` calls will
hit local files instead of making HTTP requests to HuggingFace, avoiding 429
rate-limit errors during multi-GPU training.

Uses the same clip selection logic as ``build_alpamayo_dataset()`` so the
downloaded data exactly matches what training will need.

Usage:
    # Dry run — see what would be downloaded
    python scripts/data/download_training_clips.py --max-samples 100 --dry-run

    # Download chunks for first 100 training clips
    python scripts/data/download_training_clips.py --max-samples 100

    # Download all clips matching grpo_default.yaml (3000)
    python scripts/data/download_training_clips.py --max-samples 3000

    # Download entire train split
    python scripts/data/download_training_clips.py
"""

import argparse

import pandas as pd
from physical_ai_av import PhysicalAIAVDatasetInterface


def resolve_clip_ids(avdi, *, split, clip_ids_file, exclude_clip_ids_file, max_samples):
    """Resolve training clip IDs using the same logic as build_alpamayo_dataset()."""
    clip_index = avdi.clip_index

    if clip_ids_file is not None:
        clip_ids_df = pd.read_parquet(clip_ids_file)
        clip_ids = clip_ids_df["clip_id"].tolist()
        valid_for_split = clip_index[
            (clip_index["split"] == split) & clip_index["clip_is_valid"]
        ].index
        clip_ids = [c for c in clip_ids if c in valid_for_split]
        print(f"Loaded {len(clip_ids)} clips from {clip_ids_file} (split={split})")
    else:
        split_df = clip_index[(clip_index["split"] == split) & clip_index["clip_is_valid"]]
        clip_ids = split_df.index.tolist()
        print(f"Found {len(clip_ids)} valid clips for split '{split}'")

    if exclude_clip_ids_file is not None:
        exclude_df = pd.read_parquet(exclude_clip_ids_file)
        exclude_set = set(exclude_df["clip_id"].tolist())
        before = len(clip_ids)
        clip_ids = [c for c in clip_ids if c not in exclude_set]
        print(f"Excluded {before - len(clip_ids)} clips ({exclude_clip_ids_file}), {len(clip_ids)} remaining")

    if max_samples is not None:
        clip_ids = clip_ids[:max_samples]
        print(f"Capped to {len(clip_ids)} clips (--max-samples {max_samples})")

    return clip_ids


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download training clip data to avoid HF rate limits during multi-GPU GRPO"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of clips to download (default: all)",
    )
    parser.add_argument(
        "--clip-ids-file",
        default=None,
        help="Optional parquet file with clip_id column to override split-based selection",
    )
    parser.add_argument(
        "--exclude-clip-ids-file",
        default="notebooks/clip_ids.parquet",
        help="Parquet file with clip_ids to exclude (default: notebooks/clip_ids.parquet)",
    )
    parser.add_argument(
        "--dataset-revision",
        default="05e158af89ba",
        help="HuggingFace dataset revision (default: 05e158af89ba)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download threads (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    avdi = PhysicalAIAVDatasetInterface(
        revision=args.dataset_revision,
        confirm_download_threshold_gb=float("inf"),
    )
    print(f"Dataset revision: {avdi.revision}")

    clip_ids = resolve_clip_ids(
        avdi,
        split=args.split,
        clip_ids_file=args.clip_ids_file,
        exclude_clip_ids_file=args.exclude_clip_ids_file,
        max_samples=args.max_samples,
    )

    if not clip_ids:
        print("No clips to download.")
        return

    # Find unique chunks
    chunks = sorted(set(avdi.clip_index.loc[clip_ids, "chunk"]))

    # Same 5 features used by evaluation and training rollouts
    features = [
        avdi.features.LABELS.EGOMOTION,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
    ]

    print(f"\nClips: {len(clip_ids)}")
    print(f"Unique chunks: {len(chunks)}")
    print(f"Features per chunk: {len(features)}")
    print(f"Total files to download: {len(chunks) * len(features)}")

    if args.dry_run:
        print("\n[DRY RUN] Would download these chunks:")
        for chunk in chunks:
            print(f"  {chunk}")
        print(f"\n[DRY RUN] Exiting without downloading.")
        return

    print()
    avdi.download_chunk_features(chunks, features, max_workers=args.max_workers)
    print("\nDone. All chunk files are now cached locally.")


if __name__ == "__main__":
    main()
