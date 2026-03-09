#!/usr/bin/env python3
"""Pre-download all chunk files needed for the curated evaluation set.

Downloads egomotion + 4 camera features for every chunk that contains at least
one clip from notebooks/clip_ids.parquet (test split only).  Once cached, the
evaluation script's ``maybe_stream=True`` calls will hit local files instead of
making HTTP requests to HuggingFace.

Usage:
    python scripts/data/download_eval_clips.py
    python scripts/data/download_eval_clips.py --all-test     # full test split, not just curated
    python scripts/data/download_eval_clips.py --max-workers 4
"""

import argparse

import pandas as pd
from physical_ai_av import PhysicalAIAVDatasetInterface


def main():
    parser = argparse.ArgumentParser(description="Pre-download evaluation clip data")
    parser.add_argument(
        "--all-test",
        action="store_true",
        help="Download all valid test clips instead of just curated set",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download threads (default: 8)",
    )
    args = parser.parse_args()

    avdi = PhysicalAIAVDatasetInterface(confirm_download_threshold_gb=float("inf"))
    print(f"Dataset revision: {avdi.revision}")

    # Determine clip list
    if args.all_test:
        clip_index = avdi.clip_index
        test_df = clip_index[(clip_index["split"] == "test") & clip_index["clip_is_valid"]]
        clip_ids = test_df.index.tolist()
        print(f"Full test split: {len(clip_ids)} clips")
    else:
        clip_ids_df = pd.read_parquet("notebooks/clip_ids.parquet")
        all_eval_ids = set(clip_ids_df["clip_id"].tolist())
        clip_index = avdi.clip_index
        test_ids = set(clip_index[clip_index["split"] == "test"].index)
        clip_ids = sorted(all_eval_ids & test_ids)
        print(f"Curated test set: {len(clip_ids)} clips")

    # Find unique chunks
    chunks = sorted(set(avdi.clip_index.loc[clip_ids, "chunk"]))
    print(f"Unique chunks to download: {len(chunks)}")

    # Features needed for evaluation
    features = [
        avdi.features.LABELS.EGOMOTION,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
    ]
    print(f"Features per chunk: {len(features)}")
    print(f"Total files to download: {len(chunks) * len(features)}")
    print()

    avdi.download_chunk_features(chunks, features, max_workers=args.max_workers)
    print("\nDone. All chunk files are now cached locally.")


if __name__ == "__main__":
    main()
