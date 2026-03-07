#!/usr/bin/env python3
"""Check how many evaluation clip features are cached locally vs would stream.

Usage:
    python scripts/check_eval_cache.py
    python scripts/check_eval_cache.py --all-test
"""

import argparse

import pandas as pd
from physical_ai_av import PhysicalAIAVDatasetInterface


def main():
    parser = argparse.ArgumentParser(description="Check eval clip cache status")
    parser.add_argument("--all-test", action="store_true", help="Check full test split")
    args = parser.parse_args()

    avdi = PhysicalAIAVDatasetInterface()
    print(f"Revision: {avdi.revision}")

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

    features = [
        avdi.features.LABELS.EGOMOTION,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
    ]

    # Check cache at chunk level (what download_eval_clips.py downloads)
    chunks = sorted(set(avdi.clip_index.loc[clip_ids, "chunk"]))
    cached_chunks = 0
    missing_chunks = 0
    missing_files = []

    for chunk_id in chunks:
        all_cached = True
        for feature in features:
            filename = avdi.features.get_chunk_feature_filename(chunk_id, feature)
            if avdi.is_file_cached(filename):
                pass
            else:
                all_cached = False
                missing_files.append(filename)
        if all_cached:
            cached_chunks += 1
        else:
            missing_chunks += 1

    total_files = len(chunks) * len(features)
    cached_files = total_files - len(missing_files)

    print(f"\nChunks: {len(chunks)} total, {cached_chunks} fully cached, {missing_chunks} missing")
    print(f"Files:  {total_files} total, {cached_files} cached, {len(missing_files)} missing")

    if missing_files:
        print(f"\nFirst 10 missing files:")
        for f in missing_files[:10]:
            print(f"  {f}")
        print(f"\nRun 'python scripts/download_eval_clips.py' to download missing files.")
    else:
        print("\nAll files cached. Evaluation will not make HF streaming requests.")


if __name__ == "__main__":
    main()
