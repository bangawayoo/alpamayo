#!/usr/bin/env python3
"""Save camera frames from a clip to disk for visual inspection.

Usage:
    python scripts/data/save_clip_images.py --clip-id 25cd4769-5dcf-4b53-a351-bf2c5deb6124
    python scripts/data/save_clip_images.py --clip-id 25cd4769-5dcf-4b53-a351-bf2c5deb6124 --output outputs/clip_frames
"""
import argparse
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "src")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--output", default="outputs/clip_frames")
    args = parser.parse_args()

    import os
    os.makedirs(args.output, exist_ok=True)

    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from physical_ai_av import PhysicalAIAVDatasetInterface

    avdi = PhysicalAIAVDatasetInterface()
    data = load_physical_aiavdataset(
        clip_id=args.clip_id, t0_us=args.t0_us, avdi=avdi, maybe_stream=True
    )

    # image_frames shape: (N_cameras, N_frames, 3, H, W)
    frames = data["image_frames"]
    n_cameras, n_frames = frames.shape[0], frames.shape[1]
    print(f"Clip: {args.clip_id}")
    print(f"Cameras: {n_cameras}, Frames per camera: {n_frames}")
    print(f"Frame shape: {frames.shape[2:]}")

    for cam_idx in range(n_cameras):
        for frame_idx in range(n_frames):
            frame = frames[cam_idx, frame_idx]  # (3, H, W)
            img = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
            path = os.path.join(args.output, f"cam{cam_idx}_frame{frame_idx}.png")
            img.save(path)
            print(f"  Saved {path} ({img.size})")

    print(f"\nAll frames saved to {args.output}/")


if __name__ == "__main__":
    main()
