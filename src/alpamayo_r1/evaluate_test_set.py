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

"""
Optimized evaluation script with batched processing and parallel data loading.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from physical_ai_av import PhysicalAIAVDatasetInterface
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class AlpamayoDataset(Dataset):
    """Dataset for batched loading of Alpamayo samples."""

    def __init__(self, clip_ids, t0_us=5_100_000):
        self.clip_ids = clip_ids
        self.t0_us = t0_us
        # Each worker process needs its own AVDI instance
        self.avdi = None

    def __len__(self):
        return len(self.clip_ids)

    def _get_avdi(self):
        """Lazy initialization of AVDI per worker."""
        if self.avdi is None:
            self.avdi = PhysicalAIAVDatasetInterface()
        return self.avdi

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        try:
            avdi = self._get_avdi()
            data = load_physical_aiavdataset(
                clip_id=clip_id,
                t0_us=self.t0_us,
                avdi=avdi,
                maybe_stream=True,
            )
            return {
                "clip_id": clip_id,
                "image_frames": data["image_frames"],
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
                "ego_future_xyz": data["ego_future_xyz"],
                "ego_future_rot": data["ego_future_rot"],
                "success": True,
                "error": None,
            }
        except Exception as e:
            return {
                "clip_id": clip_id,
                "image_frames": None,
                "ego_history_xyz": None,
                "ego_history_rot": None,
                "ego_future_xyz": None,
                "ego_future_rot": None,
                "success": False,
                "error": str(e),
            }


def collate_fn(batch):
    """Custom collate function that handles failed samples."""
    return batch


def compute_minADE(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> float:
    """Compute minimum Average Displacement Error (minADE)."""
    gt_xy = gt_xyz.cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    return float(min_ade)


def compute_minFDE(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> float:
    """Compute minimum Final Displacement Error (minFDE)."""
    gt_xy_final = gt_xyz.cpu()[0, 0, -1, :2].numpy()
    pred_xy_final = pred_xyz.cpu().numpy()[0, 0, :, -1, :2]
    diff = np.linalg.norm(pred_xy_final - gt_xy_final[None, ...], axis=1)
    min_fde = diff.min()
    return float(min_fde)


def evaluate_batch(
    model: AlpamayoR1,
    processor,
    batch: list,
    num_traj_samples: int,
    temperature: float,
    top_p: float,
    device: str,
) -> list:
    """Evaluate a batch of samples."""
    results = []

    for sample in batch:
        if not sample["success"]:
            results.append(
                {
                    "clip_id": sample["clip_id"],
                    "t0_us": 5_100_000,
                    "minADE": None,
                    "minFDE": None,
                    "success": False,
                    "error": sample["error"],
                    "coc": None,
                }
            )
            continue

        try:
            # Prepare inputs
            model_inputs = helper.prepare_model_inputs(sample, processor, device)

            # Run inference
            with torch.autocast(device, dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=top_p,
                    temperature=temperature,
                    num_traj_samples=num_traj_samples,
                    max_generation_length=256,
                    return_extra=True,
                )

            # Compute metrics
            min_ade = compute_minADE(pred_xyz, sample["ego_future_xyz"])
            min_fde = compute_minFDE(pred_xyz, sample["ego_future_xyz"])

            # Extract CoC
            coc_text = None
            if "cot" in extra and extra["cot"] is not None and len(extra["cot"]) > 0:
                if len(extra["cot"][0]) > 0:
                    coc_text = extra["cot"][0][0]

            results.append(
                {
                    "clip_id": sample["clip_id"],
                    "t0_us": 5_100_000,
                    "minADE": min_ade,
                    "minFDE": min_fde,
                    "success": True,
                    "error": None,
                    "coc": coc_text,
                }
            )

        except Exception as e:
            import traceback

            results.append(
                {
                    "clip_id": sample["clip_id"],
                    "t0_us": 5_100_000,
                    "minADE": None,
                    "minFDE": None,
                    "success": False,
                    "error": f"{str(e)}\n{traceback.format_exc()}",
                    "coc": None,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Alpamayo-R1 evaluation with batched data loading"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvidia/Alpamayo-R1-10B",
        help="Model name or path",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Number of test samples to evaluate"
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=5,
        help="Number of trajectory samples per prediction",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.98, help="Nucleus sampling top-p")
    parser.add_argument(
        "--t0-us", type=int, default=5_100_000, help="Default t0 timestamp in microseconds"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-clip-ids-file",
        action="store_true",
        help="Use notebooks/clip_ids.parquet instead of full test set",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker (default: 2)",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Use torch.compile for faster inference (PyTorch 2.0+)",
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Alpamayo-R1 Optimized Evaluation (Batched Data Loading)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Trajectory samples per prediction: {args.num_traj_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Data loading workers: {args.num_workers}")
    print(f"Prefetch factor: {args.prefetch_factor}")
    print(f"Model compilation: {'enabled' if args.compile_model else 'disabled'}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model = AlpamayoR1.from_pretrained(args.model_name, dtype=torch.bfloat16).to(args.device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # Optionally compile model for faster inference
    if args.compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    print("Model loaded successfully!")

    # Get clip IDs
    if args.use_clip_ids_file:
        print("\nUsing clip_ids.parquet file...")
        clip_ids_df = pd.read_parquet("notebooks/clip_ids.parquet")
        test_clips = clip_ids_df["clip_id"].tolist()
        print(f"Found {len(test_clips)} clips in clip_ids.parquet")
    else:
        print("\nLoading test split from dataset...")
        avdi = PhysicalAIAVDatasetInterface()
        clip_index = avdi.clip_index
        test_df = clip_index[(clip_index["split"] == "test") & clip_index["clip_is_valid"]]
        test_clips = test_df.index.tolist()
        print(f"Found {len(test_clips)} valid test clips")

    # Limit number of samples if specified
    if args.num_samples is not None:
        test_clips = test_clips[: args.num_samples]
        print(f"Limiting evaluation to {len(test_clips)} samples")

    print(f"\nEvaluating {len(test_clips)} test samples...")
    print("=" * 80)

    # Create dataset and dataloader
    dataset = AlpamayoDataset(test_clips, t0_us=args.t0_us)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one at a time for inference, but prefetch in background
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Evaluate all samples
    all_results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", total=len(test_clips)):
            results = evaluate_batch(
                model=model,
                processor=processor,
                batch=batch,
                num_traj_samples=args.num_traj_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )
            all_results.extend(results)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Compute aggregate statistics
    successful_results = results_df[results_df["success"] == True]
    failed_results = results_df[results_df["success"] == False]

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    if len(successful_results) > 0:
        min_ade_values = successful_results["minADE"].values
        min_fde_values = successful_results["minFDE"].values

        stats = {
            "total_samples": len(results_df),
            "successful_samples": len(successful_results),
            "failed_samples": len(failed_results),
            "minADE": {
                "mean": float(np.mean(min_ade_values)),
                "median": float(np.median(min_ade_values)),
                "std": float(np.std(min_ade_values)),
                "min": float(np.min(min_ade_values)),
                "max": float(np.max(min_ade_values)),
            },
            "minFDE": {
                "mean": float(np.mean(min_fde_values)),
                "median": float(np.median(min_fde_values)),
                "std": float(np.std(min_fde_values)),
                "min": float(np.min(min_fde_values)),
                "max": float(np.max(min_fde_values)),
            },
            "config": {
                "model_name": args.model_name,
                "num_traj_samples": args.num_traj_samples,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "t0_us": args.t0_us,
                "seed": args.seed,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "compile_model": args.compile_model,
            },
        }

        print(f"\nTotal samples: {stats['total_samples']}")
        print(f"Successful: {stats['successful_samples']}")
        print(f"Failed: {stats['failed_samples']}")
        print(f"\nminADE (meters):")
        print(f"  Mean:   {stats['minADE']['mean']:.4f}")
        print(f"  Median: {stats['minADE']['median']:.4f}")
        print(f"  Std:    {stats['minADE']['std']:.4f}")
        print(f"  Min:    {stats['minADE']['min']:.4f}")
        print(f"  Max:    {stats['minADE']['max']:.4f}")
        print(f"\nminFDE (meters):")
        print(f"  Mean:   {stats['minFDE']['mean']:.4f}")
        print(f"  Median: {stats['minFDE']['median']:.4f}")
        print(f"  Std:    {stats['minFDE']['std']:.4f}")
        print(f"  Min:    {stats['minFDE']['min']:.4f}")
        print(f"  Max:    {stats['minFDE']['max']:.4f}")

    else:
        stats = {
            "total_samples": len(results_df),
            "successful_samples": 0,
            "failed_samples": len(failed_results),
            "error": "All evaluations failed",
        }
        print("\n⚠️  All evaluations failed!")

    # Save results
    results_csv = output_dir / "results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Detailed results saved to: {results_csv}")

    stats_json = output_dir / "statistics.json"
    with open(stats_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Summary statistics saved to: {stats_json}")

    # Print failure summary if any
    if len(failed_results) > 0:
        print(f"\n⚠️  {len(failed_results)} samples failed. Error summary:")
        error_counts = failed_results["error"].value_counts()
        for error, count in error_counts.head(5).items():
            print(f"  - {error[:80]}: {count} samples")

    print("=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
