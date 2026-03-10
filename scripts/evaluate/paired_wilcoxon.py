#!/usr/bin/env python3
"""Paired statistical tests on per-clip minADE/minFDE.

Each model directory should contain a results.csv produced by
evaluate_quick_test.sh. The script handles both single-trial outputs
(clip_id, t0_us, minADE, minFDE, success, ...) and multi-trial aggregated
outputs (clip_id, minADE, minFDE, success).

Usage:
    python paired_wilcoxon.py <model_a_dir> <model_b_dir>

Example:
    python paired_wilcoxon.py evaluation_results/curated_set/base \\
                              evaluation_results/curated_set/temp06-step600
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


METRICS = ["minADE", "minFDE"]


def load_results(model_dir: Path) -> pd.DataFrame:
    """Load and filter results.csv from model_dir."""
    csv_path = model_dir / "results.csv"
    if not csv_path.exists():
        sys.exit(f"Error: {csv_path} not found")
    df = pd.read_csv(csv_path)
    if "success" in df.columns:
        df = df[df["success"] == True]
    n_missing = df[METRICS].isna().any(axis=1).sum()
    if n_missing:
        print(f"  Warning: dropping {n_missing} rows with missing metrics in {model_dir.name}")
        df = df.dropna(subset=METRICS)
    print(f"  {model_dir.name}: {len(df)} clips")
    return df[["clip_id", *METRICS]]


def main():
    parser = argparse.ArgumentParser(
        description="Paired statistical tests on per-clip metrics from evaluate_quick_test.sh output."
    )
    parser.add_argument("model_a", type=Path, help="Directory for model A (contains results.csv)")
    parser.add_argument("model_b", type=Path, help="Directory for model B (contains results.csv)")
    args = parser.parse_args()

    for d in (args.model_a, args.model_b):
        if not d.is_dir():
            sys.exit(f"Error: {d} is not a directory")

    name_a = args.model_a.name
    name_b = args.model_b.name

    print("Loading results...")
    df_a = load_results(args.model_a)
    df_b = load_results(args.model_b)

    merged = df_a.merge(df_b, on="clip_id", suffixes=(f"_{name_a}", f"_{name_b}"))
    print(f"  Paired clips: {len(merged)}\n")

    for metric in METRICS:
        vals_a = merged[f"{metric}_{name_a}"].values
        vals_b = merged[f"{metric}_{name_b}"].values
        mean_a = np.mean(vals_a)
        mean_b = np.mean(vals_b)
        diff = vals_a - vals_b
        better = name_b if np.mean(diff) > 0 else name_a

        w_stat, w_p = wilcoxon(vals_a, vals_b)
        t_stat, t_p = ttest_rel(vals_a, vals_b)

        print(f"=== {metric} ===")
        print(f"  {name_a:>25s}: {mean_a:.4f}")
        print(f"  {name_b:>25s}: {mean_b:.4f}")
        print(f"  {'mean diff (A - B)':>25s}: {np.mean(diff):+.4f}")
        print(f"  {'better':>25s}: {better}")
        print(f"  {'Wilcoxon stat':>25s}: {w_stat:.1f}   p={w_p:.4f}")
        print(f"  {'paired t stat':>25s}: {t_stat:+.3f}   p={t_p:.4f}")
        print()


if __name__ == "__main__":
    main()
