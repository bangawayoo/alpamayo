#!/usr/bin/env python3
"""Plot GRPO training curves from a TensorBoard event directory.

Usage:
    python plot_training_curves.py <tb_run_dir> [--output <path>] [--title <str>] [--decay <float>]

Example:
    python scripts/tools/plot_training_curves.py \\
        outputs/grpo/temp06/runs/Mar06_21-30-09_trl-no-vllm-run-06temp-2323-default0-0 \\
        --output docs/grpo_training_curves_temp06.png \\
        --title "GRPO Training Curves (temp=0.6)"
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PANELS = [
    ("train/rewards/trajectory_quality_reward/mean", "Trajectory Quality Reward (minADE)", "steelblue"),
    ("train/rewards/reasoning_quality_reward/mean",  "Reasoning Quality Reward (rule-based)", "darkorange"),
    ("train/rewards/consistency_reward/mean",        "Consistency Reward",                    "firebrick"),
    ("train/reward",                                 "Total Weighted Reward",                 "mediumseagreen"),
    ("train/grad_norm",                              "Gradient Norm",                         "indianred"),
    ("train/entropy",                                "Entropy",                               "mediumpurple"),
]


def get_scalar(ea: EventAccumulator, tag: str):
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    return steps, vals


def ema(vals: np.ndarray, decay: float) -> np.ndarray:
    out = np.empty_like(vals)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = decay * out[i - 1] + (1 - decay) * vals[i]
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training curves from TensorBoard logs.")
    parser.add_argument("tb_run_dir", type=Path, help="TensorBoard run directory")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path (default: <tb_run_dir>/training_curves.png)")
    parser.add_argument("--title", type=str, default="GRPO Training Curves", help="Plot title")
    parser.add_argument("--decay", type=float, default=0.99, help="EMA smoothing decay factor (default: 0.99)")
    args = parser.parse_args()

    if not args.tb_run_dir.is_dir():
        raise SystemExit(f"Error: {args.tb_run_dir} is not a directory")

    output = args.output or args.tb_run_dir / "training_curves.png"

    print(f"Loading events from {args.tb_run_dir}...")
    ea = EventAccumulator(str(args.tb_run_dir))
    ea.Reload()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    for ax, (tag, title, color) in zip(axes.flat, PANELS):
        if tag not in ea.Tags()["scalars"]:
            ax.set_visible(False)
            continue
        steps, vals = get_scalar(ea, tag)
        ax.plot(steps, vals, color=color, alpha=0.25, linewidth=0.8)
        ax.plot(steps, ema(vals, args.decay), color=color, linewidth=2.0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
