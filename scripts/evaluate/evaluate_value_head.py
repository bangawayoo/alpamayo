#!/usr/bin/env python3
"""Qualitative evaluation of the SceneValueHead.

For each sampled clip, runs VLM inference to:
  1. Extract h0 (last-prompt hidden state) and predict V(scene)
  2. Generate trajectories and compute actual composite reward
  3. Produce a two-panel visualization (camera + BEV) annotated with
     predicted value vs actual reward

After all samples, produces a correlation scatter plot and prints
summary statistics (Pearson/Spearman correlation, MSE).

Usage:
    python scripts/evaluate/evaluate_value_head.py \
        --value-head-path checkpoints/value_head.pt \
        --num-samples 12
    python scripts/evaluate/evaluate_value_head.py \
        --value-head-path checkpoints/value_head.pt \
        --num-samples 20 --seed 99 --output-dir outputs/value_head_eval
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from physical_ai_av import PhysicalAIAVDatasetInterface
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.training.meta_actions import (
    LAT_GO_STRAIGHT,
    LAT_REVERSE_LEFT,
    LAT_REVERSE_RIGHT,
    LAT_SHARP_LEFT,
    LAT_SHARP_RIGHT,
    LAT_STEER_LEFT,
    LAT_STEER_RIGHT,
    extract_meta_actions,
    extract_meta_actions_summary,
)
from alpamayo_r1.training.rewards import (
    consistency_reward,
    reasoning_quality_reward,
    trajectory_quality_reward,
)
from alpamayo_r1.training.value_head import SceneValueHead

# Reward weights (from grpo_default.yaml)
TRAJ_WEIGHT = 0.50
REASONING_WEIGHT = 0.25
CONSISTENCY_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Projection helpers (shared with visualize_meta_actions.py)
# ---------------------------------------------------------------------------


def project_ego_to_pixels(
    traj_ego: np.ndarray,
    cam_pose,
    cam_model,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-frame 3D points to 2D pixel coordinates."""
    p_cam = cam_pose.inv().apply(traj_ego)
    z = p_cam[:, 2]
    in_front = z > 0.5

    pixels = cam_model.ray2pixel(p_cam)

    valid = in_front.copy()
    for i in range(len(pixels)):
        if valid[i] and cam_model.is_out_of_bounds(pixels[i]):
            valid[i] = False

    return pixels, valid


def get_camera_image(data: dict, cam_idx: int = 1) -> np.ndarray:
    """Extract the last frame from a specific camera as HWC uint8 numpy array."""
    cam_indices = data["camera_indices"].numpy()
    slot = int(np.where(cam_indices == cam_idx)[0][0])
    frame = data["image_frames"][slot, -1]
    return frame.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Color mapping for lateral meta-actions
# ---------------------------------------------------------------------------

_LAT_COLORS = {
    LAT_SHARP_LEFT: "#FF4444",
    LAT_STEER_LEFT: "#FF8844",
    LAT_GO_STRAIGHT: "#44BB44",
    LAT_STEER_RIGHT: "#4488FF",
    LAT_SHARP_RIGHT: "#8844FF",
    LAT_REVERSE_LEFT: "#FF44AA",
    LAT_REVERSE_RIGHT: "#AA44FF",
}

_LAT_ORDER = [
    LAT_SHARP_LEFT, LAT_STEER_LEFT, LAT_GO_STRAIGHT,
    LAT_STEER_RIGHT, LAT_SHARP_RIGHT, LAT_REVERSE_LEFT, LAT_REVERSE_RIGHT,
]


def _lat_to_color_array(labels: list[str]) -> list[str]:
    return [_LAT_COLORS.get(l, "#888888") for l in labels]


# ---------------------------------------------------------------------------
# h0 extraction
# ---------------------------------------------------------------------------


def compute_scene_h0(
    model: AlpamayoR1,
    model_inputs: dict,
    device: str,
) -> torch.Tensor:
    """Extract VLM hidden state at the last prompt token.

    Mirrors the training pipeline: fuses history trajectory tokens into
    the prompt before running the VLM forward, then takes the hidden
    state at the last prompt position.

    Returns:
        h0: shape (1, hidden_dim), float32, on CPU.
    """
    from alpamayo_r1.inference import prepare_vlm_inputs

    input_ids, gen_kwargs = prepare_vlm_inputs(model, model_inputs)

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        outputs = model.vlm(
            input_ids=input_ids,
            output_hidden_states=True,
            **gen_kwargs,
        )

    h0 = outputs.hidden_states[-1][:, -1, :].float().cpu()
    return h0


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------


def evaluate_sample(
    model: AlpamayoR1,
    value_head: SceneValueHead,
    processor,
    avdi,
    clip_id: str,
    t0_us: int,
    num_traj_samples: int,
    temperature: float,
    top_p: float,
    ade_threshold: float,
    device: str,
) -> dict | None:
    """Run inference on one clip, compute value prediction and actual reward."""
    try:
        data = load_physical_aiavdataset(
            clip_id=clip_id, t0_us=t0_us, avdi=avdi, maybe_stream=True,
        )
    except Exception as e:
        print(f"  data load failed: {e}")
        return None

    try:
        model_inputs = helper.prepare_model_inputs(data, processor, device)

        # 1. Extract h0 and predict value
        h0 = compute_scene_h0(model, model_inputs, device)
        with torch.no_grad():
            v_pred = value_head(h0.to(value_head.net[0].weight.device)).item()

        # 2. Run VLM inference
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
        print(f"  inference failed: {e}")
        return None

    # 3. Compute actual rewards
    pred_samples = pred_xyz.cpu().numpy()[0, 0]  # (S, T, 3)
    gt = data["ego_future_xyz"].cpu().numpy()[0, 0]  # (T, 3)

    pred_per_sample = [pred_samples[i].flatten().tolist() for i in range(pred_samples.shape[0])]
    gt_flat = gt.flatten().tolist()

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

    n_texts = len(coc_texts) if coc_texts else 1
    per_sample_preds = pred_per_sample[:n_texts]

    traj_r = trajectory_quality_reward(
        completions=[""] * n_texts,
        pred_xyz=per_sample_preds,
        gt_xyz=[gt_flat] * n_texts,
        ade_threshold=ade_threshold,
    )
    reason_r = reasoning_quality_reward(completions=coc_texts) if coc_texts else [0.0]
    consist_r = consistency_reward(
        completions=coc_texts if coc_texts else [""],
        pred_xyz=per_sample_preds,
    )

    traj_mean = float(np.mean(traj_r))
    reason_mean = float(np.mean(reason_r))
    consist_mean = float(np.mean(consist_r))
    composite = (
        TRAJ_WEIGHT * traj_mean
        + REASONING_WEIGHT * reason_mean
        + CONSISTENCY_WEIGHT * consist_mean
    )

    # Meta-action summary from ground truth
    gt_summary = extract_meta_actions_summary(gt)

    return {
        "clip_id": clip_id,
        "v_pred": v_pred,
        "composite_reward": composite,
        "traj_reward": traj_mean,
        "reasoning_reward": reason_mean,
        "consistency_reward": consist_mean,
        "residual": v_pred - composite,
        "gt_summary_lon": gt_summary.longitudinal,
        "gt_summary_lat": gt_summary.lateral,
        "pred_samples_np": pred_samples,
        "gt_np": gt,
        "data": data,
        "coc_text": coc_texts[0] if coc_texts else "",
    }


# ---------------------------------------------------------------------------
# Plotting: per-sample two-panel figure
# ---------------------------------------------------------------------------


def plot_sample_with_value(
    result: dict,
    cam_model,
    cam_pose,
    output_path: Path,
):
    """Two-panel figure: camera view + BEV, annotated with V(scene) vs reward."""
    clip_id = result["clip_id"]
    data = result["data"]
    gt = result["gt_np"]
    pred = result["pred_samples_np"][0]  # best/first sample for visualization
    v_pred = result["v_pred"]
    composite = result["composite_reward"]
    residual = result["residual"]
    T = gt.shape[0]

    meta = extract_meta_actions(gt)
    summary = extract_meta_actions_summary(gt)

    lat_labels_full = [meta.lateral[0]] + meta.lateral + [meta.lateral[-1]]
    lat_colors = _lat_to_color_array(lat_labels_full)

    img = get_camera_image(data, cam_idx=1)
    gt_pixels, gt_valid = project_ego_to_pixels(gt, cam_pose, cam_model)
    pred_pixels, pred_valid = project_ego_to_pixels(pred, cam_pose, cam_model)
    history = data["ego_history_xyz"].cpu().numpy()[0, 0]
    hist_pixels, hist_valid = project_ego_to_pixels(history, cam_pose, cam_model)

    fig, (ax_img, ax_bev) = plt.subplots(
        1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # --- Left panel: Camera image ---
    ax_img.imshow(img)

    # GT trajectory on image (colored by lateral action)
    if gt_valid.any():
        px_valid = gt_pixels[gt_valid]
        colors_valid = [lat_colors[i] for i in range(T) if gt_valid[i]]
        ax_img.plot(
            px_valid[:, 0], px_valid[:, 1],
            color="white", alpha=0.5, linewidth=3, zorder=4,
        )
        for j in range(len(px_valid)):
            ax_img.scatter(
                px_valid[j, 0], px_valid[j, 1],
                c=colors_valid[j], s=50, zorder=5,
                edgecolors="white", linewidths=0.3,
            )

    # Predicted trajectory on image (dashed yellow)
    if pred_valid.any():
        ppx = pred_pixels[pred_valid]
        ax_img.plot(
            ppx[:, 0], ppx[:, 1],
            color="yellow", alpha=0.7, linewidth=2, linestyle="--", zorder=6,
        )

    # History in gray
    if hist_valid.any():
        hpx = hist_pixels[hist_valid]
        ax_img.plot(hpx[:, 0], hpx[:, 1], color="gray", alpha=0.6, linewidth=2.5, zorder=3)

    # Value annotation box
    color = "#44BB44" if abs(residual) < 0.15 else ("#FF8844" if abs(residual) < 0.3 else "#FF4444")
    value_text = (
        f"V(scene): {v_pred:.3f}\n"
        f"Actual:   {composite:.3f}\n"
        f"Residual: {residual:+.3f}"
    )
    ax_img.text(
        0.02, 0.98, value_text, transform=ax_img.transAxes,
        fontsize=12, fontweight="bold", color="white",
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.85),
    )

    ax_img.set_title(f"Camera View  |  {clip_id[:16]}...", fontsize=10)
    ax_img.axis("off")

    # --- Right panel: Bird's-eye view ---
    def ego_to_bev(pts):
        return -pts[:, 1], pts[:, 0]

    # History
    hx, hy = ego_to_bev(history)
    ax_bev.plot(hx, hy, "o-", color="gray", markersize=3, alpha=0.5, label="History")

    # GT trajectory colored by lateral meta-action
    fx, fy = ego_to_bev(gt)
    ax_bev.plot(fx, fy, color="#CCCCCC", alpha=0.4, linewidth=1.5, zorder=4)
    for j in range(T):
        ax_bev.scatter(
            fx[j], fy[j], c=lat_colors[j], s=40,
            zorder=5, edgecolors="white", linewidths=0.3,
        )

    # Predicted trajectory (dashed yellow)
    px_bev, py_bev = ego_to_bev(pred)
    ax_bev.plot(px_bev, py_bev, "--", color="gold", linewidth=2, alpha=0.8, label="Predicted", zorder=6)

    # Ego marker
    ax_bev.plot(0, 0, "s", color="red", markersize=10, zorder=10, label="Ego (t0)")
    ax_bev.annotate(
        "", xy=(0, 3), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )

    ax_bev.set_xlabel("Lateral (m)  [left +]", fontsize=10)
    ax_bev.set_ylabel("Forward (m)", fontsize=10)
    ax_bev.set_title("Bird's-Eye View (ego frame)", fontsize=10)

    # Axis limits
    all_bev_x = np.concatenate([hx, fx, px_bev])
    all_bev_y = np.concatenate([hy, fy, py_bev])
    x_range = all_bev_x.max() - all_bev_x.min()
    y_range = all_bev_y.max() - all_bev_y.min()
    min_x_half = max(5.0, y_range * 0.2)
    x_center = (all_bev_x.max() + all_bev_x.min()) / 2
    if x_range < min_x_half * 2:
        ax_bev.set_xlim(x_center - min_x_half, x_center + min_x_half)
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.3)

    # Legend
    present_labels = set(lat_labels_full)
    legend_handles = [ax_bev.plot([], [], "s", color="red", markersize=8)[0]]
    legend_labels = ["Ego (t0)"]
    legend_handles.append(ax_bev.plot([], [], "o-", color="gray", markersize=4)[0])
    legend_labels.append("History")
    legend_handles.append(ax_bev.plot([], [], "--", color="gold", linewidth=2)[0])
    legend_labels.append("Predicted")
    for lat_label in _LAT_ORDER:
        if lat_label in present_labels:
            legend_handles.append(Patch(facecolor=_LAT_COLORS[lat_label], edgecolor="white"))
            legend_labels.append(f"GT: {lat_label}")
    ax_bev.legend(legend_handles, legend_labels, loc="lower left", fontsize=7)

    # Stats box on BEV
    stats_text = (
        f"V(scene):  {v_pred:.3f}\n"
        f"Actual:    {composite:.3f}\n"
        f"Residual:  {residual:+.3f}\n"
        f"\n"
        f"traj_r:    {result['traj_reward']:.3f}\n"
        f"reason_r:  {result['reasoning_reward']:.3f}\n"
        f"consist_r: {result['consistency_reward']:.3f}\n"
        f"\n"
        f"LON: {summary.longitudinal}\n"
        f"LAT: {summary.lateral}"
    )
    ax_bev.text(
        0.98, 0.98, stats_text, transform=ax_bev.transAxes,
        fontsize=8, fontfamily="monospace", verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    fig.suptitle(
        f"V={v_pred:.3f}  vs  R={composite:.3f}  |  {summary.longitudinal} + {summary.lateral}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Correlation scatter plot
# ---------------------------------------------------------------------------


def plot_correlation(results: list[dict], output_path: Path):
    """Scatter plot of V(scene) vs actual composite reward with regression line."""
    v_preds = [r["v_pred"] for r in results]
    actuals = [r["composite_reward"] for r in results]

    pearson_r, pearson_p = stats.pearsonr(v_preds, actuals)
    spearman_r, spearman_p = stats.spearmanr(v_preds, actuals)
    mse = float(np.mean([(v - a) ** 2 for v, a in zip(v_preds, actuals)]))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Panel 1: V(scene) vs Composite Reward ---
    ax = axes[0]
    ax.scatter(actuals, v_preds, c="#4488FF", s=60, alpha=0.7, edgecolors="white", linewidths=0.5)

    # Perfect prediction line
    lims = [
        min(min(actuals), min(v_preds)) - 0.05,
        max(max(actuals), max(v_preds)) + 0.05,
    ]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="Perfect prediction")

    # Regression line
    if len(actuals) > 2:
        slope, intercept = np.polyfit(actuals, v_preds, 1)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, slope * x_fit + intercept, "-", color="red", alpha=0.7, label="Linear fit")

    ax.set_xlabel("Actual Composite Reward", fontsize=11)
    ax.set_ylabel("V(scene) Prediction", fontsize=11)
    ax.set_title("Value Head Calibration", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    stats_text = (
        f"Pearson r:  {pearson_r:.3f} (p={pearson_p:.2e})\n"
        f"Spearman r: {spearman_r:.3f} (p={spearman_p:.2e})\n"
        f"MSE:        {mse:.4f}\n"
        f"N:          {len(results)}"
    )
    ax.text(
        0.03, 0.97, stats_text, transform=ax.transAxes,
        fontsize=9, fontfamily="monospace", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    # --- Panel 2: Residual histogram ---
    ax2 = axes[1]
    residuals = [r["residual"] for r in results]
    ax2.hist(residuals, bins=15, color="#4488FF", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Residual (V - R)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    res_stats = (
        f"mean: {np.mean(residuals):+.3f}\n"
        f"std:  {np.std(residuals):.3f}"
    )
    ax2.text(
        0.97, 0.97, res_stats, transform=ax2.transAxes,
        fontsize=9, fontfamily="monospace", verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    # --- Panel 3: V(scene) vs individual reward components ---
    ax3 = axes[2]
    traj_vals = [r["traj_reward"] for r in results]
    reason_vals = [r["reasoning_reward"] for r in results]
    consist_vals = [r["consistency_reward"] for r in results]

    ax3.scatter(traj_vals, v_preds, c="#FF4444", s=40, alpha=0.6, label=f"Traj (w={TRAJ_WEIGHT})")
    ax3.scatter(reason_vals, v_preds, c="#44BB44", s=40, alpha=0.6, label=f"Reason (w={REASONING_WEIGHT})")
    ax3.scatter(consist_vals, v_preds, c="#4488FF", s=40, alpha=0.6, label=f"Consist (w={CONSISTENCY_WEIGHT})")
    ax3.set_xlabel("Component Reward", fontsize=11)
    ax3.set_ylabel("V(scene) Prediction", fontsize=11)
    ax3.set_title("V(scene) vs Reward Components", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"Value Head Evaluation  |  Pearson r={pearson_r:.3f}  |  MSE={mse:.4f}",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "mse": mse,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Qualitative evaluation of SceneValueHead")
    parser.add_argument("--value-head-path", required=True,
                        help="Path to saved value head weights (.pt)")
    parser.add_argument("--model-name", default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--num-traj-samples", type=int, default=4,
                        help="Trajectory samples per clip for reward estimation")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--ade-threshold", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=4096,
                        help="VLM hidden state dimension (must match value head)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset-revision", default="05e158af89ba",
                        help="HuggingFace dataset revision (default: 05e158af89ba)")
    parser.add_argument("--output-dir", default="outputs/value_head_eval")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running on CPU, this will be very slow.")

    # Load model
    print("Loading AlpamayoR1...")
    model = AlpamayoR1.from_pretrained(args.model_name, dtype=torch.bfloat16).to(device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # Load value head
    print(f"Loading value head from {args.value_head_path}...")
    value_head = SceneValueHead(hidden_dim=args.hidden_dim)
    state_dict = torch.load(args.value_head_path, map_location="cpu", weights_only=True)
    value_head.load_state_dict(state_dict)
    value_head.eval()
    value_head.to(device)
    print(f"  Value head loaded ({sum(p.numel() for p in value_head.parameters()):,} params)")

    # Sample clips
    print(f"Loading {args.split} clip index...")
    avdi = PhysicalAIAVDatasetInterface(revision=args.dataset_revision)
    print(f"  Dataset revision: {avdi.revision}")
    clip_index = avdi.clip_index
    split_df = clip_index[(clip_index["split"] == args.split) & clip_index["clip_is_valid"]]
    all_clips = split_df.index.tolist()
    print(f"  {len(all_clips)} valid {args.split} clips")

    rng = np.random.default_rng(args.seed)
    sampled = rng.choice(all_clips, size=min(args.num_samples, len(all_clips)), replace=False)
    print(f"  Sampling {len(sampled)} clips\n")

    # Get camera calibration for visualization
    print("Loading camera calibration from first clip...")
    first_intrinsics = avdi.get_clip_feature(
        sampled[0], avdi.features.CALIBRATION.CAMERA_INTRINSICS, maybe_stream=True,
    )
    first_extrinsics = avdi.get_clip_feature(
        sampled[0], avdi.features.CALIBRATION.SENSOR_EXTRINSICS, maybe_stream=True,
    )
    cam_model = first_intrinsics.camera_models["camera_front_wide_120fov"]
    cam_pose = first_extrinsics.sensor_poses["camera_front_wide_120fov"]

    # Evaluate all samples
    results = []
    for i, clip_id in enumerate(tqdm(sampled, desc="Evaluating")):
        tqdm.write(f"[{i+1}/{len(sampled)}] {clip_id}")

        result = evaluate_sample(
            model=model,
            value_head=value_head,
            processor=processor,
            avdi=avdi,
            clip_id=clip_id,
            t0_us=args.t0_us,
            num_traj_samples=args.num_traj_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            ade_threshold=args.ade_threshold,
            device=device,
        )
        if result is None:
            continue

        tqdm.write(
            f"  V={result['v_pred']:.3f}  R={result['composite_reward']:.3f}  "
            f"res={result['residual']:+.3f}  "
            f"({result['gt_summary_lon']} + {result['gt_summary_lat']})"
        )

        # Per-clip camera calibration for accurate projection
        try:
            clip_intrinsics = avdi.get_clip_feature(
                clip_id, avdi.features.CALIBRATION.CAMERA_INTRINSICS, maybe_stream=True,
            )
            clip_extrinsics = avdi.get_clip_feature(
                clip_id, avdi.features.CALIBRATION.SENSOR_EXTRINSICS, maybe_stream=True,
            )
            clip_cam_model = clip_intrinsics.camera_models["camera_front_wide_120fov"]
            clip_cam_pose = clip_extrinsics.sensor_poses["camera_front_wide_120fov"]
        except Exception:
            clip_cam_model, clip_cam_pose = cam_model, cam_pose

        # Plot individual sample
        out_path = (
            output_dir
            / f"{i:02d}_V{result['v_pred']:.2f}_R{result['composite_reward']:.2f}"
            f"_{result['gt_summary_lon']}_{result['gt_summary_lat']}"
            f"_{clip_id[:8]}.png"
        )
        plot_sample_with_value(result, clip_cam_model, clip_cam_pose, out_path)
        tqdm.write(f"  -> {out_path.name}")

        results.append(result)

    if len(results) < 2:
        print(f"\nOnly {len(results)} successful sample(s). Need >=2 for correlation analysis.")
        return

    # Correlation scatter plot
    corr_path = output_dir / "correlation_summary.png"
    corr_stats = plot_correlation(results, corr_path)
    print(f"\nCorrelation plot -> {corr_path.name}")

    # Print summary
    print("\n" + "=" * 60)
    print("  VALUE HEAD EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Samples evaluated: {len(results)}")
    print(f"  Pearson r:         {corr_stats['pearson_r']:.4f}  (p={corr_stats['pearson_p']:.2e})")
    print(f"  Spearman r:        {corr_stats['spearman_r']:.4f}  (p={corr_stats['spearman_p']:.2e})")
    print(f"  MSE:               {corr_stats['mse']:.4f}")
    print(f"  Residual mean:     {corr_stats['residual_mean']:+.4f}")
    print(f"  Residual std:      {corr_stats['residual_std']:.4f}")

    # Save JSON results (strip non-serializable fields)
    json_results = []
    for r in results:
        json_results.append({
            "clip_id": r["clip_id"],
            "v_pred": r["v_pred"],
            "composite_reward": r["composite_reward"],
            "traj_reward": r["traj_reward"],
            "reasoning_reward": r["reasoning_reward"],
            "consistency_reward": r["consistency_reward"],
            "residual": r["residual"],
            "gt_summary_lon": r["gt_summary_lon"],
            "gt_summary_lat": r["gt_summary_lat"],
            "coc_text": r["coc_text"],
        })

    json_path = output_dir / "value_head_eval.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "correlation": corr_stats,
                "per_sample": json_results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to {json_path}")
    print(f"  Figures saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
