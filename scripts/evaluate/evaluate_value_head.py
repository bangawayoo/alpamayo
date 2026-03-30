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
from alpamayo_r1.training.sft_rollout import _compute_batch_logprobs, _extract_segment_hidden
from alpamayo_r1.training.value_head import SceneValueHead, SegmentValueHead

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

    valid = in_front.flatten().copy()
    pixels = pixels.reshape(-1, 2) if pixels.ndim > 2 else pixels
    for i in range(len(valid)):
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



def compute_segment_values(
    model: AlpamayoR1,
    value_head: SceneValueHead,
    model_inputs: dict,
    completion_ids: list[int],
    device: str,
) -> dict:
    """Compute obs-level and trajectory-level value predictions.

    Runs a teacher-forced VLM forward to extract hidden states at segment
    boundaries, then evaluates the value head at both obs and traj levels.

    Returns:
        Dict with v_obs, v_traj_mean, v_traj_per_step, and segment_map.
    """
    from alpamayo_r1.inference import prepare_vlm_inputs

    input_ids, gen_kwargs = prepare_vlm_inputs(model, model_inputs)
    prompt_len = input_ids.shape[1]

    logprob_result = _compute_batch_logprobs(
        model, model_inputs, input_ids, [completion_ids],
        prompt_len, device, mini_batch_size=1, output_hidden_states=True,
    )
    _, batch_hidden = logprob_result
    hidden_states = batch_hidden[0]  # (1, 1+comp_len, D)

    seg_hidden, seg_map = _extract_segment_hidden(
        hidden_states, completion_ids,
        model.special_token_ids,
        model.config.traj_token_start_idx,
        model.config.traj_vocab_size,
    )

    vh_device = next(value_head.parameters()).device

    # Obs-level value
    h_obs = seg_hidden["h_obs"].to(vh_device)  # (1, D)
    with torch.no_grad():
        v_obs = value_head(h_obs, level=SegmentValueHead.LEVEL_OBS).item()

    # Trajectory-level values (h_traj is already curvature-only from _extract_segment_hidden)
    h_traj = seg_hidden["h_traj"]  # (T_curv, D)
    v_traj_mean = 0.0
    v_traj_per_step = []

    n_curv = h_traj.shape[0]
    if n_curv > 0:
        h_traj_curv = h_traj.to(vh_device)
        traj_pos = (torch.arange(n_curv) + SegmentValueHead.POS_TRAJ_START).to(vh_device)
        with torch.no_grad():
            v_traj = value_head(
                h_traj_curv, level=SegmentValueHead.LEVEL_TRAJ, positions=traj_pos,
            )
        v_traj_per_step = v_traj.cpu().tolist()
        v_traj_mean = float(v_traj.mean().item())

    return {
        "v_obs": v_obs,
        "v_traj_mean": v_traj_mean,
        "v_traj_per_step": v_traj_per_step,
        "segment_map": seg_map,
    }


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
    from alpamayo_r1.inference import generate_coc, prepare_vlm_inputs
    from alpamayo_r1.models.alpamayo_r1 import decode_vlm_trajectories
    from alpamayo_r1.models.token_utils import extract_text_tokens

    try:
        data = load_physical_aiavdataset(
            clip_id=clip_id, t0_us=t0_us, avdi=avdi, maybe_stream=True,
        )
    except Exception as e:
        print(f"  data load failed: {e}")
        return None

    try:
        model_inputs = helper.prepare_model_inputs(data, processor, device)
        input_ids, gen_kwargs = prepare_vlm_inputs(model, model_inputs)
        prompt_len = input_ids.shape[1]

        # 1. Run VLM generation (captures raw sequences for value head)
        max_new_tokens = 256 + model.config.tokens_per_future_traj + 10
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            vlm_output = generate_coc(
                model, input_ids, gen_kwargs,
                mode="vlm", temperature=temperature, top_p=top_p,
                num_samples=num_traj_samples,
                max_new_tokens=max_new_tokens,
                pad_token_id=model.tokenizer.pad_token_id,
            )

        # Decode trajectories — squeeze out the scene dim (1,1,T,3) → (1,T,3)
        # before repeating, so traj_tokenizer.decode gets (S, T, 3)
        ego_history_xyz = model_inputs["ego_history_xyz"].squeeze(0)  # (1, T, 3)
        ego_history_rot = model_inputs["ego_history_rot"].squeeze(0)  # (1, T, 3, 3)
        hist_xyz_rep = ego_history_xyz.repeat(num_traj_samples, 1, 1)
        hist_rot_rep = ego_history_rot.repeat(num_traj_samples, 1, 1, 1)
        pred_xyz, pred_rot = decode_vlm_trajectories(
            model, vlm_output, hist_xyz_rep, hist_rot_rep,
        )
        pred_xyz = pred_xyz.unsqueeze(0).unsqueeze(0)  # (1, 1, S, T, 3)

        # Extract CoC text
        extra = extract_text_tokens(model.tokenizer, vlm_output)

        # 2. Compute segment-level values from first completion
        first_gen_ids = vlm_output[0, prompt_len:].cpu().tolist()
        # Strip padding
        pad_id = model.tokenizer.pad_token_id
        while first_gen_ids and first_gen_ids[-1] == pad_id:
            first_gen_ids.pop()

        seg_values = compute_segment_values(
            model, value_head, model_inputs, first_gen_ids, device,
        )
        v_obs = seg_values["v_obs"]
        v_traj_mean = seg_values["v_traj_mean"]

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

    traj_r_scalar = float(np.mean(traj_r))
    reason_mean = float(np.mean(reason_r))
    consist_mean = float(np.mean(consist_r))
    composite = (
        TRAJ_WEIGHT * traj_r_scalar
        + REASONING_WEIGHT * reason_mean
        + CONSISTENCY_WEIGHT * consist_mean
    )

    # 4. Compute return targets G(s_obs) and G(s_traj) to match training
    # G(s_obs) = w_reason * r_reason + sum(w_traj * r_per_step_j + w_consist * r_consist)
    # G(s_traj_j) = cumulative reward from position j to end
    T_traj = seg_values["segment_map"]["traj_len"]
    if T_traj > 0:
        # Per-waypoint displacement-relative rewards, matching training
        from alpamayo_r1.training.rewards import trajectory_per_timestep_rewards

        r_per_step_arrs = [
            trajectory_per_timestep_rewards(p, gt_flat)
            for p in per_sample_preds
        ]
        valid_arrs = [a for a in r_per_step_arrs if a is not None]
        r_per_step_t = torch.tensor(
            np.mean(valid_arrs, axis=0), dtype=torch.float32
        )
        # Normalize to match training: training divides r_per_step by count
        # so that sum(r_per_step_t) ≈ mean per-step reward (not total).
        r_per_step_t = r_per_step_t / len(r_per_step_t)
        # r_per_step_t has one entry per waypoint. Each waypoint corresponds
        # to one curvature token (the value head evaluation point), so
        # returns-to-go over waypoints align 1:1 with curvature positions.
        r_traj_weighted = TRAJ_WEIGHT * r_per_step_t
        r_traj_weighted[-1] = r_traj_weighted[-1] + CONSISTENCY_WEIGHT * consist_mean
        g_traj_all = torch.flip(
            torch.cumsum(torch.flip(r_traj_weighted, [0]), dim=0), [0]
        )
        g_traj_curv = g_traj_all.tolist()
        g_traj_mean = float(np.mean(g_traj_curv)) if g_traj_curv else 0.0
        traj_total = r_traj_weighted.sum().item()
    else:
        g_traj_curv = []
        g_traj_mean = 0.0
        traj_total = 0.0

    g_obs = REASONING_WEIGHT * reason_mean + traj_total

    # Meta-action summary from ground truth
    gt_summary = extract_meta_actions_summary(gt)

    return {
        "clip_id": clip_id,
        "v_obs": v_obs,
        "g_obs": g_obs,
        "v_traj_mean": v_traj_mean,
        "g_traj_mean": g_traj_mean,
        "v_traj_per_step": seg_values["v_traj_per_step"],
        "g_traj_per_step": g_traj_curv,
        "composite_reward": composite,
        "traj_reward": traj_r_scalar,
        "reasoning_reward": reason_mean,
        "consistency_reward": consist_mean,
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
    v_obs = result["v_obs"]
    g_obs = result["g_obs"]
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
    obs_residual = v_obs - g_obs
    color = "#44BB44" if abs(obs_residual) < 0.15 else (
        "#FF8844" if abs(obs_residual) < 0.3 else "#FF4444"
    )
    value_text = (
        f"V_obs:  {v_obs:.3f}\n"
        f"G_obs:  {g_obs:.3f}\n"
        f"res:    {obs_residual:+.3f}"
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
        f"v_obs:   {result['v_obs']:.3f}  G_obs:  {result['g_obs']:.3f}\n"
        f"v_traj:  {result['v_traj_mean']:.3f}  G_traj: {result['g_traj_mean']:.3f}\n"
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
        f"V_obs={v_obs:.3f} G_obs={g_obs:.3f}  |  {summary.longitudinal} + {summary.lateral}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Correlation scatter plot
# ---------------------------------------------------------------------------


def _traj_per_step_stats(results: list[dict]) -> dict:
    """Compute per-step traj V vs G statistics with early/mid/late breakdown.

    Pools per-step values across all samples and splits curvature positions
    into three equal-sized bins (early/mid/late in the trajectory).
    """
    all_v = []
    all_g = []
    all_pos = []
    for r in results:
        v_steps = r["v_traj_per_step"]
        g_steps = r["g_traj_per_step"]
        if not v_steps or not g_steps:
            continue
        n = min(len(v_steps), len(g_steps))
        all_v.extend(v_steps[:n])
        all_g.extend(g_steps[:n])
        all_pos.extend(range(n))

    if len(all_v) < 3:
        return {"per_step_n": len(all_v)}

    all_v = np.array(all_v)
    all_g = np.array(all_g)
    all_pos = np.array(all_pos)
    n_curv = int(all_pos.max()) + 1

    # Overall per-step correlation
    overall_r, overall_p = stats.pearsonr(all_v, all_g)
    overall_mse = float(np.mean((all_v - all_g) ** 2))

    # Split into first step, early (first third), mid, late (last third)
    third = max(1, n_curv // 3)
    bins = {
        "first": all_pos == 0,
        "early": all_pos < third,
        "mid": (all_pos >= third) & (all_pos < 2 * third),
        "late": all_pos >= 2 * third,
    }

    bin_stats = {}
    for name, mask in bins.items():
        if mask.sum() < 2:
            bin_stats[name] = {"n": int(mask.sum()), "r": 0.0, "mse": 0.0,
                               "v_mean": 0.0, "g_mean": 0.0}
            continue
        bv, bg = all_v[mask], all_g[mask]
        r_val, _ = stats.pearsonr(bv, bg)
        bin_stats[name] = {
            "n": int(mask.sum()),
            "r": float(r_val),
            "mse": float(np.mean((bv - bg) ** 2)),
            "v_mean": float(bv.mean()),
            "g_mean": float(bg.mean()),
        }

    return {
        "per_step_n": len(all_v),
        "per_step_pearson_r": float(overall_r),
        "per_step_pearson_p": float(overall_p),
        "per_step_mse": overall_mse,
        "n_curv_positions": n_curv,
        "bins": bin_stats,
        "all_v": all_v,
        "all_g": all_g,
        "all_pos": all_pos,
    }


def plot_correlation(results: list[dict], output_path: Path):
    """Scatter plots of V vs G at obs and traj levels with per-step breakdown."""
    v_obs = [r["v_obs"] for r in results]
    g_obs = [r["g_obs"] for r in results]
    v_traj = [r["v_traj_mean"] for r in results]
    g_traj = [r["g_traj_mean"] for r in results]

    obs_pearson, obs_p = stats.pearsonr(v_obs, g_obs)
    obs_spearman, _ = stats.spearmanr(v_obs, g_obs)
    obs_mse = float(np.mean([(v - g) ** 2 for v, g in zip(v_obs, g_obs)]))

    has_traj = any(v != 0.0 or g != 0.0 for v, g in zip(v_traj, g_traj))
    if has_traj:
        traj_pearson, traj_p = stats.pearsonr(v_traj, g_traj)
        traj_spearman, _ = stats.spearmanr(v_traj, g_traj)
        traj_mse = float(np.mean([(v - g) ** 2 for v, g in zip(v_traj, g_traj)]))
    else:
        traj_pearson = traj_mse = 0.0

    traj_step_stats = _traj_per_step_stats(results)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Panel 1: V_obs vs G_obs ---
    ax = axes[0, 0]
    ax.scatter(g_obs, v_obs, c="#4488FF", s=60, alpha=0.7, edgecolors="white", linewidths=0.5)
    lims = [
        min(min(g_obs), min(v_obs)) - 0.05,
        max(max(g_obs), max(v_obs)) + 0.05,
    ]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="Perfect prediction")
    if len(g_obs) > 2:
        slope, intercept = np.polyfit(g_obs, v_obs, 1)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, slope * x_fit + intercept, "-", color="red", alpha=0.7, label="Linear fit")
    ax.set_xlabel("G(s_obs)", fontsize=11)
    ax.set_ylabel("V(s_obs)", fontsize=11)
    ax.set_title("Obs-Level Calibration", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.03, 0.97,
        f"Pearson r:  {obs_pearson:.3f} (p={obs_p:.2e})\n"
        f"Spearman r: {obs_spearman:.3f}\n"
        f"MSE:        {obs_mse:.4f}\nN:          {len(results)}",
        transform=ax.transAxes, fontsize=9, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    # --- Panel 2: V_traj vs G_traj (per-step, colored by position) ---
    ax2 = axes[0, 1]
    if "all_v" in traj_step_stats:
        sc = ax2.scatter(
            traj_step_stats["all_g"], traj_step_stats["all_v"],
            c=traj_step_stats["all_pos"], cmap="viridis",
            s=30, alpha=0.6, edgecolors="white", linewidths=0.3,
        )
        plt.colorbar(sc, ax=ax2, label="Curvature position (0=early)")
        all_vals = np.concatenate([traj_step_stats["all_g"], traj_step_stats["all_v"]])
        lims2 = [all_vals.min() - 0.05, all_vals.max() + 0.05]
        ax2.plot(lims2, lims2, "--", color="gray", alpha=0.5)
        ax2.text(
            0.03, 0.97,
            f"Per-step r: {traj_step_stats['per_step_pearson_r']:.3f}\n"
            f"MSE:        {traj_step_stats['per_step_mse']:.4f}\n"
            f"N:          {traj_step_stats['per_step_n']}",
            transform=ax2.transAxes, fontsize=9, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
        )
    ax2.set_xlabel("G(s_traj) per step", fontsize=11)
    ax2.set_ylabel("V(s_traj) per step", fontsize=11)
    ax2.set_title("Traj Per-Step Calibration", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Early/mid/late breakdown bar chart ---
    ax3 = axes[1, 0]
    if "bins" in traj_step_stats:
        bin_names = ["first", "early", "mid", "late"]
        bin_labels = []
        bin_r = []
        bin_mse = []
        bin_v_means = []
        bin_g_means = []
        for name in bin_names:
            b = traj_step_stats["bins"].get(name, {})
            bin_labels.append(f"{name}\n(n={b.get('n', 0)})")
            bin_r.append(b.get("r", 0.0))
            bin_mse.append(b.get("mse", 0.0))
            bin_v_means.append(b.get("v_mean", 0.0))
            bin_g_means.append(b.get("g_mean", 0.0))

        x = np.arange(len(bin_names))
        w = 0.35
        ax3.bar(x - w / 2, bin_g_means, w, label="G (target)", color="#4488FF", alpha=0.7)
        ax3.bar(x + w / 2, bin_v_means, w, label="V (predicted)", color="#FF4444", alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(bin_labels)
        ax3.set_ylabel("Mean value", fontsize=11)
        ax3.set_title("V vs G by Trajectory Phase", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis="y")

        # Annotate with correlation
        for i, name in enumerate(bin_names):
            b = traj_step_stats["bins"].get(name, {})
            ax3.text(
                i, max(bin_g_means[i], bin_v_means[i]) + 0.02,
                f"r={b.get('r', 0):.2f}", ha="center", fontsize=8,
            )

    # --- Panel 4: Residual histograms ---
    ax4 = axes[1, 1]
    obs_residuals = [v - g for v, g in zip(v_obs, g_obs)]
    ax4.hist(obs_residuals, bins=15, color="#4488FF", alpha=0.7, edgecolor="white", label="Obs")
    if has_traj:
        traj_residuals = [v - g for v, g in zip(v_traj, g_traj)]
        ax4.hist(traj_residuals, bins=15, color="#FF4444", alpha=0.5, edgecolor="white", label="Traj (mean)")
    ax4.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Residual (V - G)", fontsize=11)
    ax4.set_ylabel("Count", fontsize=11)
    ax4.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        f"Value Head Evaluation  |  Obs r={obs_pearson:.3f}  "
        f"Traj per-step r={traj_step_stats.get('per_step_pearson_r', 0):.3f}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "obs_pearson_r": obs_pearson,
        "obs_pearson_p": obs_p,
        "obs_spearman_r": obs_spearman,
        "obs_mse": obs_mse,
        "obs_residual_mean": float(np.mean(obs_residuals)),
        "obs_residual_std": float(np.std(obs_residuals)),
        "traj_mean_pearson_r": traj_pearson,
        "traj_mean_mse": traj_mse,
        "traj_per_step": {
            k: v for k, v in traj_step_stats.items()
            if k not in ("all_v", "all_g", "all_pos")
        },
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
    parser.add_argument("--output-dir", default="eval_results/value_head_eval")
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
            f"  obs: V={result['v_obs']:.3f} G={result['g_obs']:.3f}  "
            f"traj: V={result['v_traj_mean']:.3f} G={result['g_traj_mean']:.3f}  "
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
            / f"{i:02d}_Vobs{result['v_obs']:.2f}_Gobs{result['g_obs']:.2f}"
            f"_{result['gt_summary_lon']}_{result['gt_summary_lat']}"
            f"_{clip_id[:8]}.png"
        )
        try:
            plot_sample_with_value(result, clip_cam_model, clip_cam_pose, out_path)
            tqdm.write(f"  -> {out_path.name}")
        except Exception as e:
            tqdm.write(f"  plot failed: {e}")

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
    print("  --- Obs level ---")
    print(f"  Pearson r:         {corr_stats['obs_pearson_r']:.4f}  (p={corr_stats['obs_pearson_p']:.2e})")
    print(f"  Spearman r:        {corr_stats['obs_spearman_r']:.4f}")
    print(f"  MSE:               {corr_stats['obs_mse']:.4f}")
    print(f"  Residual mean:     {corr_stats['obs_residual_mean']:+.4f}")
    print(f"  Residual std:      {corr_stats['obs_residual_std']:.4f}")
    print("  --- Traj level (per-sample mean) ---")
    print(f"  Pearson r:         {corr_stats['traj_mean_pearson_r']:.4f}")
    print(f"  MSE:               {corr_stats['traj_mean_mse']:.4f}")
    tps = corr_stats.get("traj_per_step", {})
    if "per_step_pearson_r" in tps:
        print("  --- Traj level (per-step, pooled) ---")
        print(f"  Pearson r:         {tps['per_step_pearson_r']:.4f}")
        print(f"  MSE:               {tps['per_step_mse']:.4f}")
        print(f"  N points:          {tps['per_step_n']}")
        if "bins" in tps:
            for phase in ("first", "early", "mid", "late"):
                b = tps["bins"].get(phase, {})
                print(
                    f"    {phase:5s}: r={b.get('r', 0):.3f}  "
                    f"MSE={b.get('mse', 0):.4f}  "
                    f"V_mean={b.get('v_mean', 0):.3f}  "
                    f"G_mean={b.get('g_mean', 0):.3f}  "
                    f"(n={b.get('n', 0)})"
                )

    # Save JSON results (strip non-serializable fields)
    json_results = []
    for r in results:
        json_results.append({
            "clip_id": r["clip_id"],
            "v_obs": r["v_obs"],
            "g_obs": r["g_obs"],
            "v_traj_mean": r["v_traj_mean"],
            "g_traj_mean": r["g_traj_mean"],
            "v_traj_per_step": r["v_traj_per_step"],
            "g_traj_per_step": r["g_traj_per_step"],
            "composite_reward": r["composite_reward"],
            "traj_reward": r["traj_reward"],
            "reasoning_reward": r["reasoning_reward"],
            "consistency_reward": r["consistency_reward"],
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
