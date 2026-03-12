#!/usr/bin/env python3
"""Visualize ground-truth trajectories with per-timestep meta-action labels.

For each sampled clip, produces a two-panel figure:
  Left:  Front-wide camera image at t0 with the trajectory projected onto it,
         colored by per-timestep lateral meta-action
  Right: Bird's-eye (top-down) view with dots colored by lateral meta-action
         and the action sequence annotated

Usage:
    python scripts/evaluate/visualize_meta_actions.py --num-samples 6
    python scripts/evaluate/visualize_meta_actions.py --num-samples 12 --seed 99 --output-dir outputs/meta_actions
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from physical_ai_av import PhysicalAIAVDatasetInterface

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
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


# ---------------------------------------------------------------------------
# Projection helpers
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
    LAT_SHARP_LEFT: "#FF4444",     # red
    LAT_STEER_LEFT: "#FF8844",     # orange
    LAT_GO_STRAIGHT: "#44BB44",    # green
    LAT_STEER_RIGHT: "#4488FF",    # blue
    LAT_SHARP_RIGHT: "#8844FF",    # purple
    LAT_REVERSE_LEFT: "#FF44AA",   # pink
    LAT_REVERSE_RIGHT: "#AA44FF",  # magenta
}

_LAT_ORDER = [
    LAT_SHARP_LEFT, LAT_STEER_LEFT, LAT_GO_STRAIGHT,
    LAT_STEER_RIGHT, LAT_SHARP_RIGHT, LAT_REVERSE_LEFT, LAT_REVERSE_RIGHT,
]


def _lat_to_color_array(labels: list[str]) -> list[str]:
    """Map lateral label list to color list."""
    return [_LAT_COLORS.get(l, "#888888") for l in labels]


def _ordered_unique(labels: list[str]) -> list[str]:
    """Return unique labels in order of first appearance."""
    seen: set[str] = set()
    result: list[str] = []
    for l in labels:
        if l not in seen:
            seen.add(l)
            result.append(l)
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sample(
    clip_id: str,
    data: dict,
    cam_model,
    cam_pose,
    output_path: Path | None = None,
):
    """Plot a two-panel visualization for one clip."""
    traj = data["ego_future_xyz"].cpu().numpy()[0, 0]  # (T, 3)
    history = data["ego_history_xyz"].cpu().numpy()[0, 0]  # (H, 3)
    T = traj.shape[0]

    meta = extract_meta_actions(traj)
    summary = extract_meta_actions_summary(traj)

    # Kinematics
    velocity = np.diff(traj[:, 0]) / 0.1
    v_start = float(np.median(velocity[:10]))
    v_end = float(np.median(velocity[-10:]))
    net_y = float(traj[-1, 1] - traj[0, 1])

    # Per-timestep labels are length T-2; pad to T for alignment with traj points
    # Pad start with first label, end with last label
    lat_labels_full = [meta.lateral[0]] + meta.lateral + [meta.lateral[-1]]
    lon_labels_full = [meta.longitudinal[0]] + meta.longitudinal + [meta.longitudinal[-1]]
    lat_colors = _lat_to_color_array(lat_labels_full)

    # Camera image
    img = get_camera_image(data, cam_idx=1)

    # Project trajectory to image
    pixels, valid = project_ego_to_pixels(traj, cam_pose, cam_model)
    hist_pixels, hist_valid = project_ego_to_pixels(history, cam_pose, cam_model)

    # --- Figure ---
    fig, (ax_img, ax_bev) = plt.subplots(
        1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # --- Left panel: Camera image with projected trajectory ---
    ax_img.imshow(img)

    if valid.any():
        px_valid = pixels[valid]
        colors_valid = [lat_colors[i] for i in range(T) if valid[i]]
        # Line in light gray behind dots
        ax_img.plot(
            px_valid[:, 0], px_valid[:, 1],
            color="white", alpha=0.5, linewidth=3, zorder=4,
        )
        # Dots colored by lateral action
        for j in range(len(px_valid)):
            ax_img.scatter(
                px_valid[j, 0], px_valid[j, 1],
                c=colors_valid[j], s=50, zorder=5,
                edgecolors="white", linewidths=0.3,
            )

    # History in gray
    if hist_valid.any():
        hpx = hist_pixels[hist_valid]
        ax_img.plot(hpx[:, 0], hpx[:, 1], color="gray", alpha=0.6, linewidth=2.5, zorder=3)
        ax_img.scatter(hpx[:, 0], hpx[:, 1], c="gray", s=25, alpha=0.6, zorder=3)

    # Summary label on image
    label_text = f"LON: {summary.longitudinal}\nLAT: {summary.lateral}"
    ax_img.text(
        0.02, 0.98, label_text, transform=ax_img.transAxes,
        fontsize=11, fontweight="bold", color="white",
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7),
    )

    ax_img.set_title(f"Camera View  |  {clip_id[:16]}...", fontsize=10)
    ax_img.axis("off")

    # --- Right panel: Bird's-eye view ---
    def ego_to_bev(pts):
        return -pts[:, 1], pts[:, 0]

    # History
    hx, hy = ego_to_bev(history)
    ax_bev.plot(hx, hy, "o-", color="gray", markersize=3, alpha=0.5, label="History")

    # Future trajectory colored by lateral meta-action
    fx, fy = ego_to_bev(traj)
    ax_bev.plot(fx, fy, color="#CCCCCC", alpha=0.4, linewidth=1.5, zorder=4)
    for j in range(T):
        ax_bev.scatter(
            fx[j], fy[j], c=lat_colors[j], s=40,
            zorder=5, edgecolors="white", linewidths=0.3,
        )

    # Ego vehicle marker at origin
    ax_bev.plot(0, 0, "s", color="red", markersize=10, zorder=10, label="Ego (t0)")

    # Direction arrow
    ax_bev.annotate(
        "", xy=(0, 3), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )

    ax_bev.set_xlabel("Lateral (m)  [left +]", fontsize=10)
    ax_bev.set_ylabel("Forward (m)", fontsize=10)
    ax_bev.set_title("Bird's-Eye View (ego frame)", fontsize=10)

    # Enforce minimum lateral range
    all_bev_x = np.concatenate([hx, fx])
    all_bev_y = np.concatenate([hy, fy])
    x_range = all_bev_x.max() - all_bev_x.min()
    y_range = all_bev_y.max() - all_bev_y.min()
    min_x_half = max(5.0, y_range * 0.2)
    x_center = (all_bev_x.max() + all_bev_x.min()) / 2
    if x_range < min_x_half * 2:
        ax_bev.set_xlim(x_center - min_x_half, x_center + min_x_half)

    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.3)

    # Legend: lateral action colors (only those present)
    present_labels = set(lat_labels_full)
    legend_handles = [ax_bev.plot([], [], "s", color="red", markersize=8)[0]]
    legend_labels = ["Ego (t0)"]
    legend_handles.append(ax_bev.plot([], [], "o-", color="gray", markersize=4)[0])
    legend_labels.append("History")
    for lat_label in _LAT_ORDER:
        if lat_label in present_labels:
            legend_handles.append(Patch(facecolor=_LAT_COLORS[lat_label], edgecolor="white"))
            legend_labels.append(lat_label)
    ax_bev.legend(legend_handles, legend_labels, loc="lower left", fontsize=7)

    # Stats + action sequence annotation
    lon_seq = " -> ".join(_ordered_unique(meta.longitudinal))
    lat_seq = " -> ".join(_ordered_unique(meta.lateral))
    stats_text = (
        f"v_start: {v_start:.1f} m/s\n"
        f"v_end:   {v_end:.1f} m/s\n"
        f"net_y:   {net_y:+.1f} m\n"
        f"\nLON: {lon_seq}\n"
        f"LAT: {lat_seq}"
    )
    ax_bev.text(
        0.98, 0.98, stats_text, transform=ax_bev.transAxes,
        fontsize=8, fontfamily="monospace", verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    fig.suptitle(
        f"{summary.longitudinal}  +  {summary.lateral}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualize meta-actions on real trajectories")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="outputs/meta_actions")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clip index...")
    avdi = PhysicalAIAVDatasetInterface()
    clip_index = avdi.clip_index
    split_df = clip_index[(clip_index["split"] == args.split) & clip_index["clip_is_valid"]]
    all_clips = split_df.index.tolist()
    print(f"  {len(all_clips)} valid {args.split} clips")

    sampled = rng.choice(all_clips, size=min(args.num_samples, len(all_clips)), replace=False)
    print(f"  Sampling {len(sampled)} clips\n")

    for i, clip_id in enumerate(sampled):
        print(f"[{i+1}/{len(sampled)}] {clip_id} ... ", end="", flush=True)
        try:
            data = load_physical_aiavdataset(
                clip_id=clip_id, t0_us=args.t0_us, avdi=avdi, maybe_stream=True,
            )

            intrinsics = avdi.get_clip_feature(
                clip_id, avdi.features.CALIBRATION.CAMERA_INTRINSICS, maybe_stream=True,
            )
            extrinsics = avdi.get_clip_feature(
                clip_id, avdi.features.CALIBRATION.SENSOR_EXTRINSICS, maybe_stream=True,
            )

            cam_model = intrinsics.camera_models["camera_front_wide_120fov"]
            cam_pose = extrinsics.sensor_poses["camera_front_wide_120fov"]

            traj = data["ego_future_xyz"].cpu().numpy()[0, 0]
            summary = extract_meta_actions_summary(traj)

            out_path = output_dir / f"{i:02d}_{summary.longitudinal}_{summary.lateral}_{clip_id[:8]}.png"
            plot_sample(clip_id, data, cam_model, cam_pose, output_path=out_path)
            print(f"{summary.longitudinal} + {summary.lateral}  -> {out_path.name}")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone. Saved {len(sampled)} figures to {output_dir}/")


if __name__ == "__main__":
    main()
