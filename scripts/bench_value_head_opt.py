"""Dry-run benchmark: serial vs batched value head inference + training.

Measures wall-clock latency and peak GPU memory for:
  1. compute_segment_advantages_from_rollouts (inference)
  2. train_segment_value_head (training)
  3. trajectory_quality_reward (CPU vectorization)

Usage:
    python scripts/bench_value_head_opt.py [--completions 128] [--hidden-dim 4096] [--traj-len 64]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import the NEW (batched) implementations
# ---------------------------------------------------------------------------
from alpamayo_r1.training.advantage_conditioning import (
    compute_segment_advantages_from_rollouts as compute_advantages_batched,
    train_segment_value_head as train_vh_batched,
    compute_value_targets,
)
from alpamayo_r1.training.rewards import trajectory_quality_reward as traj_reward_vectorized
from alpamayo_r1.training.value_head import SegmentValueHead


# ---------------------------------------------------------------------------
# OLD (serial) implementations — inlined for A/B comparison
# ---------------------------------------------------------------------------

def compute_advantages_serial(
    segment_hidden_stash, segment_reward_stash, completion_segment_map,
    value_head, reward_weights=(0.5, 0.25, 0.25),
):
    """Original serial loop (copied from pre-optimization code)."""
    w_traj, w_reason, w_consist = reward_weights
    vh_device = next(value_head.parameters()).device
    results = []

    for i in range(len(segment_hidden_stash)):
        seg = segment_hidden_stash[i]
        seg_map = completion_segment_map[i]
        seg_rew = segment_reward_stash[i]

        h_obs = seg["h_obs"].to(vh_device)
        h_traj = seg["h_traj"].to(vh_device)

        r_traj_scalar = seg_rew.get("r_traj", 0.0)
        r_reason = seg_rew.get("r_reason", 0.0)
        r_consist = seg_rew.get("r_consist", 0.0)

        T_traj = seg_map["traj_len"]
        if T_traj > 0:
            r_per_step = seg_rew.get("r_traj_per_step")
            if r_per_step is not None and len(r_per_step) == T_traj:
                r_per_step_t = torch.tensor(r_per_step, dtype=torch.float32, device=vh_device)
            else:
                r_per_step_t = torch.full(
                    (T_traj,), r_traj_scalar / max(T_traj, 1), device=vh_device
                )
            r_traj_weighted = w_traj * r_per_step_t
            r_traj_weighted[-1] = r_traj_weighted[-1] + w_consist * r_consist
            g_traj_all = torch.flip(torch.cumsum(torch.flip(r_traj_weighted, [0]), dim=0), [0])
        else:
            r_traj_weighted = torch.zeros(0, device=vh_device)
            g_traj_all = torch.zeros(0, device=vh_device)

        traj_total = r_traj_weighted.sum().item() if T_traj > 0 else 0.0
        g_obs = w_reason * r_reason + traj_total

        with torch.no_grad():
            v_obs = value_head(h_obs, level=SegmentValueHead.LEVEL_OBS).item()
            if T_traj > 0:
                curv_idx = torch.arange(1, T_traj, 2, device=vh_device)
                h_traj_curv = h_traj[curv_idx]
                traj_pos = (curv_idx // 2 + SegmentValueHead.POS_TRAJ_START).unsqueeze(0)
                v_traj = value_head(
                    h_traj_curv.unsqueeze(0), level=SegmentValueHead.LEVEL_TRAJ,
                    positions=traj_pos,
                ).squeeze(0)
                g_traj = g_traj_all[curv_idx]
            else:
                v_traj = torch.zeros(0, device=vh_device)
                g_traj = torch.zeros(0, device=vh_device)

        a_obs = g_obs - v_obs
        if T_traj > 0:
            a_traj_per_step = g_traj - v_traj
            a_traj_mean = a_traj_per_step.mean().item()
            a_traj_list = a_traj_per_step.cpu().tolist()
        else:
            a_traj_mean = 0.0
            a_traj_list = []

        results.append({"a_obs": a_obs, "a_traj": a_traj_mean, "a_traj_per_step": a_traj_list})
    return results


def train_vh_serial(
    value_head, optimizer, segment_hidden_stash, g_obs_list, g_traj_list,
    num_epochs=10, batch_size=64,
):
    """Original serial traj-level training loop."""
    import math
    B = len(g_obs_list)
    if B == 0 or num_epochs <= 0:
        return {}

    vh_device = next(value_head.parameters()).device

    h_obs_all = torch.cat([segment_hidden_stash[i]["h_obs"] for i in range(B)], dim=0)
    g_obs_all = torch.tensor(g_obs_list, dtype=torch.float32)

    for epoch in range(num_epochs):
        g = torch.Generator().manual_seed(epoch * 31337)
        perm = torch.randperm(B, generator=g)

        for batch_start in range(0, B, batch_size):
            idx = perm[batch_start : batch_start + batch_size]

            h_obs_batch = h_obs_all[idx].to(vh_device)
            g_obs_batch = g_obs_all[idx].to(vh_device)
            obs_pred = value_head(h_obs_batch, level=SegmentValueHead.LEVEL_OBS)
            loss_obs = F.mse_loss(obs_pred, g_obs_batch)

            # Serial traj loop
            traj_losses = []
            for i_local in idx.tolist():
                h_traj = segment_hidden_stash[i_local]["h_traj"]
                g_traj = g_traj_list[i_local]
                if h_traj.shape[0] == 0:
                    continue
                T_t = h_traj.shape[0]
                curvature_idx = torch.arange(1, T_t, 2)
                h_sub = h_traj[curvature_idx].to(vh_device)
                g_sub = g_traj[curvature_idx].to(vh_device)
                traj_pos = (curvature_idx // 2 + SegmentValueHead.POS_TRAJ_START).unsqueeze(0).to(vh_device)
                v = value_head(
                    h_sub.unsqueeze(0), level=SegmentValueHead.LEVEL_TRAJ, positions=traj_pos
                ).squeeze(0)
                traj_losses.append(F.mse_loss(v, g_sub))

            if traj_losses:
                loss_traj = torch.stack(traj_losses).mean()
            else:
                loss_traj = torch.tensor(0.0, device=vh_device)

            total_loss = (loss_obs + loss_traj) / 2.0
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


def traj_reward_serial(completions, pred_xyz, gt_xyz, **kwargs):
    """Original per-sample NumPy loop."""
    ade_threshold = kwargs.get("ade_threshold", 5.0)
    if pred_xyz is None or gt_xyz is None:
        return [0.0] * len(completions)

    rewards = []
    for pred_flat, gt_flat in zip(pred_xyz, gt_xyz):
        try:
            pred = np.array(pred_flat, dtype=np.float32).reshape(-1, 3)
            gt = np.array(gt_flat, dtype=np.float32).reshape(-1, 3)
            l2 = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1)
            ade = float(l2.mean())
            rewards.append(max(0.0, 1.0 - ade / ade_threshold))
        except Exception:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_synthetic_data(num_completions: int, hidden_dim: int, traj_len: int):
    """Generate synthetic stash data matching real training shapes."""
    segment_hidden_stash = []
    segment_reward_stash = []
    completion_segment_map = []

    for _ in range(num_completions):
        h_obs = torch.randn(1, hidden_dim)
        h_traj = torch.randn(traj_len, hidden_dim)
        segment_hidden_stash.append({"h_obs": h_obs, "h_coc": h_obs.clone(), "h_traj": h_traj})

        r_traj = np.random.uniform(0, 1)
        r_per_step = np.random.uniform(0, 0.05, size=traj_len).tolist()
        segment_reward_stash.append({
            "r_traj": r_traj,
            "r_reason": np.random.uniform(0, 1),
            "r_consist": np.random.uniform(0, 1),
            "r_traj_per_step": r_per_step,
        })
        completion_segment_map.append({
            "coc_len": 50,
            "traj_len": traj_len,
            "traj_positions": list(range(traj_len)),
        })

    return segment_hidden_stash, segment_reward_stash, completion_segment_map


def make_trajectory_data(num_completions: int, num_timesteps: int = 64):
    """Generate synthetic trajectory data for reward benchmarking."""
    completions = ["test"] * num_completions
    pred_xyz = [np.random.randn(num_timesteps * 3).tolist() for _ in range(num_completions)]
    gt_xyz = [np.random.randn(num_timesteps * 3).tolist() for _ in range(num_completions)]
    return completions, pred_xyz, gt_xyz


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def measure_gpu_memory():
    """Return current and peak GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    torch.cuda.synchronize()
    current = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return current, peak


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def verify_results(results_a, results_b, label: str, atol: float = 1e-4):
    """Check that two result lists match within tolerance."""
    assert len(results_a) == len(results_b), f"{label}: length mismatch"
    mismatches = 0
    for i, (a, b) in enumerate(zip(results_a, results_b)):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a:
                va = a[key]
                vb = b[key]
                if isinstance(va, float):
                    if abs(va - vb) > atol:
                        mismatches += 1
                elif isinstance(va, list):
                    for x, y in zip(va, vb):
                        if abs(x - y) > atol:
                            mismatches += 1
                            break
        elif isinstance(a, (int, float)):
            if abs(a - b) > atol:
                mismatches += 1
    if mismatches == 0:
        print(f"  [{label}] Correctness: PASS (all values match within atol={atol})")
    else:
        print(f"  [{label}] Correctness: FAIL ({mismatches} mismatches)")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Completions: {args.completions}, Hidden dim: {args.hidden_dim}, Traj len: {args.traj_len}")
    print(f"VH train epochs: {args.vh_epochs}, VH batch size: {args.vh_batch_size}")
    print("=" * 72)

    # --- Setup ---
    stash, rewards_stash, seg_map = make_synthetic_data(
        args.completions, args.hidden_dim, args.traj_len,
    )
    value_head = SegmentValueHead(hidden_dim=args.hidden_dim).to(device).eval()

    # Warmup
    _ = compute_advantages_serial(stash[:2], rewards_stash[:2], seg_map[:2], value_head)
    _ = compute_advantages_batched(stash[:2], rewards_stash[:2], seg_map[:2], value_head)

    # ================================================================
    # Benchmark 1: compute_segment_advantages_from_rollouts
    # ================================================================
    print("\n--- Benchmark 1: Advantage Computation (Inference) ---")

    # Serial
    reset_peak_memory()
    mem_before_serial, _ = measure_gpu_memory()
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        results_serial = compute_advantages_serial(stash, rewards_stash, seg_map, value_head)
    t_serial = (time.perf_counter() - t0) / args.repeats
    _, mem_peak_serial = measure_gpu_memory()

    # Batched
    reset_peak_memory()
    mem_before_batched, _ = measure_gpu_memory()
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        results_batched = compute_advantages_batched(stash, rewards_stash, seg_map, value_head)
    t_batched = (time.perf_counter() - t0) / args.repeats
    _, mem_peak_batched = measure_gpu_memory()

    print(f"  Serial:  {t_serial*1000:8.1f} ms | peak GPU: {mem_peak_serial:8.1f} MB")
    print(f"  Batched: {t_batched*1000:8.1f} ms | peak GPU: {mem_peak_batched:8.1f} MB")
    speedup = t_serial / t_batched if t_batched > 0 else float("inf")
    mem_delta = mem_peak_batched - mem_peak_serial
    print(f"  Speedup: {speedup:.1f}x | Memory delta: {mem_delta:+.1f} MB")

    verify_results(results_serial, results_batched, "advantages")

    # ================================================================
    # Benchmark 2: train_segment_value_head
    # ================================================================
    print("\n--- Benchmark 2: Value Head Training ---")

    g_obs_list, g_traj_list = compute_value_targets(rewards_stash, seg_map)

    # Serial
    vh_serial = SegmentValueHead(hidden_dim=args.hidden_dim).to(device)
    opt_serial = torch.optim.Adam(vh_serial.parameters(), lr=1e-3)
    reset_peak_memory()
    t0 = time.perf_counter()
    train_vh_serial(
        vh_serial, opt_serial, stash, g_obs_list, g_traj_list,
        num_epochs=args.vh_epochs, batch_size=args.vh_batch_size,
    )
    t_train_serial = time.perf_counter() - t0
    _, mem_peak_train_serial = measure_gpu_memory()

    # Batched
    vh_batched = SegmentValueHead(hidden_dim=args.hidden_dim).to(device)
    opt_batched = torch.optim.Adam(vh_batched.parameters(), lr=1e-3)
    reset_peak_memory()
    t0 = time.perf_counter()
    train_vh_batched(
        vh_batched, opt_batched, stash, g_obs_list, g_traj_list,
        num_epochs=args.vh_epochs, batch_size=args.vh_batch_size,
        log_interval=9999,  # suppress logging
    )
    t_train_batched = time.perf_counter() - t0
    _, mem_peak_train_batched = measure_gpu_memory()

    print(f"  Serial:  {t_train_serial*1000:8.1f} ms | peak GPU: {mem_peak_train_serial:8.1f} MB")
    print(f"  Batched: {t_train_batched*1000:8.1f} ms | peak GPU: {mem_peak_train_batched:8.1f} MB")
    speedup_train = t_train_serial / t_train_batched if t_train_batched > 0 else float("inf")
    mem_delta_train = mem_peak_train_batched - mem_peak_train_serial
    print(f"  Speedup: {speedup_train:.1f}x | Memory delta: {mem_delta_train:+.1f} MB")

    # ================================================================
    # Benchmark 3: trajectory_quality_reward
    # ================================================================
    print("\n--- Benchmark 3: Trajectory Quality Reward (CPU) ---")

    completions, pred_xyz, gt_xyz = make_trajectory_data(args.completions)

    # Serial
    t0 = time.perf_counter()
    for _ in range(args.repeats * 10):
        r_serial = traj_reward_serial(completions, pred_xyz, gt_xyz)
    t_reward_serial = (time.perf_counter() - t0) / (args.repeats * 10)

    # Vectorized
    t0 = time.perf_counter()
    for _ in range(args.repeats * 10):
        r_vectorized = traj_reward_vectorized(completions, pred_xyz=pred_xyz, gt_xyz=gt_xyz)
    t_reward_vectorized = (time.perf_counter() - t0) / (args.repeats * 10)

    print(f"  Serial:     {t_reward_serial*1000:8.3f} ms")
    print(f"  Vectorized: {t_reward_vectorized*1000:8.3f} ms")
    speedup_reward = t_reward_serial / t_reward_vectorized if t_reward_vectorized > 0 else float("inf")
    print(f"  Speedup: {speedup_reward:.1f}x")

    verify_results(r_serial, r_vectorized, "trajectory_reward", atol=1e-5)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Advantage inference:  {speedup:.1f}x faster  |  {mem_delta:+.1f} MB GPU overhead")
    print(f"  Value head training:  {speedup_train:.1f}x faster  |  {mem_delta_train:+.1f} MB GPU overhead")
    print(f"  Trajectory reward:    {speedup_reward:.1f}x faster  |  CPU only (no GPU)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark value head optimizations")
    parser.add_argument("--completions", type=int, default=128, help="Number of completions")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="VLM hidden dimension")
    parser.add_argument("--traj-len", type=int, default=64, help="Trajectory token length")
    parser.add_argument("--vh-epochs", type=int, default=5, help="Value head training epochs")
    parser.add_argument("--vh-batch-size", type=int, default=64, help="Value head batch size")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats for timing stability")
    args = parser.parse_args()
    run_benchmark(args)
