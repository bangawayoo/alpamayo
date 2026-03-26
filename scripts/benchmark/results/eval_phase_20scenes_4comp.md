# Evaluate Phase Profile: 20 scenes × 4 completions (80 total)

Date: 2026-03-26
GPU: Single GPU (CUDA_VISIBLE_DEVICES=1)
Mode: vlm_only, scene_batch_size=1
Model: nvidia/Alpamayo-R1-10B (bf16)
Expert: disabled
Value head: enabled (segment-level), 1 train epoch

## Per-Scene Rollout Breakdown (with stash)

| Step | Avg Time | Per Completion | Description |
|---|---|---|---|
| vlm_generate | 12.4s | 3.1s | AR generation (4 completions) |
| **stash_hidden** | **4.1s** | **1.0s** | TF forward for segment hidden states |
| prepare_batch | 0.08s | — | Data loading + token fusion |
| traj_decode | 0.02s | — | Discrete → continuous trajectory |
| **Total per scene** | **~16.5s** | — | |

## Phase Timings

| Phase | Time | Description |
|---|---|---|
| Phase 1: ROLLOUT | 371s | 20 scenes × 4 completions (includes stash TF forwards) |
| **Phase 2: EVALUATE** | **6.1s** | Rewards + advantages + VH training (CPU-only except VH) |
| Phase 3: TRAIN | 450s | SFT training (1 epoch) |

## Phase 2 Stage Breakdown

| Stage | Time | Description |
|---|---|---|
| 1. compute_rewards | 0.1s | CPU-only: minADE, CoC parsing, consistency |
| 2. segment_hidden (stash) | 0.0s | Read from rollout stash |
| 3. compute_advantages | 2.3s | Value head forward per completion |
| 4. train_value_head | 3.7s | 1 epoch MLP training |

## What the optimization does

The TF forward cost (~4.1s/scene, ~82s total) **moved from Phase 2 into Phase 1**.
Total VLM compute is unchanged. The structural gain is that Phase 2 becomes
CPU-only (no VLM forward passes), which simplifies the GPU scheduling between
rollout and training phases.

## Baseline comparison (before stash, from earlier profiling)

| | Before (Phase 2 TF) | After (Phase 1 stash) |
|---|---|---|
| Phase 1 | ~245s | ~371s (+~82s stash) |
| Phase 2 | ~52s | ~6s |
| Phase 2 Stage 2 | 45s | 0s |

Note: Phase 1 baseline was measured on a different run with different GPU
contention; the ~82s stash cost vs ~45s Phase 2 TF cost difference is partly
due to Phase 1 processing scenes one at a time (scene_batch_size=1) vs
Phase 2 grouping by clip_id.
