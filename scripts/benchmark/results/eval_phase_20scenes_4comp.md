# Evaluate Phase Profile: 20 scenes × 4 completions (80 total)

Date: 2026-03-25
GPU: Single GPU (CUDA_VISIBLE_DEVICES=1)
Mode: vlm_only, scene_batch_size=1
Model: nvidia/Alpamayo-R1-10B (bf16)
Expert: disabled
Value head: enabled (segment-level), 1 train epoch

## Full Iteration Timing

| Phase | Time | Description |
|---|---|---|
| Phase 1: ROLLOUT | 245.2s | 20 scenes × 4 completions, ~12s/scene |
| **Phase 2: EVALUATE** | **51.9s** | Scoring + binarizing advantages |
| Phase 3: TRAIN | 331.1s | SFT training (1 epoch) |

## Phase 2 Breakdown

| Stage | Time | % of Phase 2 | Description |
|---|---|---|---|
| 1. compute_rewards | 0.1s | 0.2% | CPU-only: minADE, CoC parsing, consistency |
| **2. extract_segment_hidden** | **45.4s** | **87.5%** | Sequential VLM teacher-forced forwards |
| 3. compute_advantages | 2.3s | 4.4% | Value head forward per completion |
| 4. train_value_head | 4.1s | 7.9% | 1 epoch MLP training |
| 5+6. buffer update + binarize | <0.1s | ~0% | CPU percentile thresholds |

## Key Finding

Stage 2 (`extract_segment_hidden`) dominates at **87.5%** of Phase 2.
Each completion requires a teacher-forced VLM forward pass (~0.57s/completion).
Completions are grouped by clip_id to share prompt KV cache, but the forward
passes within each group are sequential.

## Optimization Applied

Moved TF forwards into Phase 1 (rollout) via `stash_segment_hidden_in_results()`.
Phase 2 reads from the stash instead of re-running forwards.

**After optimization (80 completions):**

| Phase | Before | After |
|---|---|---|
| Phase 1 (Rollout) | 245s | ~224s (+TF forwards absorbed) |
| Phase 2 (Evaluate) | 52s | 6s |

Note: the TF forward cost (~45s) moved from Phase 2 into Phase 1 — the total
VLM forward compute is unchanged. The gain is structural: Phase 2 becomes
CPU-only, and the TF forwards in Phase 1 benefit from data cache locality
(model inputs already loaded for generation).

## Reward Statistics

```
traj=[0.117, 0.780, 0.980] reason=[0.438, 0.730, 0.875] consist=[0.000, 0.312, 1.000]
```

39/80 completions (48.8%) were all-positive after binarization.
