# Profiling Summary (Self-Play + Inner Training Step)

This document summarizes profiling metrics collected from:

- `outputs/sft_advcond/profiles/selfplay_profile_rank{0,1}.jsonl`
- `outputs/sft_advcond/iter_*/profiles/sft_trainer_profile_rank{0,1}.jsonl`

The run was still in progress when this snapshot was generated, so numbers reflect currently available records.

## How to run with profiling enabled

```bash
source activate alpa

./scripts/run_sft.sh --no-fsdp --num-gpus 1 \
  training.enable_profiling=true \
  training.profile_inner_steps=true \
  training.profile_inner_every_n_steps=1 \
  data.max_samples=60 \
  advantage_conditioning.num_iterations=3
```

## Bubble-time definition

`bubble_before_gpu_s` is measured as the wall-clock gap between the end of the previous profiled GPU-active section and the start of the current profiled GPU-active section.

- In `AdvCondSFTTrainer` profiling, CUDA synchronization is enabled at section boundaries for better timing fidelity.
- This is a **section-level idle-gap heuristic**, not kernel-level occupancy.

## Consistent bubble offenders (inner training steps only)

Filtered to recurring training-step stages (excluding phase-boundary effects).

| Stage | Count | Bubble P50 (s) | Bubble Mean (s) | Bubble P95 (s) | Bubble Max (s) | Fraction > 1s |
|---|---:|---:|---:|---:|---:|---:|
| `sft_forward_backward` | 112 | **1.923** | 1.898 | 5.503 | 7.257 | **50.9%** |
| `expert_phaseA_extract_kv` | 112 | 0.016 | 0.099 | 0.337 | 0.561 | 0.0% |
| `expert_phaseB_forward_backward` | 112 | 0.002 | 0.029 | 0.160 | 0.246 | 0.0% |
| `expert_phaseB_optimizer_step` | 112 | 0.092 | 0.101 | 0.238 | 0.547 | 0.0% |
| `expert_cfm_total` | 112 | 0.011 | 0.053 | 0.257 | 0.278 | 0.0% |

**Conclusion:** the only consistent large bubble is before `sft_forward_backward`.

---

## Effect of DataLoader workers (`outputs/profile_workers4`)

We ran the same profiling with:

- `training.dataloader_num_workers=4`
- `training.dataloader_pin_memory=true`
- `training.dataloader_persistent_workers=true`

and analyzed:

- `outputs/profile_workers4/profiles/selfplay_profile_rank{0,1}.jsonl`
- `outputs/profile_workers4/iter_*/profiles/sft_trainer_profile_rank{0,1}.jsonl`

### Inner-step bubble comparison (step > 1)

| Stage | Baseline P50 (s) | Workers=4 P50 (s) | Baseline Fraction > 1s | Workers=4 Fraction > 1s |
|---|---:|---:|---:|---:|
| `sft_forward_backward` | **1.923** | **0.033** | **50.9%** | **23.2%** |

For non-`sft_forward_backward` stages, median bubbles remain near-zero in both runs.

### Patterns seen in remaining spikes

- Spikes are still concentrated before `sft_forward_backward`.
- Cross-rank overlap is empty (rank-local stalls):
  - iter1 rank0 spikes: `[6, 8, 9, 11, 21]`
  - iter1 rank1 spikes: `[3, 4, 7, 14, 15, 16, 22]`
  - iter2 rank0 spikes: `[9, 11, 15, 18, 21]`
  - iter2 rank1 spikes: `[3, 4, 6, 7, 10, 13, 14, 17, 19]`
- Many spike steps coincide with heavy `expert_phaseA_extract_kv` duration in the same step.

### Interpretation

Increasing DataLoader workers + pin-memory significantly reduces the **consistent** pre-forward bubbles, but does not eliminate all rank-local spikes.

---

## Phase/stage wall-clock + memory + VRAM (P50 / P99)

### SelfPlay phase/stage summary

(Combined across rank0/rank1 self-play profile logs.)

| Phase/Stage | Count | Wall P50 (s) | Wall P99 (s) | RSS P50/P99 (GB) | VRAM Reserved P50/P99 (GB) | VRAM Alloc P50/P99 (GB) | VRAM Used P50/P99 (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `TRAIN/phase_total` | 6 | 325.608 | 415.937 | 27.639 / 28.414 | 0.971 / 0.973 | 0.052 / 0.052 | 2.205 / 2.207 |
| `TRAIN/trainer_train` | 6 | 183.571 | 198.121 | 6.847 / 9.313 | 53.131 / 76.852 | 22.603 / 22.605 | 54.366 / 78.089 |
| `ROLLOUT/phase_total` | 6 | 63.054 | 66.851 | 6.378 / 8.961 | 55.046 / 62.764 | 22.212 / 22.212 | 56.259 / 63.998 |
| `GT_AUGMENT/phase_total` | 6 | 16.420 | 23.006 | 6.378 / 9.417 | 55.046 / 62.764 | 22.230 / 22.238 | 56.278 / 63.998 |
| `EVALUATE/phase_total` | 6 | 16.356 | 17.366 | 6.378 / 9.417 | 55.046 / 62.764 | 22.230 / 22.238 | 56.278 / 63.998 |
| `EVALUATE/extract_prompt_hidden` | 6 | 16.055 | 16.829 | 6.378 / 9.417 | 55.046 / 62.764 | 22.212 / 22.212 | 56.259 / 63.998 |
| `CLEANUP/pre_train_cleanup` | 6 | 0.228 | 0.524 | 6.378 / 9.378 | 22.509 / 23.337 | 22.230 / 22.238 | 23.743 / 24.569 |
| `EVALUATE/train_value_head` | 6 | 0.015 | 0.578 | 6.378 / 9.417 | 55.046 / 62.764 | 22.230 / 22.238 | 56.278 / 63.998 |
| `EVALUATE/compute_advantages` | 6 | 0.015 | 0.255 | 6.378 / 9.417 | 55.046 / 62.764 | 22.212 / 22.212 | 56.259 / 63.998 |
| `EVALUATE/compute_rewards` | 6 | 0.020 | 0.069 | 6.378 / 8.961 | 55.046 / 62.764 | 22.212 / 22.212 | 56.259 / 63.998 |
| `BOOKKEEPING/phase_total` | 6 | 0.001 | 0.002 | 27.639 / 28.414 | 0.971 / 0.973 | 0.052 / 0.052 | 2.205 / 2.207 |

### Trainer inner-step stage summary

(Combined across all `iter_*/profiles/sft_trainer_profile_rank*.jsonl` files.)

| Stage | Count | Wall P50 (s) | Wall P99 (s) | RSS P50/P99 (GB) | VRAM Reserved P50/P99 (GB) | VRAM Alloc P50/P99 (GB) | VRAM Used P50/P99 (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft_forward_backward` | 118 | 3.183 | 8.483 | 8.411 / 9.378 | 53.127 / 76.852 | 22.806 / 22.808 | 54.361 / 78.089 |
| `expert_cfm_total` | 118 | 1.808 | 2.477 | 8.411 / 9.378 | 53.127 / 76.852 | 22.806 / 22.808 | 54.361 / 78.089 |
| `expert_phaseA_extract_kv` | 118 | 1.035 | 1.489 | 8.411 / 9.378 | 53.127 / 76.852 | 23.783 / 23.788 | 54.361 / 78.089 |
| `expert_phaseB_forward_backward` | 118 | 0.480 | 1.024 | 8.411 / 9.378 | 53.127 / 76.852 | 25.619 / 25.632 | 54.361 / 78.089 |
| `expert_phaseB_optimizer_step` | 118 | 0.010 | 0.240 | 8.411 / 9.378 | 53.127 / 76.852 | 25.619 / 25.632 | 54.361 / 78.089 |

---

## Iteration trend (bubble in `sft_forward_backward`, step > 1)

### Baseline run (`outputs/sft_advcond`)

| Iteration | Count | Bubble P50 (s) | Bubble Mean (s) | Bubble P90 (s) | Bubble Max (s) |
|---|---:|---:|---:|---:|---:|
| 0 | 28 | 0.070 | 0.156 | 0.526 | 0.662 |
| 1 | 42 | 2.493 | 2.246 | 4.474 | 6.295 |
| 2 | 42 | 2.975 | 2.710 | 5.254 | 7.257 |

### Workers=4 run (`outputs/profile_workers4`)

| Iteration | Count | Bubble P50 (s) | Bubble Mean (s) | Bubble P90 (s) | Bubble Max (s) |
|---|---:|---:|---:|---:|---:|
| 0 | 28 | 0.032 | 0.100 | 0.209 | 0.662 |
| 1 | 42 | 0.050 | 0.998 | 3.046 | 5.477 |
| 2 | 42 | 0.040 | 1.055 | 3.495 | 6.574 |

Workers reduce median bubble dramatically, but long-tail spikes remain in later iterations.
