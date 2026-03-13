# Scene-Level Value Head for GRPO Baseline Estimation

## Motivation

GRPO computes per-sample advantages by normalising rewards within a group of completions for the same prompt:

```
A_i = (r_i − mean(r_group)) / std(r_group)
```

The group mean is a noisy, scene-agnostic baseline. It has two failure modes:

1. **Scene heterogeneity** — a batch may mix easy (empty highway) and hard (crowded intersection) driving scenes. A hard scene with a decent trajectory looks "bad" if it is averaged against easy scenes, and vice versa. The advantage signal conflates intrinsic scene difficulty with policy quality.

2. **Small group variance** — with `num_generations=8` per scene, the group mean is a high-variance estimate. Across gradient accumulation steps the effective baseline for any single scene can drift substantially.

A learned `V(scene)` that predicts `E[r | scene]` provides a scene-conditioned, low-variance baseline. The advantage becomes:

```
A_i = r_i − V(scene_i)
```

This is the standard actor-critic advantage and is strictly lower-variance than the group mean when `V` is even a crude predictor of scene difficulty.

### Literature basis

| Paper | Contribution used here |
|---|---|
| [π₀.₆ / RECAP](https://arxiv.org/abs/2511.14759) | Distributional value head on a separate VLM backbone; MC return targets; offline advantage → policy improvement |
| [RL for VLA Generalization](https://arxiv.org/abs/2505.19789) | Shared actor-critic backbone (no duplicate weights); value head attached to **first action-token** h₀; PPO >> GRPO for robotic POMDPs |
| [VAPO](https://arxiv.org/abs/2504.05118) | Value pre-training on fixed-policy MC returns before RL begins; decoupled GAE (λ=1 for value, adaptive for policy); token-level loss |
| [VinePPO](https://arxiv.org/abs/2410.01679) | Learned critics fail for long reasoning chains → MC rollouts from intermediate states are preferable for token-level credit assignment |

---

## Architecture

```
VLM (Qwen3-VL, frozen during value forward)
  └─ last hidden layer, last prompt token → h₀: (1, 4096)

SceneValueHead
  ├─ Linear(4096 → 512) + GELU
  ├─ Linear(512  → 128) + GELU
  └─ Linear(128  →   1) → V(scene): scalar ∈ ℝ
```

**`h₀` is the VLM's hidden state at the final prompt token**, computed via a single forward pass with `output_hidden_states=True` before any generation tokens are produced. This position encodes the model's full scene understanding — camera images, history trajectory, and the task instruction — without yet committing to any completion tokens.

This choice follows the RL-for-VLA paper (2505.19789), which found `h₀` (first action-token embedding) outperforms last-token or concatenated representations as a critic input.

The value head is a **lightweight MLP** (~82k parameters) attached to the existing VLM backbone. No second backbone is used (unlike π₀.₆), keeping memory overhead negligible on the 40GB A100 run environment.

---

## Value Estimation Pipeline

### 1. h₀ collection (during rollout)

Inside `AlpamayoGRPOTrainer._generate_single_turn`, once per **unique scene** (not per completion), `_compute_scene_h0` runs a no-grad VLM forward pass:

```python
outputs = self.full_model.vlm(
    input_ids=prompt_input_ids,
    output_hidden_states=True,
    **vision_kwargs,          # pixel_values, attention_mask, image_grid_thw
)
h0 = outputs.hidden_states[-1][:, -1, :].float().cpu()  # (1, 4096)
```

The same `h₀` is then stashed once for each of the `G=num_generations` completions of that scene. This avoids recomputing the VLM forward `G` times — the scene embedding is identical for all completions.

### 2. Composite reward collection (during reward computation)

Inside `AlpamayoGRPOTrainer._calculate_rewards`, after the parent computes all reward function scores, `_stash_value_rewards` computes the weighted composite reward and appends it to a parallel stash:

```
r_composite = 0.50 × r_trajectory + 0.25 × r_reasoning + 0.25 × r_consistency
```

The weights mirror the `rewards.{trajectory,reasoning,consistency}_weight` Hydra config values.

### 3. Value head update (during compute_loss)

Each call to `compute_loss` drains `batch_size` items from both stashes via `_train_value_head_step`:

```python
v_pred  = value_head(h0_tensor)                   # (B,)
v_loss  = MSE(v_pred, r_composite_tensor)          # scalar
value_optimizer.zero_grad()
v_loss.backward()                                  # gradients stay inside value head
value_optimizer.step()
```

The value head has a **separate Adam optimizer** (lr=1e-4, independent of the LoRA policy optimizer at 1e-5). Gradients from `v_loss.backward()` never reach the VLM because `h0_tensor` was detached at collection time (`.float().cpu()` severs the autograd graph). This matches VAPO's "decoupled" training design.

---

## Two-Stage Training

### Stage 0 — Value Pre-training (optional)

Controlled by `value_head.pretrain_steps`. When set to N > 0, the first N `compute_loss` calls:
- Train the value head normally (stash → MSE → Adam step)
- **Skip** `super().compute_loss()` entirely — the VLM receives no gradient
- Return `torch.tensor(0.0, requires_grad=True)` as a no-op policy loss

This is the VAPO "value model pre-training" phase. Rollouts still run to collect (h₀, reward) pairs; only the policy update is suppressed. `_value_pretrain_remaining` counts down to zero and stage 1 begins automatically mid-training without any restart.

```
Step 1..N:    rollout → stash → value_head update → return loss=0
Step N+1..:   rollout → stash → value_head update → GRPO policy update
```

### Stage 1 — Online Joint Training

After stage 0, both the value head and the policy train simultaneously. The value head continues updating from the stash on every `compute_loss` call, tracking the shifting reward distribution as the policy improves.

### Workflow

```bash
# Stage 0: bootstrap value head for 300 steps, save checkpoint
./scripts/run_grpo.sh value_head.enabled=true \
  value_head.pretrain_steps=300 \
  "value_head.save_path=outputs/value_head_pretrained.pt"

# Stage 1: full GRPO with warm-started value head
./scripts/run_grpo.sh value_head.enabled=true \
  "value_head.load_path=outputs/value_head_pretrained.pt"
```

---

## Configuration Reference

All options live under `value_head:` in `grpo_default.yaml`:

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable the value head. When false, no code path changes. |
| `hidden_dim` | `4096` | Must match VLM hidden size (4096 for Qwen3-VL-7B/10B). |
| `lr` | `1e-4` | Adam learning rate for the value head (independent of policy lr). |
| `pretrain_steps` | `0` | Stage 0 steps where only the value head trains. 0 = skip stage 0. |
| `save_path` | `null` | If set, saves `value_head.state_dict()` here at each checkpoint. |
| `load_path` | `null` | If set, restores weights from this path at init (pre-trained head). |

---

## Metrics

When enabled, the following keys appear in the standard training log dict alongside reward metrics:

| Metric | Meaning |
|---|---|
| `value_head/loss` | MSE between V(scene) predictions and composite rewards |
| `value_head/pred_mean` | Mean predicted value across the batch |
| `value_head/target_mean` | Mean composite reward (ground truth target) |
| `value_head/pretrain_steps_remaining` | Counts down during stage 0; 0 in stage 1 |

A healthy value head shows `pred_mean` converging toward `target_mean` over training, with `loss` decreasing monotonically through stage 0 and continuing to track the distribution in stage 1.

---

## Current Limitations and Future Work

**V1 (current):** The value head is an auxiliary predictor only. Its output does not yet replace the GRPO group-mean normalisation. Stage 0 produces a useful warm-started checkpoint, but the advantage computation in `super().compute_loss()` still uses:

```
A_i = (r_i − mean(r_group)) / std(r_group)
```

**V2 (planned):** Wire `V(scene)` into advantage computation:

```python
advantages = rewards - value_head(h0_stash).detach()
```

This requires overriding the section of TRL's `GRPOTrainer.compute_loss` that builds the advantage tensor, substituting the group mean with per-scene value predictions.

**V3 (future):** Length-adaptive GAE (VAPO-style) for token-level credit assignment within the CoC reasoning chain. The λ parameter adapts to completion length: `λ(l) = 1 − 1/(α·l)`, providing unbiased estimates for short completions and bootstrapped estimates for long ones. This addresses heterogeneous CoC lengths (some 2-sentence, some 10-sentence reasoning chains) which fixed-λ GAE handles poorly.
