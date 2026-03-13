# Segment-Level Value Head for Alpamayo-R1

## Motivation

Standard GRPO assigns a single scalar reward to an entire rollout, then broadcasts it as the advantage for every token. This means a good trajectory boosts all CoC tokens equally, and good reasoning boosts all trajectory tokens equally — there is no credit assignment between segments.

We want segment-level advantages: the value network estimates expected return at three points in the sequence, enabling TD-style advantage computation that attributes credit to the right part of the generation.

This design stays within `AlpamayoGRPOTrainer` (no separate PPO trainer needed) because the core optimization is still group-relative clipped surrogate. The value network serves as a **learned baseline for variance reduction**, not a PPO-style critic that changes the objective. However, as noted below, bootstrapping token-level values makes this a natural stepping stone toward a full GRPO–PPO hybrid.

---

## Current State (SceneValueHead)

The existing `SceneValueHead` (in `value_head.py`) provides a single-point baseline:

```
V(observation) = MLP(h_obs)
```

Where `h_obs` is the VLM's hidden state at the last prompt token, extracted via a separate forward pass before generation. This gives one scalar per scene, used only for logging — it does not yet feed back into advantage computation.

---

## Three-Level Value Architecture

### Sequence Structure

A completion has three semantic segments:

```
[prompt tokens]  [CoC reasoning tokens]  [trajectory tokens]
     s_obs    →    <cot_start>...<cot_end>   →   <traj_future_start> <i0>...<i63> <traj_future_end>
```

### Value Estimates

The value network computes V(s) at three granularities using VLM hidden states at specific positions:

| Level | Position | Hidden state | Encodes |
|---|---|---|---|
| V_obs | Last prompt token | h_obs | Scene understanding before any generation |
| V_coc | `<cot_end>` token | h_coc | Scene + quality of reasoning produced |
| V_traj(t) | Each `<iN>` token | h_traj_t | Scene + reasoning + trajectory-so-far |

All three levels share the same MLP weights. The hidden state at each position already carries different information — the VLM's autoregressive nature means h_traj_t has "seen" both the prompt and CoC text, so no explicit level embedding is needed (though one could be added if training signal is weak).

### Network Architecture

Upgrade `SceneValueHead` → `SegmentValueHead`:

```python
class SegmentValueHead(nn.Module):
    """Shared MLP: h → V(s) at any sequence position."""

    def __init__(self, hidden_dim: int = 4096) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, hidden_dim) or (B, T, hidden_dim) → (B,) or (B, T)"""
        return self.net(h).squeeze(-1)
```

Same parameter count as today (~2.1M). The only change is that it accepts batched token-level hidden states, not just a single h_0.

---

## Reward Decomposition

To compute segment-level advantages, we need segment-level rewards.

### CoC Segment Rewards

Received when CoC generation completes:
- **R_reasoning**: rule-based reasoning quality (causal connectors, driving terms, length, no repetition)
- **R_consistency**: agreement between CoC text and predicted trajectory (meta-action keyword matching)

These are the existing `reasoning_quality_reward` and `consistency_reward` functions, unchanged.

### Trajectory Token Rewards

The existing `trajectory_quality_reward` computes a scalar ADE over the full trajectory. For token-level advantages, we decompose it into **per-timestep rewards**:

```python
# Current: single scalar
ade = l2_per_step.mean()
reward = max(0, 1 - ade / threshold)

# New: per-timestep rewards
r_t = max(0, 1 - l2_per_step[t] / threshold)   # one reward per trajectory token
```

Each trajectory token `<iN>` maps to a specific timestep in the decoded (T, 3) trajectory, so the mapping from token index to timestep reward is direct. With 64 trajectory tokens encoding T timesteps (T depends on tokenizer configuration), each token gets its own L2-based reward.

---

## Advantage Computation

### Semi-MDP Formulation

The generation process forms a semi-MDP with two segment types:

```
s_obs ──[CoC tokens]──→ s_coc ──[traj_1]──→ s_1 ──[traj_2]──→ ... ──→ s_T ──→ terminal
         segment 1                          segment 2 (per-token)
```

### CoC Segment (Shared Advantage)

All CoC tokens receive the same advantage — a single TD step spanning the entire reasoning segment:

```
A_coc = (R_reasoning + R_consistency) + γ · V(s_coc) - V(s_obs)
```

This says: "how much better was this reasoning trace than what we expected from the scene alone?"

### Trajectory Tokens (Per-Token GAE)

Each trajectory token gets its own advantage via Generalized Advantage Estimation:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)       for t = 1..T-1
δ_T = r_T - V(s_T)                          terminal (γ · 0)

A_t = Σ_{k=0}^{T-t-1} (γλ)^k · δ_{t+k}    GAE with discount γ, trace decay λ
```

With sequences of ~64 trajectory tokens, `γ = 1.0` (or 0.99) and `λ = 0.95` are reasonable starting points.

### Resulting Advantage Tensor

The output is a `(B, T_completion)` tensor where `T_completion` is the full completion length:

```
advantages[b, :] = [A_coc, A_coc, ..., A_coc, A_traj_1, A_traj_2, ..., A_traj_T]
                    ├── CoC positions ──┤  ├── trajectory positions ──────────┤
```

TRL's `_compute_loss` already supports `(B, T)` advantages (see the `if advantages.dim() == 1: advantages = advantages.unsqueeze(1)` guard in the base GRPOTrainer).

### Group Normalization

GRPO's key property is normalizing within a group of G generations per prompt. With segment-level advantages, normalize **per segment type** across the G completions:

```
A_coc_normalized = (A_coc - mean(A_coc over G)) / (std(A_coc over G) + ε)
A_traj_t_normalized = (A_traj_t - mean(A_traj_t over G)) / (std(A_traj_t over G) + ε)
```

This prevents the trajectory signal from dominating simply because it has more tokens. Each segment type competes on its own scale within the group.

---

## Where Hidden States Come From

### Current: Separate Forward Pass

`_compute_scene_h0()` runs an extra VLM forward on the prompt alone with `output_hidden_states=True`. This is expensive (~same cost as the generation forward pass).

### Proposed: Piggyback on Teacher-Forced Pass

`_compute_batch_logprobs()` already runs a teacher-forced VLM forward over [prompt + completion] to get per-token log-probs. Adding `output_hidden_states=True` to this pass gives hidden states at **every position** in the sequence — including all three value estimation points — for free (no additional forward pass).

Extract:
- `h_obs = hidden_states[-1][:, prompt_len - 1, :]` — last prompt token
- `h_coc = hidden_states[-1][:, cot_end_pos, :]` — `<cot_end>` position
- `h_traj = hidden_states[-1][:, traj_start:traj_end, :]` — all trajectory positions

### Memory Cost

Hidden states are extracted under `torch.no_grad()` and moved to CPU immediately.

Per completion: `(2 + T_traj) × hidden_dim × 4 bytes`
= `(2 + 64) × 4096 × 4 ≈ 1 MB`

With G=8 generations and batch_size=4: ~32 MB total in CPU RAM. Negligible.

The `output_hidden_states=True` flag does add GPU memory during the forward pass (stores all layers' activations). For Qwen3-VL with 28 layers, this roughly doubles activation memory for that pass. Since this already runs under `torch.no_grad()`, it should fit within the existing memory budget, but may need monitoring on the 10GB MIG.

---

## Training the Value Head

### Loss Function

MSE between predicted values and **returns-to-go** (actual cumulative reward from that point onward):

```
L_value = (1/N) Σ (V(s) - G(s))²
```

Where G(s) is the return-to-go:
- G(s_obs) = R_reasoning + R_consistency + Σ r_t  (total return)
- G(s_coc) = Σ r_t  (remaining trajectory return)
- G(s_traj_t) = Σ_{k=t}^{T} r_k  (trajectory return from timestep t onward)

### Training Schedule

Keep the existing two-stage approach:
- **Stage 0** (pretrain): value head trains alone on stashed (h, G) pairs; policy frozen
- **Stage 1** (joint): value head trains alongside GRPO policy updates

The pretrain phase is more important now since the value head needs to learn three levels instead of one.

### Optimizer

Separate Adam optimizer (existing design), `lr ≈ 1e-4`. The value head never receives gradients through the VLM backbone — it trains on detached hidden states.

---

## Integration Points in AlpamayoGRPOTrainer

### Methods to Modify

| Method | Change |
|---|---|
| `_compute_batch_logprobs` | Add `output_hidden_states=True`, return hidden states at segment boundaries |
| `_generate_single_turn` | Stash per-level hidden states and per-segment rewards (replaces h0-only stash) |
| `_calculate_rewards` | Decompose trajectory reward into per-timestep values |
| `_generate_and_score_completions` (new override) | Compute (B, T) advantages using value head + TD/GAE, replace scalar advantages |
| `_train_value_head_step` | Train on all three levels jointly |
| `value_head.py` | Rename to `SegmentValueHead`, accept (B, T, D) inputs |

### Data Flow

```
_generate_single_turn
    │
    ├─ VLM generates completions
    ├─ _compute_batch_logprobs (+ output_hidden_states)
    │     └─ extract h_obs, h_coc, h_traj_1..T
    │
    └─ stash: {h_obs, h_coc, h_traj, R_reasoning, R_consistency, r_traj_1..T}

_generate_and_score_completions (override)
    │
    ├─ super() → scalar rewards, scalar advantages
    ├─ value_head(h_obs) → V_obs
    ├─ value_head(h_coc) → V_coc
    ├─ value_head(h_traj) → V_traj_1..T
    ├─ TD/GAE → per-segment advantages
    ├─ group-normalize per segment type
    └─ output["advantages"] = (B, T) tensor

_compute_loss
    │
    ├─ TRL's standard clipped surrogate with (B, T) advantages
    └─ _train_value_head_step (MSE on stashed data)
```

---

## Path to GRPO–PPO Hybrid

This design naturally extends toward full PPO. The key insight: once you have a value network that bootstraps at the token level, the boundary between "GRPO with a learned baseline" and "PPO" becomes a spectrum.

### What Makes This Still GRPO

- **No value gradient through the VLM backbone** — the value head trains on detached hidden states
- **Group-relative normalization** — advantages are normalized within G completions per prompt
- **No online value updates during rollout** — values are computed once after generation, not iteratively

### What Would Make This PPO

The following changes are each independent and can be adopted incrementally:

1. **Backprop value gradients through the VLM** — instead of detached hidden states, share the backbone between policy and value head. This couples the representations but can improve value estimates. Requires careful loss balancing (`L = L_policy + c · L_value`).

2. **Drop group normalization** — use raw TD/GAE advantages instead of normalizing within the group of G completions. The value baseline provides variance reduction, so group normalization becomes optional rather than essential.

3. **Online value updates** — update the value head between GRPO iterations within a single training step, not just at step boundaries. This improves value accuracy for the current policy.

4. **GAE everywhere** — extend per-token GAE from trajectory tokens to CoC tokens as well, giving every token its own advantage. This requires a value estimate at every token position (which the hidden states already support).

### The Spectrum

```
Pure GRPO                          This design                         Full PPO
─────────────────────────────────────────────────────────────────────────────────
scalar reward     segment rewards + value baseline     per-token GAE + shared backbone
broadcast adv     segment advantages (B,T)             full token-level advantages
no critic         detached critic                      end-to-end critic
group-relative    group-relative per segment           optional group normalization
```

Each step rightward adds more credit assignment precision at the cost of training complexity and stability. The segment-level design is the pragmatic middle ground: it captures the most important structural credit assignment (reasoning vs. trajectory) without the instability risks of full PPO (value-policy interference, reward hacking through the critic).

### When to Move Further Toward PPO

- If segment-level advantages show clear improvement over scalar GRPO but per-trajectory-token advantages plateau → the value head may need backbone gradients (step 1)
- If group normalization suppresses useful signal → try raw advantages with the value baseline (step 2)
- If value head lags behind rapid policy changes → add online updates (step 3)

---

## Configuration

Extends the existing `value_head` config block in `grpo_default.yaml`:

```yaml
value_head:
  enabled: true
  hidden_dim: 4096
  lr: 1e-4
  pretrain_steps: 50
  save_path: outputs/value_head.pt
  load_path: null
  # New fields for segment-level value
  segment_level: true          # false = legacy SceneValueHead behavior
  gamma: 1.0                   # discount factor (1.0 for short sequences)
  gae_lambda: 0.95             # GAE trace decay
  normalize_per_segment: true  # group-normalize advantages per segment type
```

---

## Metrics

New TensorBoard metrics alongside existing `value_head/loss`:

| Metric | Description |
|---|---|
| `value_head/loss_obs` | MSE at observation level |
| `value_head/loss_coc` | MSE at CoC level |
| `value_head/loss_traj` | MSE at trajectory token level (averaged) |
| `value_head/v_obs_mean` | Mean predicted V(obs) |
| `value_head/v_coc_mean` | Mean predicted V(coc) |
| `value_head/v_traj_mean` | Mean predicted V(traj) across tokens |
| `advantages/coc_mean` | Mean CoC segment advantage |
| `advantages/coc_std` | Std of CoC advantages within groups |
| `advantages/traj_mean` | Mean trajectory token advantage |
| `advantages/traj_std` | Std of trajectory advantages within groups |
