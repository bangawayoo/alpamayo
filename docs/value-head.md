# Segment-Level Value Head for Alpamayo-R1

## Motivation

Standard GRPO assigns a single scalar reward to an entire rollout, then broadcasts it as the advantage for every token. This means a good trajectory boosts all CoC tokens equally, and good reasoning boosts all trajectory tokens equally — there is no credit assignment between segments.

We want segment-level advantages: the value network estimates expected return at three points in the sequence, enabling TD-style advantage computation that attributes credit to the right part of the generation.

This design stays within `AlpamayoGRPOTrainer` (no separate PPO trainer needed) because the core optimization is still group-relative clipped surrogate. The value network serves as a **learned baseline for variance reduction**, not a PPO-style critic that changes the objective. However, as noted below, bootstrapping token-level values makes this a natural stepping stone toward a full GRPO–PPO hybrid.

---

## Current State (SegmentValueHead — scene-level advantage)

**Status: Implemented** (`feat/value-head-advantage` branch)

`SegmentValueHead` (renamed from `SceneValueHead`, backward-compat alias kept) maps VLM hidden states to scalar value estimates. Currently used at the scene level (V_obs only):

```
V(observation) = MLP(h_obs)
A_i = r_i - V(s_obs)          # advantage for each completion
```

Key design decisions in the current implementation:

- **No group normalization** — the learned baseline replaces GRPO's group-mean baseline entirely. Per-group normalization (`(A - mean) / std`) flattens scene difficulty: a rare good trajectory on a hard scene (std=0.3) gets a smaller normalized advantage than a slightly-above-average trajectory on an easy scene (std=0.01). Raw `r - V` advantages preserve this difficulty signal.
- **G samples still useful** — multiple completions per prompt provide (1) low-variance MC targets for value head training (`V_mc = mean(rewards)`), and (2) gradient variance reduction through batching.
- **`advantage_enabled` config flag** — set `false` for auxiliary-only mode (value head trains and logs but doesn't affect policy gradients).
- **Two-stage training preserved** — stage 0 pretrains the value head with policy frozen; stage 1 runs both together. During stage 0, advantages are not replaced (policy receives zero loss).

### Files changed

| File | Change |
|---|---|
| `value_head.py` | `SceneValueHead` → `SegmentValueHead`, accepts `(B, T, D)` for future segment-level use |
| `rollout.py` | `_generate_and_score_completions` override, `_compute_value_advantages` method, advantage stash buffers |
| `grpo_default.yaml` | `advantage_enabled: true` field |
| `tests/test_training.py` | 13 new tests (SegmentValueHead shapes + advantage computation + no-group-norm verification) |

### Observed training behavior

- Policy loss goes negative — this is expected. The clipped surrogate loss `L = -ratio * A * log_p` is negative when positive advantages reinforce high-probability tokens.
- Value head loss starts high (~0.6) and converges as V(scene) learns to predict expected return.
- `advantages/mean` is non-zero (unlike group-normalized GRPO where it's always ~0). This is correct — it reflects `mean(r) - mean(V)`, which converges to zero as the value head improves.

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

To compute segment-level advantages, we need to assign rewards to the correct segment based on **causal determination** — a reward belongs to the segment after which it is fully determined.

### What's determined at each state

```
s_obs                    s_coc                          s_T (terminal)
  │                        │                              │
  │  [CoC tokens]          │  [traj tokens]               │
  │                        │                              │
  │  R_reasoning: NO       │  R_reasoning: YES (fixed)    │  all determined
  │  R_consistency: NO     │  R_consistency: NO           │
  │  R_trajectory: NO      │  R_trajectory: NO            │
```

- `R_reasoning` is fully determined by the CoC text → **CoC segment reward**.
- `R_consistency` measures agreement between CoC and trajectory → requires BOTH segments → **trajectory segment reward** (assigned causally to the later segment).
- `R_trajectory` (ADE) depends on the generated trajectory → **trajectory segment reward**.

> **Note on R_reasoning**: The current `reasoning_quality_reward` is a noisy, rule-based heuristic (checks for causal connectors, driving vocabulary, appropriate length, no repetition). It does not measure actual reasoning quality — a fluent but factually wrong CoC can score high. This means the CoC segment advantage `A_coc` may receive unreliable signal from `R_reasoning`. However, the `V(s_coc) - V(s_obs)` term provides a complementary, learned signal: if the CoC text leads to better-than-expected trajectory outcomes, `V(s_coc) > V(s_obs)` regardless of the heuristic score. As R_reasoning improves (e.g., via a learned reward model), the CoC advantage becomes more informative.

### Reward assignment by segment

| Segment | Reward | Why |
|---|---|---|
| CoC | `w_reason · R_reasoning` | Only reward fully determined by CoC text alone |
| Trajectory | `w_traj · R_trajectory + w_consist · R_consistency` | Both require the generated trajectory |

### Per-timestep trajectory rewards

The existing `trajectory_quality_reward` computes a scalar ADE. For per-token advantages, decompose into per-timestep rewards:

```python
# Current: single scalar
ade = l2_per_step.mean()
reward = max(0, 1 - ade / threshold)

# New: per-timestep rewards
r_t = w_traj · max(0, 1 - l2_per_step[t] / threshold)   # one per trajectory token
```

`R_consistency` is binary and doesn't decompose per-timestep. Assign it as a **terminal reward** at the last trajectory token (standard RL convention):

```
r_t = w_traj · per_step_ade_reward_t                              for t = 1..T-1
r_T = w_traj · per_step_ade_reward_T + w_consist · R_consistency  terminal
```

Each trajectory token `<iN>` maps to a specific timestep in the decoded (T, 3) trajectory, so the mapping from token index to timestep reward is direct.

---

## Value Targets (Returns-to-Go)

The value function V(s) predicts expected future return from state s. The training target G(s) is the actual return-to-go:

```
G(s_obs)    = w_reason · R_reasoning + Σ_t r_t           # total return
G(s_coc)    = Σ_t r_t                                    # remaining: trajectory + consistency
G(s_traj_t) = Σ_{k=t}^{T} r_k                           # remaining per-step (last includes consistency)
```

### Per-completion vs per-scene targets

| Level | h source | Target | Samples per scene (G=8, T=64) |
|---|---|---|---|
| V(s_obs) | Same h_obs for all G | MC mean over G (low variance) | 1 |
| V(s_coc) | Different h_coc per completion | Per-completion trajectory return | 8 |
| V(s_traj_t) | Different h per completion per token | Per-completion return-to-go | 512 |

V(s_obs) uses MC averaging because all G completions share the same h_obs — the value head can't distinguish between them, so averaging reduces target variance. V(s_coc) and V(s_traj_t) have distinct hidden states per completion and must use per-completion targets.

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
A_coc = w_reason · R_reasoning + γ · V(s_coc) - V(s_obs)
```

This captures two signals:
1. `w_reason · R_reasoning` — did the reasoning text satisfy the heuristic quality checks? (noisy)
2. `γ · V(s_coc) - V(s_obs)` — did this CoC text shift expected trajectory outcome up or down? (learned, potentially more reliable)

Even with a noisy `R_reasoning`, the `V(s_coc) - V(s_obs)` term provides useful credit assignment: a CoC that produces confident, correct reasoning will yield `V(s_coc) > V(s_obs)` because the model expects better trajectory quality, regardless of whether the heuristic catches it.

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

### Group Normalization — Deliberately Omitted

Standard GRPO normalizes within a group: `(A - mean) / (std + ε)`. We intentionally skip this for the value-head baseline because:

1. **Difficulty flattening**: per-group std normalization forces every scene to unit variance. A hard scene (reward std=0.3) and an easy scene (std=0.01) get the same gradient magnitude — the model can't learn that rare successes on hard scenes are more valuable.
2. **V cancels under centering**: since V(scene) is identical for all G samples from the same scene, `A_i - mean(A) = r_i - mean(r)` — the value head provides no benefit after group centering.
3. **Variance reduction via batching**: G samples per prompt still reduce gradient variance through averaging, without needing explicit normalization.

For future segment-level advantages (CoC vs trajectory), per-segment-type normalization may be reconsidered to prevent the trajectory signal from dominating simply because it has more tokens. But this should be evaluated empirically against raw advantages.

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
- **G completions per prompt** — multiple rollouts per scene provide gradient variance reduction and MC value targets (group normalization is dropped, but the group structure remains)
- **No online value updates during rollout** — values are computed once after generation, not iteratively

### What Would Make This PPO

The following changes are each independent and can be adopted incrementally:

1. **Backprop value gradients through the VLM** — instead of detached hidden states, share the backbone between policy and value head. This couples the representations but can improve value estimates. Requires careful loss balancing (`L = L_policy + c · L_value`).

2. ~~**Drop group normalization**~~ — **Done.** Raw `r - V` advantages are used. The value baseline provides centering, and preserving per-scene variance gives harder scenes proportionally more gradient signal.

3. **Online value updates** — update the value head between GRPO iterations within a single training step, not just at step boundaries. This improves value accuracy for the current policy.

4. **GAE everywhere** — extend per-token GAE from trajectory tokens to CoC tokens as well, giving every token its own advantage. This requires a value estimate at every token position (which the hidden states already support).

### The Spectrum

```
Pure GRPO              Current (scene-level)        Planned (segment-level)        Full PPO
────────────────────────────────────────────────────────────────────────────────────────────
scalar reward          scalar reward + V baseline   segment rewards + V baseline   per-token GAE + shared backbone
broadcast adv          A = r - V(scene) (B,)        segment advantages (B,T)       full token-level advantages
no critic              detached critic               detached critic                end-to-end critic
group-relative         no group norm ✓               TBD per segment type           optional group normalization
```

Each step rightward adds more credit assignment precision at the cost of training complexity and stability. The segment-level design is the pragmatic middle ground: it captures the most important structural credit assignment (reasoning vs. trajectory) without the instability risks of full PPO (value-policy interference, reward hacking through the critic).

### When to Move Further Toward PPO

- If scene-level `r - V` advantages improve over scalar GRPO → proceed to segment-level (CoC + trajectory) advantages
- If segment-level advantages plateau → the value head may need backbone gradients (step 1)
- If value head lags behind rapid policy changes → add online updates (step 3)

---

## Configuration

The `value_head` config block in `grpo_default.yaml`:

```yaml
value_head:
  enabled: true
  hidden_dim: 4096
  lr: 1e-5
  pretrain_steps: 0            # stage 0: steps where only value head trains (0 = skip)
  save_path: outputs/value_head.pt
  load_path: null
  advantage_enabled: true      # replace GRPO group-mean baseline with V(scene)
                               # set false for auxiliary-only mode (logging, no policy effect)
```

Future fields for segment-level value (not yet implemented):

```yaml
  segment_level: true          # false = scene-level only (current behavior)
  gamma: 1.0                   # discount factor (1.0 for short sequences)
  gae_lambda: 0.95             # GAE trace decay
```

---

## Metrics

### Currently logged (scene-level)

| Metric | Description |
|---|---|
| `value_head/loss` | MSE between V(scene) and MC target |
| `value_head/pred_mean` | Mean predicted V(scene) |
| `value_head/target_mean` | Mean MC target (avg reward over G completions) |
| `value_head/scenes_per_step` | Number of scenes consumed per training step |
| `value_head/mc_group_size` | Average group size for MC aggregation |
| `value_head/pretrain_steps_remaining` | Stage 0 countdown |
| `advantages/v_baseline_mean` | Mean V(scene) used as baseline |
| `advantages/mean` | Mean advantage (non-zero; converges to 0 as V improves) |
| `advantages/std` | Std of advantages (reflects scene difficulty spread) |

### Planned (segment-level)

| Metric | Description |
|---|---|
| `value_head/loss_obs` | MSE at observation level |
| `value_head/loss_coc` | MSE at CoC level |
| `value_head/loss_traj` | MSE at trajectory token level (averaged) |
| `advantages/coc_mean` | Mean CoC segment advantage |
| `advantages/traj_mean` | Mean trajectory token advantage |
