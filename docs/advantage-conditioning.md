# Multi-Level Advantage Conditioning for Alpamayo-R1

Inspired by [RECAP](https://arxiv.org/abs/2511.14759) (Physical Intelligence, 2025), adapted for VLA trajectory generation with Chain-of-Causation (CoC) reasoning.

---

## Motivation

GRPO training produces rollouts of varying quality — some completions exhibit excellent reasoning followed by poor trajectories, others stumble through reasoning but nail the trajectory. Standard GRPO uses these rollouts for policy gradient updates, weighting tokens by scalar advantages. However, GRPO suffers from training instability, which is aggravated when trained with larger diffusion policy. To avoid this, RECAP train our policy via pure supervised fashion, but **condition the policy on whether each segment was good or bad**, training it to distinguish high-quality behavior.

RECAP showed this works at a single level: binarize the advantage into a text token (`"Advantage: positive/negative"`), train the model on both conditional and unconditional objectives, then condition on `"positive"` at inference. We extend this to **three semantic levels** matching Alpamayo-R1's generation structure — observation, reasoning, and trajectory — giving the model fine-grained conditioning signals at each stage of generation.

### Why Advantage Conditioning Instead of (or With) Policy Gradients

| Aspect | Policy gradient (GRPO) | Advantage conditioning (this) |
|---|---|---|
| How signal enters | Advantage multiplies log-prob gradient | Advantage becomes input conditioning token |
| Failure modes | High variance; reward hacking via log-prob manipulation | Mode collapse to "positive" distribution if training is imbalanced |
| Compatibility with diffusion/flow | Difficult (requires differentiating through sampling) | Natural (just another input token) |
| Data efficiency | Each rollout contributes one gradient direction | Each rollout provides a labeled (state, quality) pair for SFT |
| Multi-level credit | Requires per-token advantage computation | Natural: each segment gets its own conditioning token |

The key insight: advantage conditioning converts the RL problem into a **conditional generation** problem. The model learns `p(completion | quality_labels)`, and at inference we simply set all labels to "positive."

---

## Algorithm Overview

### Three-Phase Loop

```
┌─────────────────────────────────────────────────────────────────────┐
|  Phase 0: Value-head training                                       |
│  Phase 1: ROLLOUT                                                   │
│  ────────────────                                                   │
│  Rollout on unseen dataset 
|  For each scene, generate G completions from the current policy.    │
│  Each completion = CoC reasoning text + 64 trajectory tokens.       │
│                                                                     │
│  Phase 2: EVALUATE                                                  │
│  ─────────────────                                                  │
│  Score each completion using the segment value head + reward         │
│  functions. Compute per-segment advantages. Binarize into           │
│  conditioning labels: {obs, traj} × {positive, negative}.          │
│                                                                     │
│  Phase 3: TRAIN (SFT with advantage conditioning)                   │
│  ──────────────────────────────────────────────────                  │
│  For each completion, prepend segment-level conditioning tokens     │
│  and train via teacher-forced cross-entropy. The model learns both  │
│  conditional p(x | labels) and unconditional p(x) objectives.       │
│                                                                     │
│  Repeat.                                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Comparison with RECAP

**Differences:**

| RECAP (Physical Intelligence) | Ours |
|---|---|
| Single binary advantage token | Two segment-level tokens (obs, traj) |
| Expert demonstrations + interventions | Self-play only (policy rollouts) |
| Flow-matching action head | Autoregressive VLM (text + discrete trajectory tokens) |
| Distributional value function (201 bins) | Segment value head (MSE, three levels) |

**Shared mechanisms:**

- α-weighted dual loss (conditional + unconditional objectives)
- Classifier-free guidance at inference to amplify conditioning signal

---

## Segment-Level Conditioning Tokens

### Token Design

We introduce four new special tokens, two per segment level:

```
<|adv_obs_pos|>    <|adv_obs_neg|>     # Scene-level: was this completion good given the scene?
<|adv_traj_pos|>   <|adv_traj_neg|>    # Traj-level: was the trajectory accurate given the reasoning?
```

These are **not** added to the tokenizer vocabulary. Instead, they use sentinel IDs just past the VLM's `vocab_size` boundary. A small trainable `AdvantageEmbedding` module (4 × hidden_size parameters) intercepts these sentinel IDs via forward hooks on the VLM's input embedding layer — a pre-hook clamps them to safe values before `embed_tokens`, and a post-hook replaces those positions with learned embeddings. This avoids `resize_token_embeddings()`, which would create random rows frozen by LoRA.

> **Why not a CoC-level token?** The CoC advantage would be computed at state s_coc — after the full reasoning trace is generated but before the first trajectory token. Since V(s_coc) and V(s_traj_0) see nearly identical information (observation + complete CoC text), A_coc and A_traj at j=0 would be redundant. The trajectory-level advantage already captures whether the CoC set up good conditions for the trajectory — a positive A_traj implicitly reflects good reasoning.

### Placement in the Sequence

Each conditioning token is placed immediately before the segment it conditions, matching the causal scope of the advantage it represents:

```
[system prompt] [image tokens] [history trajectory] [task instruction]
  ↓ adv_obs: conditions entire completion (CoC + trajectory)
[<|adv_obs_pos|>]
  ↓ CoC generation
[<|cot_start|> ... reasoning text ... <|cot_end|>]
  ↓ adv_traj: conditions only trajectory (placed after CoC, before trajectory)
[<|adv_traj_neg|>]
  ↓ trajectory generation
[<|traj_future_start|> <i0> ... <i63> <|traj_future_end|>]
```

**Why this placement?** With autoregressive attention, each token can only attend to preceding tokens. Placing `adv_traj` after the CoC ensures it only influences trajectory generation — not reasoning. This matches the causal structure: A_traj is computed after the CoC is already generated, so it should not leak trajectory quality information into CoC token predictions. Placing it before the CoC would let the model generate different reasoning depending on whether the trajectory label is positive or negative, which is causally backwards.

`adv_obs` is placed before the CoC because A_obs evaluates the entire completion (CoC + trajectory), so it appropriately conditions both segments.

At inference, both tokens are set to positive:

```
[prompt] [<|adv_obs_pos|>] [CoC generation ...] [<|adv_traj_pos|>] [trajectory generation ...]
```

During generation, `adv_traj_pos` is inserted after the model generates `<|cot_end|>` (or `<|traj_future_start|>`) and before trajectory token sampling begins.

### Why Two Levels, Not One

A single binary token conflates two independent quality axes. Consider these cases:

| Scenario | Single token | Two tokens |
|---|---|---|
| Good completion on easy scene | `pos` | `obs_pos, traj_pos` — model conditions on full success |
| Good completion on hard scene | `neg` (low absolute return) | `obs_neg, traj_pos` — model learns this was a good attempt despite scene difficulty |
| Bad trajectory despite good scene | `neg` | `obs_pos, traj_neg` — model learns what bad trajectories look like on scenes that should be easy |
| Everything bad | `neg` | `obs_neg, traj_neg` — hard scene, bad execution |

The observation-level token captures **scene difficulty**. A scene with `obs_neg` tells the model "this is a hard scene" — at inference, conditioning on `obs_pos` steers the model toward behaviors it associates with easier-to-solve scenes (more cautious planning, simpler maneuvers), which may generalize as "do the best you can."

---

## Advantage Computation

### Using the Existing Segment Value Head

The segment value head (already implemented in `value_head.py`) provides value estimates at two levels:

```
V(s_obs)    = SegmentValueHead(h_obs,  level=0)    # E[total return | observation]
V(s_traj_j) = SegmentValueHead(h_traj, level=2)    # E[remaining return | obs + CoC + traj up to j]
```

Each V predicts the **expected remaining return** from that information state. V(s_obs) sees only the scene; V(s_traj_j) sees the scene, full CoC reasoning, and trajectory up to step j.

### Temporal Structure

Rewards are earned at different points in the generation sequence:

```
s_obs ──[generate CoC]──► s_coc ──[generate traj_1]──► s_traj_1 ──► ... ──► s_traj_T ──► terminal
                           ↑            ↑                                        ↑
                     R_reasoning    r_traj_1, ..., r_traj_T              R_consistency
                     earned here    earned per step                      earned here
```

- **R_reasoning**: scored once the full CoC text is generated (at s_coc)
- **r_traj_t**: per-timestep trajectory quality (e.g., displacement error at step t)
- **R_consistency**: requires both CoC and full trajectory, scored at terminal

### Returns-to-Go at Each State

The actual return from each state is the sum of all future rewards from that point:

```
G(s_obs)    = w_reason · R_reasoning + w_traj · Σ_t r_t + w_consist · R_consistency
G(s_traj_j) = w_traj · Σ_{t=j}^{T} r_t + w_consist · R_consistency  (remaining trajectory return)
```

### Per-Segment Advantage Definitions

Advantages are computed as **actual return minus value baseline** at each information level.

**Observation-level advantage** (completion quality relative to scene baseline):

```
A_obs = G(s_obs) - V(s_obs)
```

Measures: given only the observation, was this entire completion (CoC + trajectory) better or worse than expected? Captures both scene difficulty and overall completion quality.

**Trajectory-level advantage** (remaining trajectory quality, mean over timesteps):

```
A_traj_j = G(s_traj_j) - V(s_traj_j)
         = (w_traj · Σ_{t=j}^{T} r_t + w_consist · R_consistency) - V(s_traj_j)

A_traj   = mean_j(A_traj_j)
```

Measures: at step j of trajectory generation, was the remaining trajectory better or worse than expected given the observation, CoC reasoning, and trajectory so far? Since the conditioning token is a single binary indicator for the entire trajectory segment, we use the mean over all timesteps.

> **Note on alternative formulations:** The value-head design doc (`docs/value-head.md`) also describes a TD bootstrapping formulation and per-token GAE for trajectory advantages. Those formulations trade off bias for variance. For advantage conditioning — where advantages are binarized into discrete labels — the return-minus-baseline formulation is preferred. See `docs/value-head.md` for a full comparison.

### Binarization

Each advantage is binarized using a per-level threshold:

```
I_obs  = 1  if A_obs  > ε_obs   else 0
I_traj = 1  if A_traj > ε_traj  else 0
```

**Threshold selection:** Following RECAP, set `ε` at the k-th percentile of the advantage distribution (e.g., k=30), recomputed periodically. This ensures roughly 70% of completions receive "positive" labels, providing enough positive signal for the SFT objective.

A higher k (e.g., 50) is more selective but risks training on too few positive examples per level. A lower k (e.g., 20) is more permissive but dilutes the quality signal. We start with k=30 and tune per level.

```python
# Computed over a buffer of recent advantages
ε_obs  = np.percentile(recent_A_obs,  k_obs)
ε_traj = np.percentile(recent_A_traj, k_traj)
```

---

## Training Objective

### Dual Loss (Conditional + Unconditional)

Following RECAP's formulation, the training loss combines a conditional and an unconditional term. However, unlike RECAP — which has expert demonstrations anchoring the data distribution — we operate in pure self-play where a significant fraction of rollouts are low quality. Training the unconditional path on bad completions would teach the model to reproduce its own mistakes.

**Key adaptation: restrict the unconditional path to positive-advantage completions only.**

```
If completion has all-positive advantages (I_obs=1 ∧ I_traj=1):
    L = -log π_θ(x | prompt) - α · log π_θ(x | I_pos, prompt)     # both paths

Otherwise:
    L = -α · log π_θ(x | I_obs, I_traj, prompt)                    # conditional only
```

The rationale:
- The **unconditional** distribution `π(x | prompt)` sees only high-quality completions, so it becomes a "best-of" policy. This is what CFG uses as its baseline — it should represent good default behavior, not average (mediocre) behavior.
- The **conditional** path sees all completions with their true labels. Negative-labeled completions teach the model what bad behavior looks like under negative conditioning, which is essential for CFG to create contrast.
- At inference, CFG computes `logits_uncond + β · (logits_pos - logits_uncond)`. If `logits_uncond` already represents good behavior, the positive-conditioned signal amplifies the *best* behavior beyond the already-good baseline.

> **Why not RECAP's approach?** RECAP trains unconditionally on everything because their data is dominated by expert demonstrations — the unconditional distribution is naturally high-quality. In self-play, the unconditional distribution reflects the current policy's average quality, which may be poor. Training unconditionally on negative completions actively degrades the baseline that CFG relies on.

### Conditioning Dropout for Classifier-Free Guidance

During training, randomly drop the conditioning tokens with probability `p_drop` (e.g., 0.3). Combined with the positive-only unconditional rule:

```python
is_all_positive = (I_obs == 1) and (I_traj == 1)

if is_all_positive and random() < p_drop:
    # Unconditional: no advantage tokens (only for positive completions)
    input = [prompt] + [coc_tokens] + [traj_tokens]
else:
    # Conditional: advantage tokens inserted at their causal positions
    input = [prompt] + [I_obs] + [coc_tokens] + [I_traj] + [traj_tokens]
```

With this scheme, the effective unconditional training fraction is `p_drop × frac_all_positive`. If ~40% of completions are all-positive and `p_drop = 0.3`, about 12% of training examples train the unconditional path — enough for CFG but not so much that it dominates.

### Per-Level Conditioning Dropout (Optional)

For finer control, each level's token can be independently dropped:

```python
include_obs  = random() > p_drop_obs
include_traj = random() > p_drop_traj
```

This enables classifier-free guidance at each level independently during inference, at the cost of more combinations during training.

---

## Inference: Classifier-Free Guidance

At inference, we use classifier-free guidance (CFG) to amplify the conditioning signal:

```
logits_final = logits_uncond + β · (logits_cond - logits_uncond)
```

Where:
- `logits_uncond`: forward pass without conditioning tokens
- `logits_cond`: forward pass with all-positive conditioning tokens
- `β ≥ 1`: guidance strength (β=1 recovers standard conditional generation)

### Multi-Level CFG (if per-level dropout was used)

With independent per-level dropout, we can apply CFG per level:

```
logits = logits_uncond
       + β_obs  · (logits_{obs=pos}  - logits_uncond)
       + β_traj · (logits_{traj=pos} - logits_uncond)
```

This requires 3 forward passes (1 unconditional + 2 per-level). In practice, the single-block CFG (both tokens present or absent) is likely sufficient.

### Interaction with the Action Expert (Flow Matching)

The action expert receives the VLM's KV cache as a static prefix and appends 64 action embeddings as a continuation. At inference, the KV cache contains all tokens from the prompt through `<|traj_future_start|>`. The expert cross-attends to this entire context during each diffusion step.

**How conditioning tokens enter the expert's context:**

`adv_obs` is placed before the CoC in the VLM's input, so it is naturally part of the KV cache from VLM generation — no special handling needed.

`adv_traj` is placed after the CoC but before the trajectory. Since VLM generation stops at `<|traj_future_start|>`, the `adv_traj` token is not part of the generated sequence. Instead, it is **injected into the KV cache** via one extra VLM forward step before the expert takes over:

```
VLM generation produces KV cache:
  [prompt] [<adv_obs_pos>] [CoC text ...] [<traj_future_start>]
                                                                 ↑ generation stops here

Inject adv_traj by running one VLM forward step with <adv_traj_pos> token ID:
  [prompt] [<adv_obs_pos>] [CoC text ...] [<traj_future_start>] [<adv_traj_pos>]
                                                                                  ↑ new offset

Expert receives updated KV cache and appends 64 action embeddings:
  [...existing cache...] [<adv_traj_pos>] | [action_embed_1] ... [action_embed_64]
                                          ↑ expert starts here
```

This is a single-token VLM forward with KV cache reuse (negligible cost). The expert then sees `adv_traj_pos` as the last VLM context token — matching its position during SFT training where `adv_traj` appeared in the token sequence between CoC and trajectory.

For the unconditional CFG pass, `adv_traj` is simply not injected — the expert sees the original KV cache ending at `<|traj_future_start|>`.

---

## Self-Play Data Collection

Unlike RECAP, which mixes expert demonstrations and autonomous rollouts, we operate in a **pure self-play** setting: all training data comes from the policy's own rollouts.

### Iterative Training Loop

```
Iteration 0:
  - Start from base policy π_0 (pre-trained VLA)
  - Generate rollouts from π_0 on training scenes
  - Compute segment advantages using value head
  - Train π_1 via advantage-conditioned SFT on π_0's rollouts

Iteration n:
  - Generate rollouts from π_n
  - Update value head on new rollouts
  - Compute segment advantages
  - Train π_{n+1} via advantage-conditioned SFT on π_n's rollouts
```

### Mixing Policy for Stability

Pure on-policy self-play can suffer from distribution collapse: the model only sees its own outputs, reinforcing existing biases. Mitigation strategies:

1. **Temperature-scaled rollouts**: Generate with temperature > 1.0 (e.g., 1.2) during data collection to increase exploration, while training at temperature 1.0.

2. **Replay buffer**: Mix current-iteration rollouts with rollouts from prior iterations (e.g., 70% current, 30% historical). The advantage labels for historical rollouts should be recomputed using the current value head.

3. **Base policy anchoring**: Include a fraction of rollouts from the original π_0 (with recomputed advantages) to prevent the policy from drifting too far. This is analogous to RECAP's demonstration data, but from the model's own initial capabilities.

4. **KL regularization** (optional): Add `β_kl · KL(π_θ || π_ref)` to the loss to keep the trained policy close to a reference. This can be the base policy or the previous iteration.

### Reset to Checkpoint (from RECAP)

RECAP resets to the pre-trained checkpoint each iteration to prevent drift. We adopt this:

```
Each iteration:
  1. Load π_0 (base pre-trained weights)
  2. Generate rollouts from π_n (the current best policy, kept separately)
  3. Train π_0 → π_{n+1} using advantage-conditioned SFT on accumulated data
  4. π_n ← π_{n+1}
```

This prevents compounding errors from sequential fine-tuning while still benefiting from improving data quality.

---

## Value Head Training

The segment value head trains alongside (or ahead of) the policy, using the same rollout data.

### Value Head Update Schedule

```
Separate loop:
  Stage 0: Pre-train value head on initial rollouts (policy frozen).
  Within each iteration:
    1. Generate all rollouts
    2. Train value head to convergence on rollout data
    3. Compute advantages using converged value head
    4. SFT training with fixed advantages
  Pro: Clean separation, stable advantages. Con: Sequential.
```

### Value Targets

Same as the existing value head design:

```
G(s_obs)    = w_reason · R_reasoning + w_traj · Σ_t r_t + w_consist · R_consistency   # total return
G(s_traj_t) = w_traj · Σ_{k=t}^{T} r_k + w_consist · R_consistency                   # return-to-go per trajectory token
```

---

## Implementation Plan

### Phase 1: Token and Data Infrastructure

1. **Assign sentinel token IDs** via `compute_advantage_token_ids(vocab_size)`:
   - `<|adv_obs_pos|>`, `<|adv_obs_neg|>`, `<|adv_traj_pos|>`, `<|adv_traj_neg|>`
   - IDs are `vocab_size + 0..3` — no tokenizer modification or embedding resize needed
   - `AdvantageEmbedding` (trainable `nn.Embedding(4, hidden_size)`) is attached to the VLM's embed_tokens layer via forward hooks, ensuring gradients flow through the advantage token embeddings even when LoRA freezes the base model

2. **Rollout data structure**: Extend the rollout output to include per-segment advantage labels alongside existing rewards and log-probs.

3. **Advantage buffer**: Maintain a rolling buffer of per-level advantages for percentile-based threshold computation.

### Phase 2: Training Pipeline

4. **Advantage-conditioned dataset builder**: Given rollouts + advantage labels, construct training examples with conditioning tokens prepended. Implement conditioning dropout.

5. **SFT training loop**: Standard teacher-forced cross-entropy, but with conditioning tokens in the input. No policy gradient computation needed.

6. **Value head training**: Either jointly or in a separate pre-training stage. Allow training the value head on the very first iteration.

### Phase 3: Inference

7. **Inference with CFG**: Two forward passes per step — unconditional and all-positive-conditioned. Combine via CFG with tunable β.

8. **Ablation harness**: Support for toggling individual conditioning levels on/off to measure per-level contribution.

### Phase 4: Iteration and Evaluation

9. **Self-play loop**: Script that runs rollout → value head training → advantage computation → SFT training → evaluation in a loop.

10. **Metrics**: Track per-level advantage distributions, conditioning accuracy (does conditioning on "positive" actually improve quality?), and downstream task metrics (ADE, reasoning quality).

---

## Hyperparameters

| Parameter | Symbol | Default | Notes |
|---|---|---|---|
| Conditioning dropout | p_drop | 0.3 | Probability of dropping all conditioning tokens |
| CFG strength | β | 1.5 | Start conservative; increase if conditioning signal is strong |
| Advantage percentile | k | 30 | Per-level binarization threshold |
| EMA decay (obs advantage) | α_ema | 0.99 | For scene-level running mean |
| Rollout temperature | τ_rollout | 1.2 | Higher = more exploration during data collection |
| Replay ratio | ρ | 0.3 | Fraction of historical rollouts in training mix |
| Conditional weight | α | 1.0 | Weight of conditional loss relative to unconditional |
| Completions per scene | G | 8 | Same as GRPO; more = better advantage estimates |
| KL penalty (optional) | β_kl | 0.0 | Set > 0 to regularize toward reference policy |

---

## Expected Benefits

1. **Stability**: Pure SFT is more stable than policy gradient methods — no clipping, no ratio explosion, no reward hacking.
2. **Multi-level credit**: The model receives explicit, interpretable signals about what went right/wrong at each stage.
3. **Test-time control**: CFG β allows post-hoc adjustment of how strongly the model follows the "positive" conditioning, without retraining.
4. **Data efficiency**: Every rollout contributes to both conditional and unconditional objectives — no rollouts are "wasted" on negative advantages.
5. **Simplicity**: The training loop is standard SFT with extra input tokens. No TRL dependency for the core training.


## Relationship to Existing Components

```
┌──────────────────────────────────────────────────────────────────┐
│                    Existing (value-head branch)                   │
│  SegmentValueHead ─── computes V(obs), V(coc), V(traj)          │
│  Reward functions ─── R_reasoning, R_trajectory, R_consistency   │
│  Advantage math   ─── return-minus-baseline, binarization        │
├──────────────────────────────────────────────────────────────────┤
│                    New (advantage conditioning)                   │
│  Sentinel tokens  ─── 4 IDs past vocab_size (no tokenizer mod)  │
│  AdvantageEmbed   ─── trainable side embedding via fwd hooks     │
│  Prompt builder   ─── prepend conditioning tokens to completion  │
│  Training loop    ─── dual SFT loss with conditioning dropout    │
│  Inference        ─── CFG with all-positive conditioning         │
│  Self-play loop   ─── iterative rollout → evaluate → train      │
└──────────────────────────────────────────────────────────────────┘
```

The advantage conditioning layer sits **on top of** the existing segment value head infrastructure. The value head computes advantages; this module consumes them as conditioning labels. Both can coexist with GRPO — the conditioning tokens don't interfere with policy gradient computation.
