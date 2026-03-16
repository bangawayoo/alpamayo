# Advantage Computation Approaches for Multi-Level Conditioning

This document compares two approaches for computing per-segment advantages in the advantage-conditioned SFT pipeline. Both use the same three-level value head (`SegmentValueHead` with levels obs/coc/traj) but differ in how advantages are derived from value predictions and actual returns.

---

## Setup (Common to Both)

The generation structure produces a sequence: **observation → CoC reasoning → trajectory tokens**.

The segment value head predicts at three information levels:
```
V(obs)     = SegmentValueHead(h_obs,   level=0)
V(coc)     = SegmentValueHead(h_coc,   level=1)
V(traj_j)  = SegmentValueHead(h_traj_j, level=2)
```

Reward functions produce per-completion scores:
```
R_reasoning    (rule-based CoC quality)
R_traj         (minADE-based trajectory quality)
R_consistency  (CoC-trajectory agreement)
r_t            (per-timestep trajectory rewards, t = 1..T)
```

Weighted total return:
```
R_total = w_reason · R_reasoning + w_traj · R_traj + w_consist · R_consistency
```

Both approaches produce `(A_obs, A_coc, A_traj)` per completion, which are then binarized via percentile thresholds into conditioning labels `(I_obs, I_coc, I_traj)`.

---

## Approach A: TD Bootstrapping / GAE

Models the generation as a sequential MDP where each segment is a "step":

```
s_obs  →  (generate CoC)  →  s_coc  →  (generate traj_1)  →  s_traj_1  →  ...  →  terminal
```

### Value Head Semantics

Each V predicts the **expected return from that state onward**, treating downstream generation as future steps:
- `V(obs)` = E[R_total | observation]
- `V(coc)` = E[R_remaining | observation + CoC] (expected trajectory + consistency return)
- `V(traj_j)` = E[R_remaining_from_j | observation + CoC + trajectory up to j]

### Advantage Formulas

**A_obs** — scene-level, uses EMA baseline across scenes:
```
ema_return = α · ema_return + (1 - α) · G(s_obs)
A_obs = G(s_obs) - ema_return
```
Where `G(s_obs)` is the total return (optionally MC-averaged over G completions).

**A_coc** — one-step TD residual across the CoC segment:
```
A_coc = w_reason · R_reasoning + γ · V(s_coc) - V(s_obs)
```

Interpretation: the immediate CoC reward plus the discounted predicted future from `s_coc`, minus what `s_obs` predicted. Measures whether the CoC reasoning *shifted the expected future value* more than the scene alone predicted, accounting for the immediate reasoning reward.

**A_traj** — GAE(γ, λ) over trajectory timesteps:
```
δ_t = r_t + γ · V(traj_{t+1}) - V(traj_t)       (TD residual at each step)
A_traj_t = Σ_k (γλ)^k · δ_{t+k}                  (GAE weighted sum)
A_traj = mean(A_traj_t)                            (aggregate for conditioning label)
```

At the terminal step T, `V(traj_{T+1}) = 0` and `r_T` includes the consistency reward.

**Special case (γ=1, λ=1):** GAE reduces to MC return minus baseline: `A_traj_t = Σ_{k=t}^T r_k - V(traj_t)`, which is the return-to-go minus value prediction at each step.

### Properties

- **Bias-variance tradeoff via λ:** At λ < 1, bootstrapping through V reduces variance at the cost of bias from value estimation error. At λ = 1, equivalent to MC return minus baseline (zero bias, higher variance).
- **Credit assignment within trajectory:** Per-timestep advantages via GAE allow identifying which trajectory segments were good/bad, not just the trajectory as a whole. This matters if combining with per-token policy gradient methods.
- **CoC advantage uses bootstrapping:** A_coc depends on V(s_coc)'s accuracy. If the value head hasn't converged, the bootstrapped V(s_coc) term introduces bias. However, the TD formulation also gives A_coc a natural interpretation as "immediate reward + predicted future value change."
- **Discount factor γ:** When γ < 1, downstream returns are downweighted, which can be useful if the value head is less reliable for far-future predictions. At γ = 1, all returns are weighted equally.
- **Compatible with GRPO:** The same advantage formulas work as per-token weights in policy gradient methods, not just as conditioning labels.

---

## Approach B: Return Minus Baseline

Models each value head as a **baseline predictor at a specific information level**. Advantages are the residual between actual returns and the baseline prediction.

### Value Head Semantics

Each V predicts the **expected total return given all information available at that level**:
- `V(obs)` = E[R_total | observation]
- `V(coc)` = E[R_total | observation + CoC]
- `V(traj_j)` = E[R_total | observation + CoC + trajectory up to j]

Note: V(coc) and V(traj_j) predict the **total return** (same target as V(obs)), but with progressively more information. As more of the completion is revealed, the prediction should become more accurate — converging toward the actual return.

### Advantage Formulas

**A_obs** — total return vs. observation-only baseline:
```
A_obs = R_total - V(obs)
```

Measures: "was this entire completion better or worse than expected from just the scene?"

**A_coc** — remaining return after CoC vs. CoC-conditioned baseline:
```
R_remaining_after_coc = w_traj · R_traj + w_consist · R_consistency
A_coc = R_remaining_after_coc - V(coc)
```

Measures: "was the trajectory outcome better or worse than expected after seeing the CoC reasoning?" A positive A_coc means the CoC set up conditions for a better-than-predicted trajectory. A negative A_coc means the CoC was misleading or unhelpful — the trajectory underperformed relative to what the reasoning suggested.

Note: V(coc) here predicts the remaining return (traj + consistency) conditioned on obs + CoC. This differs from the "total return" framing above. An alternative is `A_coc = R_total - V(coc)` where V(coc) predicts total return. The choice depends on whether V(coc) is trained to predict total return or remaining return.

**A_traj_j** — remaining return from step j vs. mid-trajectory baseline:
```
R_remaining_from_j = w_traj · Σ_{t=j}^{T} r_t + w_consist · R_consistency
A_traj_j = R_remaining_from_j - V(traj_j)
A_traj = mean(A_traj_j)     (aggregate for conditioning label)
```

Measures: "from trajectory step j onward, was the remaining trajectory better or worse than expected given everything seen so far?"

### Properties

- **Zero bias:** Uses actual returns, no bootstrapping through potentially inaccurate value predictions. Advantages are exact residuals.
- **Higher variance:** Single-sample MC returns can be noisy, especially for short completions or stochastic reward functions. However, variance is absorbed by the percentile-based binarization (thresholding smooths out noise).
- **Simpler implementation:** No GAE recursion, no discount factor, no λ parameter. Each advantage is a single subtraction.
- **Value head as pure baseline:** The value head's only role is to center the advantage distribution. Its accuracy affects the *efficiency* of binarization (better baselines → cleaner separation of positive/negative) but not the *correctness* of the labels (a constant-offset error in V shifts all advantages equally, leaving binarization unchanged).
- **No per-token credit assignment:** A_traj is computed per-timestep but without the temporal smoothing of GAE. Each step's advantage is independent: `R_remaining - V(traj_j)`. There's no mechanism for a good step at t=5 to propagate credit backward to t=3.
- **Not directly compatible with policy gradients:** These advantages are designed for binary conditioning labels, not for weighting log-probs. Using them as PG weights would require additional variance reduction.

---

## Comparison

| Aspect | Approach A (TD/GAE) | Approach B (Return - Baseline) |
|---|---|---|
| **A_obs** | `G(s_obs) - ema_return` | `R_total - V(obs)` |
| **A_coc** | `w_reason · R + γ · V(coc) - V(obs)` | `R_remaining - V(coc)` |
| **A_traj** | `GAE(γ, λ)` over timesteps | `R_remaining_from_j - V(traj_j)` |
| **Bias** | Nonzero when V is inaccurate (bootstrapping) | Zero (uses actual returns) |
| **Variance** | Lower at λ < 1 (bootstrapping smooths) | Higher (single-sample MC) |
| **Hyperparameters** | γ, λ | None (beyond value head lr) |
| **Credit assignment** | Temporal smoothing via GAE | Independent per step |
| **Value head role** | Bootstrapping target + baseline | Pure baseline |
| **Compatibility** | Works for both PG and conditioning | Designed for conditioning labels |
| **Sensitivity to V accuracy** | High (V appears in both + and - terms) | Lower (V only subtracted as baseline) |
| **At γ=1, λ=1** | Reduces to Approach B for A_traj | — |

### When Approach A is Preferable

- The value head is well-trained (low approximation error), making bootstrapping reliable.
- Combining advantage-conditioned SFT with policy gradient methods (GRPO) that benefit from GAE variance reduction.
- Per-token credit assignment matters (e.g., identifying which specific trajectory timesteps caused failure).
- Discount factor γ < 1 is desired to downweight uncertain far-future predictions.

### When Approach B is Preferable

- The value head is trained between iterations (offline), not online, and may have residual error that would bias bootstrapped estimates.
- Advantages are used exclusively for binary conditioning labels, where percentile binarization absorbs MC variance.
- Simplicity is valued — fewer hyperparameters to tune (no γ, λ).
- The SFT training loop is iterative (rollout → evaluate → train), so value head accuracy improves across iterations but may be unreliable within early iterations.

### Hybrid Considerations

- **A_obs:** Both approaches differ. Approach A uses an EMA baseline (population-level), Approach B uses V(obs) (per-sample value prediction). These answer different questions: "was this scene easier than average?" vs. "was this completion better than the value head predicted for this scene?"
- **A_coc:** The key divergence. TD bootstrapping couples A_coc to V(coc)'s accuracy, while return-minus-baseline treats V(coc) as a centering constant. If V(coc) is well-calibrated, both give similar results. If V(coc) is biased, Approach A amplifies the bias (V appears in both + and - positions with different coefficients), while Approach B only shifts the distribution.
- **A_traj at γ=1, λ=1:** Mathematically identical between the two approaches. The default config uses these values, so the trajectory advantage is the same regardless of which approach is chosen. The distinction only matters if γ or λ are tuned below 1.

---

## Implementation Notes

Both approaches use the same `SegmentValueHead` architecture and the same hidden state extraction pipeline (`h_obs`, `h_coc`, `h_traj` from teacher-forced VLM forward). The only code difference is in `compute_segment_advantages_from_rollouts()` in `advantage_conditioning.py`.

The current implementation uses Approach A. Switching to Approach B requires changing the advantage formulas but not the value head training targets, the binarization logic, or the dataset/trainer infrastructure.
