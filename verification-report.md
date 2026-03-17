# Verification Report: advantage-conditioning.md vs SFT Codebase

## Summary

**Document:** `docs/advantage-conditioning.md`
**Code scope:** SFT training pipeline (`src/alpamayo_r1/training/`)

The design doc was recently simplified from 3 levels (obs/coc/traj) to 2 levels (obs/traj) and the token placement was changed from "all prepended" to "split placement." The code was not updated to match these design doc changes, producing **10 discrepancies** (8 from the CoC removal + placement change, 2 additional).

---

## Discrepancies (10 found)

### 1. Token count: doc says 4, code has 6 (CRITICAL)

**Doc** (Section: Token Design, line 78): "We introduce four new special tokens, two per segment level: `<|adv_obs_pos|>`, `<|adv_obs_neg|>`, `<|adv_traj_pos|>`, `<|adv_traj_neg|>`"

**Code** (`base_model.py:81-88`): `ADV_CONDITIONING_TOKEN_KEYS` contains 6 tokens including `adv_coc_pos` and `adv_coc_neg`. Module docstring (`advantage_conditioning.py:3`) says "6 advantage tokens."

### 2. Token placement: doc says split, code prepends all together (CRITICAL)

**Doc** (Section: Placement in the Sequence, lines 92-115): `adv_obs` before CoC, `adv_traj` after `<|cot_end|>` and before `<|traj_future_start|>`.

**Code** (`advantage_conditioning.py:534-541`): All 3 tokens are prepended as a block before the entire completion:
```python
input_ids = prompt_ids + cond_tokens + completion_ids
```
No parsing of the completion to find the CoC/trajectory boundary.

### 3. A_coc still computed (CRITICAL)

**Doc** (Section: Per-Segment Advantage Definitions): Only defines A_obs and A_traj. States "no CoC-level conditioning token."

**Code** (`advantage_conditioning.py:238-239, 276-277`): Computes `v_coc = value_head(h_coc, level=LEVEL_COC)` and `a_coc = g_coc - v_coc`. Returns `a_coc` in results dict.

### 4. Binarization uses 3 levels, not 2

**Doc** (Section: Binarization, lines 199-201): `I_obs` and `I_traj` only.

**Code** (`advantage_conditioning.py:113-167`): `AdvantageBuffer` maintains `_buf_coc`, `binarize()` returns 3-tuple `(i_obs, i_coc, i_traj)`.

### 5. is_all_positive checks 3 levels, not 2

**Doc** (Section: Dual Loss, line 224): "I_obs=1 ∧ I_traj=1"

**Code** (`advantage_conditioning.py:526`): `is_all_positive = i_obs and i_coc and i_traj`

### 6. build_conditioned_sequence takes i_coc parameter

**Doc**: No CoC conditioning token exists.

**Code** (`advantage_conditioning.py:498`): `i_coc: bool` is a required parameter. Used to select `adv_coc_pos` or `adv_coc_neg`.

### 7. CFG generate uses 3 conditioned tokens, not 2

**Doc** (Section: Inference, line 278): "forward pass with all-positive conditioning tokens" (2 tokens per doc).

**Code** (`cfg_generate.py:69-73`): Inserts `[adv_obs_pos, adv_coc_pos, adv_traj_pos]` — 3 tokens.

### 8. Config has k_coc parameter

**Doc** (Section: Binarization): Only `ε_obs` and `ε_traj` thresholds.

**Code** (`sft_default.yaml:55`): `k_coc: 30` is present in the config.

---

### 9. Binarization uses `>=`, doc says `>` (MINOR)

**Doc** (Section: Binarization, line 199): `I_obs = 1 if A_obs > ε_obs else 0`

**Code** (`advantage_conditioning.py:167`): `return a_obs >= eps_obs, a_coc >= eps_coc, a_traj >= eps_traj`

Strict `>` vs inclusive `>=`. At the percentile boundary this affects which samples are labeled positive.

### 10. Replay label recomputation never called

**Doc** (Section: Mixing Policy for Stability, line 349): "The advantage labels for historical rollouts should be recomputed using the current value head."

**Code**: `RolloutReplayBuffer.recompute_labels()` exists (`selfplay_loop.py:151-185`) but is never called in `run_iteration()`. Historical rollouts use their original (stale) labels.

---

## Internal Document Discrepancy (1 found)

### Value Targets section omits reward weights

**Returns-to-Go section** (line 165): `G(s_obs) = w_reason · R_reasoning + w_traj · Σ_t r_t + w_consist · R_consistency`

**Value Targets section** (line 393): `G(s_obs) = w_reason · R_reasoning + Σ_t r_t` — missing `w_traj` and `w_consist` weights.

The code matches the Returns-to-Go section (correct). The Value Targets section should be updated.

---

## Missing Implementations (2 found)

### 1. KL regularization

**Doc** (Section: Mixing Policy for Stability, line 353): "KL regularization (optional): Add `β_kl · KL(π_θ || π_ref)` to the loss"

**Code**: Not implemented. The config has no `beta_kl` field. This is documented as optional (default 0.0) so it's low priority.


---

## Verified Matches (14 confirmed)

- **A_obs formula**: `g_obs - v_obs` matches `G(s_obs) - V(s_obs)` (`advantage_conditioning.py:274`)
- **A_traj_j formula**: `g_traj - v_traj` via reverse cumsum matches `G(s_traj_j) - V(s_traj_j)` (`advantage_conditioning.py:282-283`)
- **A_traj = mean over timesteps** (`advantage_conditioning.py:284`)
- **G(s_obs) includes all three weighted reward components** (`advantage_conditioning.py:262-269`)
- **G(s_traj_j) via reverse cumulative sum** with consistency as terminal reward (`advantage_conditioning.py:263, 281-282`)
- **V(s_obs) uses level=0, V(s_traj) uses level=2** (`advantage_conditioning.py:238, 241-243`)
- **Percentile threshold k=30 default** (`sft_default.yaml:54,56`)
- **EMA decay α=0.99** (`sft_default.yaml:57`)
- **p_drop=0.3 default** (`sft_default.yaml:52`)
- **alpha=1.0 default** (`sft_default.yaml:53`)
- **G=8 completions per scene** (`sft_default.yaml:51`)
- **Rollout temperature 1.2** (`sft_default.yaml:110`)
- **Replay ratio 0.3** (`sft_default.yaml:58`)
- **Reset-to-checkpoint**: `_train_phase` loads `base_model_path` fresh each iteration (`selfplay_loop.py:557`)
- **Scene partitioning**: `ScenePartitioner` splits clip_ids into disjoint chunks (`selfplay_loop.py:43-84`)
- **Value head pre-training (Stage 0)**: `pretrain_value_head()` runs before loop when `pretrain_scenes > 0` (`selfplay_loop.py:267-361`)
- **Value head trained per iteration before advantages**: `train_segment_value_head()` called in `_evaluate_phase` before `compute_segment_advantages_from_rollouts` (`selfplay_loop.py:475-484`)
- **Expert deferred GPU scheduling**: `training_step` runs SFT backward, frees tensors, then expert CFM step (`sft_trainer.py:172-299`)
- **CFG formula**: `logits_uncond + beta * (logits_cond - logits_uncond)` (`cfg_generate.py:139`)

---

## Root Cause

All 8 discrepancies share the same root cause: the design doc was simplified in two recent commits (`967e513` — remove CoC level, `88c8643` — split token placement) but the code was not updated to match. The code still implements the original three-level design with all tokens prepended together.

## Recommended Fixes

1. **Remove CoC tokens** from `ADV_CONDITIONING_TOKEN_KEYS` in `base_model.py` (keep only obs + traj)
2. **Remove `i_coc` parameter** from `build_conditioned_sequence()` and `AdvCondDataset`
3. **Update `build_conditioned_sequence()`** to split the completion at the CoC/trajectory boundary and insert `adv_traj` between segments
4. **Remove `_buf_coc`** from `AdvantageBuffer` and update `binarize()` to return 2-tuple
5. **Remove `a_coc` computation** from `compute_segment_advantages_from_rollouts()`
6. **Update `cfg_generate()`** to use 2 conditioned tokens
7. **Remove `k_coc`** from `sft_default.yaml`
8. **Fix Value Targets section** in design doc to include reward weights
9. **Fix binarization operator** — change `>=` to `>` in `AdvantageBuffer.binarize()` (or update doc to `>=`)
10. **Call `recompute_labels()`** in `_evaluate_phase()` for historical replay buffer entries

## TODO: Advantage-Conditioned Action Expert

The design doc describes how conditioning tokens enter the action expert's context, but this bridge between VLM-only CFG and expert inference has not been built. The codebase currently has two disconnected pieces:

- **VLM-only CFG** (`cfg_generate.py`): Advantage-conditioned generation for autoregressive VLM output (text + discrete trajectory tokens). No expert involvement.
- **Expert inference** (`alpamayo_r1.py`): VLM→expert KV cache handoff for flow matching diffusion. No advantage conditioning.

### Training TODO

- [ ] **Advantage-conditioned expert training**: During `_expert_cfm_step()` in `sft_trainer.py`, the teacher-forced VLM forward (`_get_vlm_kv_cache_teacher_forced`) produces a KV cache from `[prompt + completion_prefix]`. This cache should include conditioning tokens at their causal positions:
  - `adv_obs` should be part of the prompt (before CoC tokens in `completion_prefix`)
  - `adv_traj` should be appended after `completion_prefix` via one extra VLM forward step before calling `compute_cfm_loss()`
  - This way the expert learns to attend to conditioning tokens during training, so it can respond to them at inference

### Inference TODO

- [ ] **Advantage-conditioned expert inference**: Create a new inference function (e.g., `cfg_generate_with_expert()`) that combines VLM generation with expert diffusion under CFG:
  1. VLM generates CoC text with `adv_obs_pos` in the prompt → KV cache includes `adv_obs`
  2. Inject `adv_traj_pos` via one extra VLM forward step (single token, KV cache reuse)
  3. Expert receives conditioned KV cache and runs diffusion → conditioned trajectory
  4. Repeat steps 1-3 without conditioning tokens → unconditional trajectory
  5. CFG interpolation on the expert's output (action space or trajectory space)

- [ ] **CFG in action space vs. logit space**: For the expert, CFG cannot operate on logits (the expert outputs continuous velocity fields, not token distributions). Options:
  - Interpolate in action space: `action_final = action_uncond + β · (action_cond - action_uncond)`
  - Interpolate velocity fields at each diffusion step: `v_final = v_uncond + β · (v_cond - v_uncond)`
  - The velocity field interpolation is more principled (applied per-step rather than post-hoc)

- [ ] **Unconditional expert pass**: For the CFG baseline, run expert with the original KV cache (no advantage tokens injected). This matches the doc's description: "the expert sees the original KV cache ending at `<|traj_future_start|>`."
