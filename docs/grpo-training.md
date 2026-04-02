# GRPO Training in Alpamayo-R1

## Overview

GRPO training is the RL post-training stage for Alpamayo-R1. The goal is to improve the **VLM backbone** so that, given a driving scene, it generates:

1. **Chain-of-Causation (CoC) reasoning text**, and then
2. **Discrete future trajectory tokens**

in a single autoregressive sequence.

In this repository, GRPO is implemented as **VLM-only rollout training**:

- the **VLM** is the policy being optimized,
- the **expert** and **diffusion** modules are kept out of the rollout path,
- the model directly emits trajectory tokens,
- those tokens are decoded into continuous waypoints only for reward computation.

The training entry point is `src/alpamayo_r1/training/train_grpo.py`, and the custom trainer is `src/alpamayo_r1/training/rollout.py`.

---

## High-level training loop

At a high level, one GRPO step looks like this:

1. Sample a batch of driving scenes.
2. For each scene, generate **G** completions (`training.num_generations`, default 8).
3. Score each completion with reward functions.
4. Compute **group-relative advantages** within each scene’s set of G samples.
5. Update the VLM so higher-reward reasoning/trajectory sequences become more likely.

That means Alpamayo is not trained against a supervised target sequence here. Instead, it explores multiple sampled completions per scene and learns from their **relative reward ranking**.

---

## End-to-end pipeline

```text
PhysicalAI-AV clips
    ↓
build_alpamayo_dataset()
    ↓
Prompt contains metadata: [clip_id=...] [t0_us=...]
    ↓
AlpamayoGRPOTrainer rollout
    ↓
Load scene data for that clip/timestamp
    - camera images
    - ego history trajectory
    - ground-truth future trajectory
    ↓
Fuse history trajectory tokens into the prompt
    ↓
VLM generates:
    CoC text + future trajectory tokens
    ↓
Decode trajectory tokens → continuous xyz waypoints
    ↓
Compute rewards
    - trajectory quality
    - reasoning quality
    - consistency
    ↓
TRL GRPOTrainer computes grouped advantages and GRPO loss
    ↓
Backprop through the VLM (typically LoRA adapters only)
```

---

## 1. Dataset construction

File: `src/alpamayo_r1/training/dataset.py`

`build_alpamayo_dataset()` creates a lightweight Hugging Face dataset. Each example stores:

- `prompt`: a chat-style prompt,
- `clip_id`: the driving clip identifier,
- `t0_us`: the timestamp used as the prediction origin.

The prompt does **not** inline all image and trajectory tensors. Instead, the prompt embeds metadata like:

```text
[clip_id=...] [t0_us=...]
```

During rollout, the trainer parses those fields and loads the real scene data through `PhysicalAIAVDatasetInterface`.

This keeps the dataset compact and lets rollouts lazily fetch the multimodal inputs they need.

---

## 2. Model setup before training

File: `src/alpamayo_r1/training/train_grpo.py`

`train_grpo.py` loads the full `AlpamayoR1` model, but only the **VLM portion** is optimized during GRPO.

### What gets frozen

All parameters outside `vlm.*` are frozen by `_freeze_non_vlm_params()`.

So GRPO does **not** update:

- expert module,
- diffusion module,
- action projection layers,
- other non-VLM components.

### What usually gets trained

By default, LoRA is applied to the VLM attention projections:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

Default config in `src/alpamayo_r1/training/configs/grpo_default.yaml`:

```yaml
lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
```

So the common GRPO setup is: **freeze the full model except the VLM, then train small LoRA adapters on the VLM**.

---

## 3. What a rollout actually does

File: `src/alpamayo_r1/training/rollout.py`

The core logic lives in `AlpamayoGRPOTrainer`, which subclasses TRL’s `GRPOTrainer`.

### Prompt expansion into groups

GRPO needs multiple sampled completions per prompt. If `num_generations=8`, then each scene is sampled 8 times with stochastic decoding.

Those 8 completions form one **group** for relative comparison.

### Scene loading

Inside `_generate_single_turn()`, the trainer:

1. parses `clip_id` and `t0_us` from the prompt,
2. loads the scene using `ClipDataCache`,
3. obtains:
   - camera images,
   - ego history xyz/rotation,
   - ground-truth future xyz.

`ClipDataCache` avoids repeatedly reloading the same clip from disk/network.

### History fusion

Before generation, the trainer injects the ego history into the token stream using:

- `full_model.fuse_traj_tokens(...)`

This replaces trajectory placeholder tokens with the actual tokenized history trajectory so the VLM conditions on recent motion.

### Generation

The VLM then generates a single sequence containing:

- CoC reasoning text first,
- future trajectory tokens after that,
- termination at `<|traj_future_end|>`.

By default, generation uses:

```yaml
rollout:
  temperature: 0.6
  top_p: 0.98
  max_generation_length: 256
```

The trainer uses `StopAfterEOS` so generation ends once the future trajectory terminator is reached.

---

## 4. From generated tokens to trajectories

After generation, the trainer extracts future trajectory tokens from the completion and decodes them.

Important pieces:

- token extraction: `extract_traj_tokens()`
- decoding: `traj_tokenizer.decode()`

This converts the generated discrete trajectory token IDs into continuous future waypoints in xyz space.

So the policy itself predicts **tokens**, but rewards are computed on decoded **continuous trajectories**.

---

## 5. Why log-probabilities are computed afterward

GRPO needs token-level log-probabilities for the sampled completions. After generation, the trainer runs a separate teacher-forced forward pass over the generated completion to compute those log-probs.

That happens in `_generate_single_turn()` via `_compute_batch_logprobs(...)`.

Conceptually:

- generation produces sampled completions,
- a second forward pass scores how probable those tokens were under the current policy,
- TRL uses those log-probs when building the GRPO objective.

This is separate from decoding trajectories for reward computation.

---

## 6. Reward functions

File: `src/alpamayo_r1/training/rewards.py`

Each sampled completion gets multiple rewards.

### Trajectory quality reward

`trajectory_quality_reward()` compares predicted and ground-truth future trajectories using ADE on the xy plane.

Approximate form:

```text
reward = max(0, 1 - ADE / threshold)
```

Default threshold is 5 meters.

This is the most grounded reward because it directly measures trajectory accuracy.

### Reasoning quality reward

`reasoning_quality_reward()` is a rule-based text reward. It scores the CoC text based on things like:

- presence of causal connectors,
- driving-domain vocabulary,
- non-empty and reasonable length,
- lack of degenerate repetition.

This is a heuristic stand-in for a more sophisticated reasoning critic.

### Consistency reward

`consistency_reward()` checks whether the generated reasoning text agrees with the generated trajectory at a coarse behavior level, such as:

- turning,
- going straight,
- accelerating,
- decelerating.

It maps the predicted trajectory to meta-actions and checks whether the text mentions matching behaviors.

### Default reward weights

Current defaults are:

```yaml
rewards:
  trajectory_weight: 0.50
  reasoning_weight: 0.50
  consistency_weight: 0.0
```

So consistency is currently disabled by default, while trajectory and reasoning each contribute half of the total reward.

---

## 7. How GRPO uses the rewards

For each scene, the trainer samples a group of completions:

```text
scene i → completion 1, completion 2, ..., completion G
```

Each completion gets a scalar total reward from the weighted reward functions.

GRPO then compares samples **within the same group**. A completion is considered good if it scores better than the other sampled completions from the same scene.

This gives a **group-relative advantage** instead of an absolute target. The policy update then increases the probability of better-ranked completions and decreases the probability of worse-ranked ones.

In practical terms:

- if one reasoning/trajectory sample is better than the others for the same scene, it gets positive advantage,
- if it is worse, it gets negative advantage,
- this helps stabilize RL by comparing like with like.

The underlying loss and optimization are handled by TRL’s `GRPOTrainer`, while `AlpamayoGRPOTrainer` customizes how multimodal rollouts and rewards are produced.

---

## 8. Comparison with the SFT pipeline

The repository also has an **advantage-conditioned iterative SFT** pipeline, implemented in:

- `src/alpamayo_r1/training/train_sft.py`
- `src/alpamayo_r1/training/selfplay_loop.py`
- `src/alpamayo_r1/training/sft_rollout.py`
- `src/alpamayo_r1/training/sft_trainer.py`
- `src/alpamayo_r1/training/advantage_conditioning.py`

Both pipelines generate rollouts and score them with rewards, but they use those scores very differently.

### GRPO vs SFT at a glance

| Aspect | GRPO | Advantage-conditioned SFT |
|---|---|---|
| Outer algorithm | RL / policy optimization | Iterative self-play + supervised learning |
| Trainer | TRL `GRPOTrainer` subclass | HF `Trainer` subclass |
| Rollout usage | Used immediately to form policy-gradient-style updates | Converted into labeled SFT examples |
| Main training loss | GRPO objective on sampled completions | Teacher-forced cross-entropy |
| What reward affects | Sample weights/advantages in the loss | Binary conditioning labels attached to sequences |
| Default rollout mode | `vlm_only` | often `expert` in current config |
| Expert/diffusion in rollout | not used by default | can be used during rollout and expert finetuning |

### The most important difference: how the advantage is computed and used

#### In GRPO

GRPO samples **G completions for the same scene** and compares them **against each other**.

Conceptually:

```text
A_i,j ≈ reward_i,j - baseline_from_other_samples_in_same_group
```

where:

- `i` = scene,
- `j` = one sampled completion among the `G` completions for that scene.

So the advantage is **group-relative**. A completion gets positive advantage if it is better than the other sampled completions for that same prompt/scene, and negative advantage if it is worse.

This is the standard GRPO idea:

- rewards are computed per completion,
- advantages are formed within each prompt group,
- those advantages directly scale the policy update.

In other words, GRPO uses the advantage as an **optimization weight**.

#### In advantage-conditioned SFT

The SFT pipeline does **not** use grouped relative ranking inside the training loss.

Instead, it computes a **return-minus-value baseline** advantage for each completion:

```text
a = G(s_obs) - V(s_obs)
```

In the current implementation in `advantage_conditioning.py`:

- reward components are first normalized,
- a scalar return target `G(s_obs)` is formed from the weighted rewards,
- a value head predicts `V(s_obs)`,
- the advantage is computed as `a_obs = G(s_obs) - V(s_obs)`,
- and `a_traj` is currently set equal to that same value in the default computation.

So unlike GRPO, the SFT pipeline's advantage is **not “how much better than the other G samples was this completion?”**
It is closer to **“how much better or worse than the value-head expectation was this completion?”**

Then that continuous advantage is **binarized** using percentile thresholds:

```text
i_obs  = 1[a_obs  > eps_obs]
i_traj = 1[a_traj > eps_traj]
```

Those binary labels are turned into conditioning tokens such as:

- positive observation label,
- negative observation label,
- positive trajectory label,
- negative trajectory label.

Then the model is trained by plain SFT to predict the same completion sequence under those labels.

So in SFT, the advantage is used as a **data label**, not as a direct policy-gradient weight.

### Why this matters

This difference changes the role of rollout sampling:

- **GRPO** needs multiple samples per scene mainly to estimate a stable **relative advantage** and improve the policy online.
- **SFT** needs rollouts mainly to create a better supervised dataset: generate completions, score them, label them as positive/negative, and train on them with cross-entropy.

Said another way:

- GRPO asks: **which sampled completion should get pushed up or down right now?**
- SFT asks: **what kind of completion should the model learn to imitate when conditioned on good/bad labels?**

### Training objective difference

#### GRPO objective

The GRPO path uses sampled completions, token log-probs, and advantages inside the RL loss.

Very roughly:

```text
loss ~ - advantage × logprob(sampled completion)
```

with GRPO-specific clipping / normalization handled by TRL.

#### SFT objective

The SFT path uses the same rollout completions as fixed targets and optimizes:

```text
loss = cross_entropy(predicted_tokens, completion_tokens)
```

The only role of the advantage is to decide which conditioning tokens get inserted before CoC and trajectory generation.

### Rollout path difference

Another important practical difference:

- **GRPO** in this repo is intentionally **VLM-only** during rollout.
- **SFT** can run in either:
  - `vlm_only` mode, or
  - `expert` mode, where the VLM generates CoC and the expert+diffusion path generates the trajectory.

In the current `sft_default.yaml`, rollout mode is:

```yaml
rollout:
  mode: expert
```

So by default the SFT pipeline is closer to the full system than GRPO is.

### Value head role

- In **GRPO**, the default pipeline does not rely on a learned value head for the main advantage computation; the core signal is the group-relative GRPO advantage from sampled rewards.
- In **SFT**, the value head is central because the advantage labels are computed from **return minus value prediction**.

That is why the SFT pipeline has explicit stages for:

- pretraining the value head,
- updating it on rollout data,
- then computing advantage labels with the current value head.

### Practical takeaway

If you want the shortest comparison:

- **GRPO**: reward → group-relative advantage → RL loss.
- **SFT**: reward → return-minus-value advantage → positive/negative labels → supervised loss.

So the biggest conceptual difference is that **GRPO uses advantage as an online optimization signal**, while **the SFT pipeline uses advantage as a label-generation mechanism for later supervised training**.

## 9. Why this is “VLM-only GRPO”

Alpamayo-R1 as a full system includes more than the VLM, but GRPO here intentionally trains only the VLM rollout path.

That means:

- the VLM directly emits trajectory tokens,
- expert and diffusion are not used to refine trajectories during rollout,
- reward is computed directly from the decoded VLM output.

This has two main advantages:

1. **Lower memory / simpler training loop**
2. **Directly aligns the VLM outputs with the RL reward**

It also means the RL signal is attached to exactly what the model generated: the reasoning text and the future trajectory token sequence.

---

## 9. Trainer customizations in this repo

`AlpamayoGRPOTrainer` adds several Alpamayo-specific behaviors on top of TRL:

- custom `_generate_single_turn()` for multimodal VLM rollouts,
- custom `_calculate_rewards()` to decode trajectory tokens before reward scoring,
- `log()` workaround so evaluation reward metrics are visible to early stopping,
- custom `_save()` for PEFT/Qwen3-VL compatibility,
- optional vLLM generation backend,
- optional value head and expert fine-tuning hooks.

For the default GRPO path, the most important customizations are the rollout override and the reward decoding logic.

---

## 10. vLLM support

The trainer supports two generation backends:

1. **Hugging Face generation** via `model.generate()`
2. **vLLM** for faster rollout generation

vLLM can run in:

- `colocate` mode: in-process,
- `server` mode: separate vLLM process.

When vLLM is enabled, generation is delegated through a custom `rollout_func`, but the overall logic is the same:

- load scene,
- fuse history,
- generate CoC + trajectory tokens,
- decode trajectory tokens,
- compute rewards,
- run GRPO updates.

---

## 11. Configuration knobs that matter most

File: `src/alpamayo_r1/training/configs/grpo_default.yaml`

The most important GRPO controls are:

### Group size

```yaml
training.num_generations: 8
```

This is the number of sampled completions per scene.

### Policy optimization

```yaml
training.learning_rate: 1e-5
training.num_train_epochs: 1
training.gradient_checkpointing: true
training.beta: 0.0
```

`beta: 0.0` means no KL penalty/reference-model term is used by default.

### Rollout sampling

```yaml
rollout.temperature: 0.6
rollout.top_p: 0.98
```

These control exploration. Higher temperature increases diversity but also noise.

### Reward mix

```yaml
rewards:
  trajectory_weight: 0.50
  reasoning_weight: 0.50
  consistency_weight: 0.0
```

### Early stopping

```yaml
early_stopping:
  enabled: true
  metric: reward
```

By default, early stopping tracks the overall weighted reward on a held-out eval set.

---

## 12. What is optimized in one sentence

GRPO in Alpamayo-R1 trains the VLM so that, for each driving scene, sampled CoC-plus-trajectory completions with **better reward than their peers** become more likely under the model.

---

## 13. Practical summary

If you want the shortest mental model:

- build prompts that point to a driving clip,
- load the real multimodal scene at rollout time,
- fuse trajectory history into the prompt,
- sample multiple CoC + trajectory completions from the VLM,
- decode trajectories,
- score each sample with trajectory and reasoning rewards,
- use GRPO to push probability mass toward the better samples,
- update mainly LoRA adapters on the VLM.

That is how GRPO training works in this repository.

---

## Key files

- `docs/grpo-training.md`
- `src/alpamayo_r1/training/train_grpo.py`
- `src/alpamayo_r1/training/rollout.py`
- `src/alpamayo_r1/training/rewards.py`
- `src/alpamayo_r1/training/dataset.py`
- `src/alpamayo_r1/training/configs/grpo_default.yaml`
