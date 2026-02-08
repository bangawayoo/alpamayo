# GRPO Post-Training for Alpamayo-R1

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module Reference](#module-reference)
   - [train_grpo.py](#train_grpopy)
   - [rollout.py](#rolloutpy)
   - [rewards.py](#rewardspy)
   - [dataset.py](#datasetpy)
4. [Reward Functions](#reward-functions)
   - [Trajectory Quality Reward](#trajectory-quality-reward)
   - [Reasoning Quality Reward](#reasoning-quality-reward)
   - [Consistency Reward](#consistency-reward)
5. [Configuration Reference](#configuration-reference)
6. [Quick Start](#quick-start)
   - [Running a Smoke Test](#running-a-smoke-test)
   - [Full Training Run](#full-training-run)
   - [Custom Overrides](#custom-overrides)
7. [Design Decisions](#design-decisions)
   - [Why LoRA on VLM Only](#why-lora-on-vlm-only)
   - [Why beta=0.0 (No KL Penalty)](#why-beta00-no-kl-penalty)
   - [Why Override _generate_single_turn](#why-override-_generate_single_turn)
   - [Input IDs Clone Pattern](#input-ids-clone-pattern)
   - [Gradient Checkpointing Toggle](#gradient-checkpointing-toggle)
8. [Testing](#testing)
   - [CPU Tests](#cpu-tests)
   - [GPU Smoke Test](#gpu-smoke-test)
9. [Data Flow Diagram](#data-flow-diagram)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The GRPO (Group Relative Policy Optimization) post-training module implements Stage 3 training for Alpamayo-R1, as described in the [Alpamayo-R1 paper](https://arxiv.org/abs/2511.00088). While the paper describes RL post-training, the current release focuses on the training infrastructure and reward functions that enable reinforcement learning-based optimization.

### What is GRPO?

GRPO is a policy optimization algorithm designed for training language models with non-differentiable reward functions. It belongs to the family of on-policy RL algorithms similar to PPO (Proximal Policy Optimization), but uses grouped advantages for more stable training.

**Key properties**:
- **On-policy**: Uses samples from the current policy for training (not from a replay buffer)
- **Group-based advantages**: Computes advantages relative to a group of samples (controlled by `num_generations` parameter)
- **Reward-driven**: Optimizes the policy to maximize expected reward, where rewards can be any Python functions

### Why GRPO for Alpamayo-R1?

Alpamayo-R1 generates chain-of-causation (CoC) reasoning text, and the quality of this reasoning affects downstream trajectory prediction. Traditional supervised learning cannot optimize for:
1. **Trajectory quality** — How well the predicted trajectories match ground truth
2. **Reasoning quality** — How coherent and driving-relevant the CoC text is
3. **Consistency** — Whether the CoC text aligns with the predicted trajectory behavior

GRPO allows us to train the VLM's text generation to maximize these objectives directly, even though they are non-differentiable.

### What This Module Does

The GRPO training module:
1. **Loads** the full AlpamayoR1 model (VLM + Expert Transformer + Flow Matching Diffusion)
2. **Freezes** all non-VLM parameters (expert, diffusion, action space, projections)
3. **Applies LoRA** to the VLM's attention layers for parameter-efficient fine-tuning
4. **Runs GRPO training** with custom rollout and reward functions:
   - **Rollout**: VLM generates CoC text → Expert + Diffusion predict trajectories
   - **Rewards**: Compute 3 reward signals from (CoC text, predicted trajectories, ground truth)
   - **GRPO loss**: Update VLM parameters to maximize expected reward

---

## Architecture

### Alpamayo-R1 Model Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                     AlpamayoR1 Full Model                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Qwen3-VL-2B (Vision-Language Model) - "vlm"            │  │
│  │  • Processes multi-camera images + egomotion history     │  │
│  │  • Generates CoC (chain-of-causation) reasoning text     │  │
│  │  • LoRA applied to q/k/v/o_proj in attention layers      │  │
│  │  • Trainable in GRPO (frozen base, LoRA adapters learn)  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              │ KV cache                         │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Expert Transformer                                       │  │
│  │  • Uses VLM's KV cache as context                        │  │
│  │  • Processes noisy action tokens from diffusion          │  │
│  │  • Frozen during GRPO training                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Flow Matching Diffusion                                  │  │
│  │  • Samples trajectory tokens from learned distribution    │  │
│  │  • Iterative denoising with Expert as step function      │  │
│  │  • Frozen during GRPO training                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Action Space Converter                                   │  │
│  │  • Converts action tokens to (xyz, rotation) trajectories │  │
│  │  • Frozen during GRPO training                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
         │ CoC text                           │ pred_xyz, pred_rot
         ▼                                    ▼
    ┌─────────────────────────────────────────────────────┐
    │           Reward Functions (3 signals)              │
    ├─────────────────────────────────────────────────────┤
    │  1. trajectory_quality_reward(pred_xyz, gt_xyz)     │
    │  2. reasoning_quality_reward(coc_text)              │
    │  3. consistency_reward(coc_text, pred_xyz)          │
    └─────────────────────────────────────────────────────┘
                          │
                          │ weighted sum
                          ▼
                    GRPO Loss → ∇ VLM (LoRA)
```

### Training Data Flow

```
Dataset (PhysicalAI-AV clips)
    │
    │ Each sample: {prompt: [system, user], clip_id, t0_us}
    │
    ▼
AlpamayoGRPOTrainer._generate_single_turn
    │
    ├─ 1. Parse clip_id and t0_us from prompt
    │
    ├─ 2. Load driving data (images, ego history/future)
    │      └─> load_physical_aiavdataset(clip_id, t0_us)
    │
    ├─ 3. Prepare model inputs
    │      └─> helper.prepare_model_inputs(data, processor)
    │
    ├─ 4. Run full pipeline (num_generations times)
    │      └─> full_model.sample_trajectories_from_data_with_vlm_rollout()
    │          ├─ VLM generates CoC text (autoregressive, with sampling)
    │          ├─ Expert + Diffusion predict trajectories (conditioned on VLM KV cache)
    │          └─ Return: (pred_xyz, pred_rot, extra={'cot': coc_texts})
    │
    ├─ 5. Compute per-token log-probs (teacher-forced VLM forward)
    │      └─> _compute_batch_logprobs(full_model, model_inputs, completions)
    │
    └─> Return: (prompt_ids, completion_ids, logprobs, extra_fields)
            extra_fields = {pred_xyz, gt_xyz, completions}
                │
                ▼
        TRL GRPOTrainer
            │
            ├─ 6. Compute rewards
            │      trajectory_quality_reward(completions, pred_xyz, gt_xyz)
            │      reasoning_quality_reward(completions)
            │      consistency_reward(completions, pred_xyz)
            │      → [r1, r2, r3] per sample
            │
            ├─ 7. Weighted sum: r = w1*r1 + w2*r2 + w3*r3
            │
            ├─ 8. Compute GRPO loss with grouped advantages
            │      loss = -log_prob * (reward - baseline)
            │
            └─ 9. Backprop through VLM (LoRA adapters only)
```

### AlpamayoGRPOTrainer Subclass Design

TRL's `GRPOTrainer` provides `rollout_func` as a customization point, but this is only invoked in vLLM code paths (for high-throughput inference). For our use case:
- We need the full Alpamayo pipeline (VLM + Expert + Diffusion), not just VLM text generation
- We need trajectory predictions for reward computation, not just text
- vLLM integration is not required for this stage

**Solution**: Override `_generate_single_turn` directly, which is the internal method TRL calls during training. This gives us full control over generation while keeping TRL's training loop, loss computation, logging, and checkpointing.

---

## Module Reference

### train_grpo.py

**Location**: `src/alpamayo_r1/training/train_grpo.py`

**Purpose**: Entry point for GRPO training. Orchestrates model loading, LoRA setup, dataset construction, and trainer initialization.

**Key Functions**:

#### `_freeze_non_vlm_params(model: AlpamayoR1) -> None`

Freezes all parameters that don't belong to the VLM backbone. Only VLM parameters will receive LoRA adapters and be trained.

**Parameters**:
- `model`: The full AlpamayoR1 model

**Behavior**:
- Iterates through all named parameters
- Sets `requires_grad = False` for any parameter not starting with `"vlm."`
- Logs the count of frozen parameter groups

**What gets frozen**:
- Expert transformer (`expert.*`)
- Diffusion model (`diffusion.*`)
- Action space converters (`action_in_proj.*`, `action_out_proj.*`, `action_space.*`)
- Any other non-VLM modules

#### `main(cfg: DictConfig) -> None`

Main training function invoked by Hydra.

**Workflow**:
1. **Set seeds** for reproducibility
2. **Load model**: `AlpamayoR1.from_pretrained(model_name, dtype=torch.bfloat16)`
3. **Freeze non-VLM params**: `_freeze_non_vlm_params(full_model)`
4. **Initialize processor**: `helper.get_processor(full_model.tokenizer)`
5. **Create LoRA config**: Target modules = `[q_proj, k_proj, v_proj, o_proj]`
6. **Build GRPO config**: `GRPOConfig` with training hyperparameters
7. **Build dataset**: `build_alpamayo_dataset()` from PhysicalAI-AV
8. **Initialize trainer**: `AlpamayoGRPOTrainer` with custom rollout
9. **Train**: `trainer.train()`
10. **Save model**: Final checkpoint + processor to `output_dir/final`

**Hydra Configuration**:
```bash
python -m alpamayo_r1.training.train_grpo --config-name grpo_default
```

All config values can be overridden via CLI:
```bash
python -m alpamayo_r1.training.train_grpo training.num_train_epochs=1 data.max_samples=10
```

---

### rollout.py

**Location**: `src/alpamayo_r1/training/rollout.py`

**Purpose**: Custom `GRPOTrainer` subclass that uses the full Alpamayo-R1 pipeline for generation instead of standard HF `model.generate()`.

#### `class AlpamayoGRPOTrainer(GRPOTrainer)`

**Constructor Parameters**:
- `full_model: AlpamayoR1` — The complete model (VLM + Expert + Diffusion)
- `avdi: PhysicalAIAVDatasetInterface` — Dataset interface for loading driving data
- `rollout_temperature: float = 0.6` — Sampling temperature for VLM generation
- `rollout_top_p: float = 0.98` — Nucleus sampling threshold
- `rollout_max_generation_length: int = 256` — Maximum CoC tokens to generate
- `**kwargs` — All other arguments passed to parent `GRPOTrainer`

**Important**: The `model` argument passed to the parent `GRPOTrainer` should be `full_model.vlm` (just the VLM component), because TRL applies LoRA to the `model` parameter. The `full_model` is stored separately for rollout.

#### `_generate_single_turn(self, prompts: list) -> tuple`

Overrides TRL's generation to use the full Alpamayo pipeline.

**Input**:
- `prompts`: List of prompt strings (TRL repeats each unique prompt `num_generations` times)

**Output**: Tuple of `(prompt_ids, completion_ids, logprobs, extra_fields)`
- `prompt_ids`: List of prompt token ID lists
- `completion_ids`: List of completion token ID lists
- `logprobs`: List of per-token log-probability lists
- `extra_fields`: Dict with `{"pred_xyz": [...], "gt_xyz": [...]}` for reward functions

**Workflow**:

1. **De-duplicate prompts**:
   ```python
   unique_prompts = prompts[::num_generations]
   ```
   TRL passes `[prompt1, prompt1, ..., prompt1, prompt2, prompt2, ...]` (each repeated `G` times). We process each unique prompt once and generate `G` samples.

2. **Disable gradient checkpointing temporarily**:
   The Alpamayo pipeline needs `use_cache=True` for the VLM's KV cache (used by Expert), which conflicts with gradient checkpointing.

3. **For each unique prompt**:
   - **Parse metadata**: Extract `clip_id` and `t0_us` from prompt using regex
   - **Load driving data**: `load_physical_aiavdataset(clip_id, t0_us, avdi)`
   - **Prepare inputs**: `helper.prepare_model_inputs(data, processor, device)`
   - **Clone input_ids**: Save before model pops them
     ```python
     prompt_input_ids = model_inputs["tokenized_data"]["input_ids"].clone()
     ```
   - **Run pipeline**: Generate `num_generations` samples
     ```python
     pred_xyz, pred_rot, extra = full_model.sample_trajectories_from_data_with_vlm_rollout(
         data=model_inputs,
         top_p=rollout_top_p,
         temperature=rollout_temperature,
         num_traj_samples=num_generations,
         max_generation_length=rollout_max_generation_length,
         return_extra=True,
     )
     ```
   - **Extract CoC texts**: From `extra["cot"]`, flatten to list
   - **Tokenize completions**: `processor.tokenizer.encode(coc_text, add_special_tokens=False)`
   - **Compute log-probs**: Teacher-forced VLM forward pass

4. **Re-enable gradient checkpointing** if it was active

5. **Return** all data in TRL's expected format

#### `_parse_clip_metadata(prompt_text: str) -> tuple[str, int]`

Helper function to extract clip metadata from the prompt.

**Pattern**: `[clip_id=...] [t0_us=...]` embedded in the system message by `_build_prompt_text()`

**Returns**: `(clip_id, t0_us)` tuple

**Raises**: `ValueError` if metadata cannot be parsed

#### `_compute_batch_logprobs(full_model, model_inputs, prompt_input_ids, completion_ids_list, prompt_len, device) -> list[list[float]]`

Computes per-token log-probabilities for a batch of completions via teacher-forced VLM forward pass.

**Why needed**: GRPO requires log-probs of the generated tokens under the current policy. We can't use the logits from sampling (they're from a different temperature), so we re-run the VLM in teacher-forcing mode with the sampled tokens.

**Workflow**:
1. Concatenate prompt + completion tokens: `full_ids = [prompt_tokens | completion_tokens]`
2. Build attention mask and other inputs (pixel_values, image_grid_thw)
3. Forward pass: `outputs = full_model.vlm(input_ids=full_ids, **forward_kwargs)`
4. Extract logits for completion tokens: `logits[prompt_len-1 : prompt_len-1 + comp_len]`
5. Compute log-softmax and gather token log-probs: `log_probs.gather(1, tokens)`
6. Return per-token log-probs as Python lists

**Note**: The indexing `logits[prompt_len-1 : ...]` is correct because logits at position `t` predict token at position `t+1`.

#### `_collate_rollout_outputs(...) -> dict`

Pads and collates rollout outputs into a single batch dict (not used by `AlpamayoGRPOTrainer` but provided for completeness).

**Padding behavior**:
- **Prompts**: Left-padded to max prompt length
- **Completions**: Right-padded to max completion length
- **Logprobs**: Right-padded with zeros

---

### rewards.py

**Location**: `src/alpamayo_r1/training/rewards.py`

**Purpose**: Implements three reward functions that score the quality of (CoC text, predicted trajectories) pairs.

All reward functions follow the TRL interface:
```python
def reward_func(completions: list[str], **kwargs) -> list[float]:
    """
    Args:
        completions: Generated CoC text strings
        **kwargs: Extra fields forwarded from rollout (pred_xyz, gt_xyz, etc.)

    Returns:
        List of rewards in [0, 1], one per completion
    """
```

#### `trajectory_quality_reward(completions, pred_xyz, gt_xyz, **kwargs) -> list[float]`

**Purpose**: Reward based on trajectory prediction accuracy (minADE metric).

**Inputs**:
- `pred_xyz`: Per-sample predicted trajectories as flattened lists
  - Original shape per sample: `(num_traj_samples, T=64, 3)`
- `gt_xyz`: Per-sample ground-truth trajectories as flattened lists
  - Original shape per sample: `(T=64, 3)`

**Algorithm**:
1. Reshape flattened arrays to proper tensor shapes
2. Compute minADE:
   ```python
   pred_xy = pred[:, :, :2]  # (S, T, 2)
   gt_xy = gt[:, :2]         # (T, 2)
   diff = norm(pred_xy - gt_xy)  # (S, T)
   ade_per_sample = mean(diff, axis=T)  # (S,)
   min_ade = min(ade_per_sample)
   ```
3. Soft threshold reward:
   ```python
   reward = max(0.0, 1.0 - min_ade / ade_threshold)
   ```
   Default `ade_threshold = 5.0` meters

**Intuition**: Trajectories with lower displacement error get higher reward. A perfect prediction (minADE=0) gets reward=1.0, while predictions >5m off get reward=0.0. The threshold creates a smooth gradient for GRPO to optimize.

**Why minADE**: We generate multiple trajectory samples (`num_traj_samples=8`), and we want to reward the model if *any* sample is good (best-of-N behavior). Using min over samples encourages diversity.

---

#### `reasoning_quality_reward(completions, **kwargs) -> list[float]`

**Purpose**: Rule-based reward for chain-of-causation reasoning quality.

**Criteria** (4 total, each worth 0.25 points):

1. **Causal connectors** (0, 0.5, or 1.0 points):
   - Patterns: `because`, `therefore`, `since`, `thus`, `hence`, `as a result`, `so that`, `due to`, `consequently`, `leads to`, `causing`, `results in`
   - Score: 1.0 if ≥2 connectors, 0.5 if 1, else 0.0

2. **Driving-domain vocabulary** (0, 0.5, or 1.0 points):
   - Patterns: `vehicle`, `car`, `truck`, `pedestrian`, `cyclist`, `lane`, `intersection`, `traffic`, `signal`, `stop`, `yield`, `merge`, `speed`, `brake`, `accelerate`, `steer`, `turn`, `obstacle`, etc.
   - Score: 1.0 if ≥4 terms, 0.5 if 2-3, else 0.0

3. **Appropriate length** (0, 0.25, or 1.0 points):
   - Min length: 40 characters
   - Max length: 2000 characters
   - Score: 1.0 if in range, 0.25 if non-empty, else 0.0

4. **No degenerate repetition** (0 or 1.0 points):
   - Pattern: Any substring ≥20 chars repeated ≥3 times consecutively
   - Score: 1.0 if no repetition, else 0.0

**Total reward**: Sum of criteria / 4.0 → range [0, 1]

**Example**:
```
Text: "The ego vehicle is approaching an intersection. Because there is a
       pedestrian crossing ahead, the vehicle should decelerate. Therefore,
       the vehicle will slow down and maintain its lane to ensure safety."

Score:
  - Causal connectors: 2 ("Because", "Therefore") → 1.0
  - Driving terms: 6 ("vehicle", "intersection", "pedestrian", "crossing",
                      "decelerate", "lane") → 1.0
  - Length: 228 chars → 1.0
  - No repetition → 1.0
  - Total: 4.0 / 4.0 = 1.0
```

---

#### `consistency_reward(completions, pred_xyz, **kwargs) -> list[float]`

**Purpose**: Reward for consistency between CoC text and predicted trajectory behavior.

**Algorithm**:
1. **Infer behaviors from trajectory** using `_trajectory_to_behaviors(pred_xyz)`:
   - **Lateral displacement** (y-axis in ego frame):
     - `y_end - y_start > 1.0m` → `turning_left`
     - `y_end - y_start < -1.0m` → `turning_right`
     - Otherwise → `going_straight`
   - **Longitudinal velocity change** (x-axis speed):
     - `speed_end < 0.05 m/s` → `stopping`
     - `speed_end > speed_start * 1.2` → `accelerating`
     - `speed_end < speed_start * 0.8` → `decelerating`

2. **Check CoC text for matching keywords**:
   - `turning_left`: ["turn left", "turning left", "left turn", "veer left"]
   - `turning_right`: ["turn right", "turning right", "right turn", "veer right"]
   - `going_straight`: ["straight", "continue ahead", "go straight", "maintain lane"]
   - `accelerating`: ["accelerat", "speed up", "faster", "increase speed"]
   - `decelerating`: ["decelerat", "slow down", "brake", "braking", "reduce speed"]
   - `stopping`: ["stop", "halt", "come to a stop", "standstill"]

3. **Compute reward**:
   ```python
   matched = sum(1 for b in behaviors if any(kw in text.lower() for kw in keywords[b]))
   reward = matched / len(behaviors)
   ```

**Example**:
```
Trajectory: Turns left (lateral displacement = +3m), slowing down (speed drops 40%)
Behaviors: {turning_left, decelerating}

Text: "The vehicle is turning left at the intersection and slowing down."
Match: Both "turning left" and "slowing down" mentioned
Reward: 2/2 = 1.0

Text: "The vehicle will accelerate and go straight."
Match: Neither mentioned
Reward: 0/2 = 0.0
```

---

### dataset.py

**Location**: `src/alpamayo_r1/training/dataset.py`

**Purpose**: Builds an HuggingFace `Dataset` from PhysicalAI-AV for GRPO training.

#### `_build_prompt_text(clip_id: str, t0_us: int) -> list[dict]`

Constructs a conversational prompt with embedded metadata.

**Format**:
```python
[
    {
        "role": "system",
        "content": "You are a driving assistant that generates safe and accurate actions. "
                   "[clip_id=abc123] [t0_us=5100000]"
    },
    {
        "role": "user",
        "content": "Given the driving scene, output the chain-of-thought reasoning "
                   "of the driving process, then output the future trajectory."
    },
]
```

**Why embed metadata**: The actual images and trajectory data are too large to include in the prompt. We load them at rollout time using the clip_id and t0_us encoded in the system message.

#### `build_alpamayo_dataset(split, t0_us, max_samples, clip_ids_file, avdi) -> Dataset`

Builds a dataset for training.

**Parameters**:
- `split`: Dataset split to use (`"train"`, `"val"`, `"test"`)
- `t0_us`: Default prediction timestamp (5.1 seconds into each clip)
- `max_samples`: Optional cap for debugging (e.g., 10 samples for smoke test)
- `clip_ids_file`: Optional path to parquet file with custom clip IDs
- `avdi`: Pre-initialized dataset interface (or creates one)

**Returns**: HF `Dataset` with columns:
- `prompt`: Chat-format messages (list of dicts)
- `clip_id`: String identifier
- `t0_us`: Integer timestamp

**Example record**:
```python
{
    "prompt": [
        {"role": "system", "content": "You are a driving assistant... [clip_id=abc] [t0_us=5100000]"},
        {"role": "user", "content": "Given the driving scene, output the chain-of-thought..."}
    ],
    "clip_id": "abc123",
    "t0_us": 5100000,
}
```

**Workflow**:
1. If `clip_ids_file` is provided, load clip IDs from parquet
2. Otherwise, query `avdi.clip_index` for valid clips in the specified split
3. Apply `max_samples` cap if set
4. Build one record per clip with `_build_prompt_text(clip_id, t0_us)`
5. Convert to HF Dataset

**Note**: The dataset is lightweight (just metadata). Heavy data (images, trajectories) is loaded at rollout time.

---

## Reward Functions

### Trajectory Quality Reward

**File**: `src/alpamayo_r1/training/rewards.py::trajectory_quality_reward`

**Purpose**: Measures how accurately the model predicts future trajectories.

**Metric**: minADE (minimum Average Displacement Error)

**Formula**:
```
For each trajectory sample s in {1, ..., S}:
    ADE_s = (1/T) * Σ_t ||pred_xy[s,t] - gt_xy[t]||_2

minADE = min_s ADE_s

reward = max(0, 1 - minADE / threshold)
```

**Hyperparameters**:
- `ade_threshold`: Default 5.0 meters
  - Predictions within 5m get positive reward
  - Predictions >5m off get reward=0

**Shape notes**:
- `pred_xyz`: Flattened list encoding shape `(S, T, 3)` where S=`num_traj_samples`, T=64
- `gt_xyz`: Flattened list encoding shape `(T, 3)`
- Only XY coordinates used (ignore Z/altitude)

**Example scores**:
| minADE (meters) | Reward |
|----------------|--------|
| 0.0            | 1.00   |
| 1.0            | 0.80   |
| 2.5            | 0.50   |
| 5.0            | 0.00   |
| 10.0           | 0.00   |

---

### Reasoning Quality Reward

**File**: `src/alpamayo_r1/training/rewards.py::reasoning_quality_reward`

**Purpose**: Measures the quality of chain-of-causation reasoning text using rule-based heuristics.

**Criteria** (each 0.25 points):

1. **Causal Connectors**: Logical connectors that indicate reasoning
   - **Patterns**: `because`, `therefore`, `since`, `thus`, `hence`, `as a result`, `so that`, `due to`, `in order to`, `consequently`, `leads to`, `causing`, `results in`
   - **Scoring**:
     - 1.0 points: ≥2 connectors
     - 0.5 points: 1 connector
     - 0.0 points: 0 connectors

2. **Driving Vocabulary**: Domain-specific terms
   - **Patterns**: `vehicle`, `car`, `truck`, `pedestrian`, `cyclist`, `lane`, `intersection`, `traffic`, `signal`, `light`, `stop`, `yield`, `merge`, `speed`, `brake`, `accelerat`, `steer`, `turn`, `left`, `right`, `straight`, `ahead`, `behind`, `front`, `rear`, `lateral`, `oncoming`, `highway`, `road`, `path`, `obstacle`, `distance`, `gap`, `follow`, `approach`, `slow`, `fast`
   - **Scoring**:
     - 1.0 points: ≥4 terms
     - 0.5 points: 2-3 terms
     - 0.0 points: 0-1 terms

3. **Appropriate Length**: Not too short, not too long
   - **Range**: 40-2000 characters
   - **Scoring**:
     - 1.0 points: Within range
     - 0.25 points: Non-empty but outside range
     - 0.0 points: Empty

4. **No Degenerate Repetition**: Avoid mode collapse
   - **Pattern**: Any substring ≥20 chars repeated ≥3 times consecutively
   - **Scoring**:
     - 1.0 points: No repetition detected
     - 0.0 points: Repetition detected

**Total Reward**: Sum of 4 criteria / 4.0 → range [0, 1]

**Example Texts**:

**Good reasoning (score ≈ 1.0)**:
```
"The ego vehicle is approaching a busy intersection. Because there is a pedestrian
crossing the street ahead and the traffic light is turning yellow, the vehicle
should decelerate to ensure safety. Therefore, the appropriate action is to slow
down gradually while maintaining the current lane position. Since the pedestrian
is expected to clear the crosswalk within 2-3 seconds, the vehicle can then
proceed straight through the intersection."

Score breakdown:
  - Causal connectors: 3 ("Because", "Therefore", "Since") → 1.0
  - Driving terms: 11 → 1.0
  - Length: 412 chars → 1.0
  - No repetition → 1.0
  Total: 1.0
```

**Poor reasoning (score ≈ 0.25)**:
```
"go straight"

Score breakdown:
  - Causal connectors: 0 → 0.0
  - Driving terms: 1 ("straight") → 0.0
  - Length: 11 chars (too short) → 0.25
  - No repetition → 1.0
  Total: 0.3125
```

**Degenerate repetition (score ≈ 0.5)**:
```
"the car is moving forward. the car is moving forward. the car is moving forward.
the car is moving forward. the car is moving forward."

Score breakdown:
  - Causal connectors: 0 → 0.0
  - Driving terms: 2 ("car", "moving") → 0.5
  - Length: 140 chars → 1.0
  - Repetition detected → 0.0
  Total: 0.375
```

---

### Consistency Reward

**File**: `src/alpamayo_r1/training/rewards.py::consistency_reward`

**Purpose**: Measures whether the CoC text description matches the predicted trajectory behavior.

**Behavior Detection** (from trajectory):

| Behavior | Detection Rule | Threshold |
|----------|---------------|-----------|
| `turning_left` | `lateral_displacement > 1.0` | +1.0m in y-axis |
| `turning_right` | `lateral_displacement < -1.0` | -1.0m in y-axis |
| `going_straight` | `-1.0 ≤ lateral_displacement ≤ 1.0` | ±1.0m in y-axis |
| `accelerating` | `speed_end > speed_start * 1.2` | +20% speed increase |
| `decelerating` | `speed_end < speed_start * 0.8` | -20% speed decrease |
| `stopping` | `speed_end < 0.05` | Near-zero final speed |

**Keyword Matching**:

Each behavior has associated keywords to search in CoC text (case-insensitive):

```python
BEHAVIOR_KEYWORDS = {
    "turning_left": ["turn left", "turning left", "left turn", "veer left"],
    "turning_right": ["turn right", "turning right", "right turn", "veer right"],
    "going_straight": ["straight", "continue ahead", "go straight", "maintain lane"],
    "accelerating": ["accelerat", "speed up", "faster", "increase speed"],
    "decelerating": ["decelerat", "slow down", "brake", "braking", "reduce speed"],
    "stopping": ["stop", "halt", "come to a stop", "standstill"],
}
```

**Reward Calculation**:
```python
traj_behaviors = _trajectory_to_behaviors(pred_xyz)
matched = 0
for behavior in traj_behaviors:
    keywords = BEHAVIOR_KEYWORDS[behavior]
    if any(kw in text.lower() for kw in keywords):
        matched += 1

reward = matched / len(traj_behaviors)
```

**Example**:

**Consistent (reward = 1.0)**:
```
Trajectory: Turns right (+3m lateral) and decelerates (speed drops 50%)
Behaviors: {turning_right, decelerating}

Text: "The vehicle is turning right into the parking lot and braking to a stop."
Matches: "turning right" ✓, "braking" ✓
Reward: 2/2 = 1.0
```

**Partially consistent (reward = 0.5)**:
```
Trajectory: Goes straight (0.3m lateral) and accelerates (speed up 30%)
Behaviors: {going_straight, accelerating}

Text: "The vehicle will continue straight ahead on the highway."
Matches: "straight ahead" ✓, "accelerating" ✗
Reward: 1/2 = 0.5
```

**Inconsistent (reward = 0.0)**:
```
Trajectory: Turns left (+2m lateral)
Behaviors: {turning_left}

Text: "The vehicle is turning right at the intersection."
Matches: "turning left" ✗
Reward: 0/1 = 0.0
```

---

## Configuration Reference

**File**: `src/alpamayo_r1/training/configs/grpo_default.yaml`

### General Settings

```yaml
seed: 42                              # Random seed for reproducibility
device: cuda                          # Device for training (cuda or cpu)
model_name: nvidia/Alpamayo-R1-10B    # HuggingFace model identifier
```

### LoRA Configuration

```yaml
lora:
  r: 16                   # LoRA rank (number of trainable parameters per layer)
  alpha: 32               # LoRA scaling factor (alpha/r = 2.0 scaling)
  dropout: 0.05           # Dropout rate for LoRA layers
  target_modules:         # Attention layers to apply LoRA
    - q_proj              # Query projection
    - k_proj              # Key projection
    - v_proj              # Value projection
    - o_proj              # Output projection
```

**Notes**:
- Higher `r` → more capacity but more parameters to train
- `alpha/r` ratio controls the effective learning rate for LoRA layers
- Only attention layers are targeted (not MLP layers)
- Task type is set to `CAUSAL_LM` automatically

### Training Configuration

```yaml
training:
  output_dir: outputs/grpo                    # Checkpoint and log directory
  num_train_epochs: 3                         # Number of passes through dataset
  per_device_train_batch_size: 1              # Batch size per GPU
  gradient_accumulation_steps: 16             # Accumulate gradients for effective batch size
  learning_rate: 1e-5                         # Adam learning rate
  num_generations: 8                          # G: number of samples per prompt for GRPO
  max_completion_length: 256                  # Max tokens to generate for CoC text
  beta: 0.0                                   # KL penalty coefficient (0 = no penalty)
  loss_type: grpo                             # Loss function (grpo or reinforce)
  logging_steps: 10                           # Log metrics every N steps
  save_steps: 200                             # Save checkpoint every N steps
  save_total_limit: 3                         # Keep only last 3 checkpoints
  warmup_ratio: 0.05                          # Fraction of steps for learning rate warmup
  max_grad_norm: 1.0                          # Gradient clipping threshold
  report_to: tensorboard                      # Logging backend (tensorboard, wandb, none)
```

**Key parameters**:

- **Effective batch size**: `per_device_train_batch_size * gradient_accumulation_steps * num_gpus`
  - Default: 1 * 16 * 1 = 16 prompts per update
  - Each prompt generates `num_generations=8` samples → 128 total samples per update

- **num_generations (G)**: Number of trajectory samples to generate per prompt
  - Used for computing group-relative advantages in GRPO
  - Higher G → more stable advantage estimates but slower rollouts
  - Must match `rollout.num_traj_samples`

- **beta**: KL divergence penalty coefficient
  - `beta=0.0` → No KL penalty, no reference model needed
  - `beta>0.0` → Penalize deviation from initial policy (requires reference model)

- **max_completion_length**: Max tokens for CoC text generation
  - Typical CoC length: 50-200 tokens
  - Set to 256 to allow detailed reasoning without truncation

### Dataset Configuration

```yaml
data:
  split: train                    # Dataset split (train, val, test)
  t0_us: 5100000                  # Prediction timestamp (5.1s into clip)
  max_samples: null               # Cap dataset size (null = use all)
  clip_ids_file: null             # Optional parquet file with clip_id column
```

**Notes**:
- `t0_us=5100000` → 5.1 seconds into each clip (leaves 1.6s history + 6.4s future)
- `max_samples` is useful for debugging (e.g., `max_samples: 10` for smoke test)
- `clip_ids_file` allows custom clip selection (e.g., hard examples)

### Rollout Configuration

```yaml
rollout:
  num_traj_samples: 8             # Must match training.num_generations
  temperature: 0.6                # Sampling temperature for VLM
  top_p: 0.98                     # Nucleus sampling threshold
  max_generation_length: 256      # Max tokens for CoC generation
```

**Sampling parameters**:
- **temperature**: Controls randomness
  - 0.6 → Balanced creativity and coherence
  - Lower (0.3) → More deterministic, less diverse
  - Higher (1.0) → More random, more diverse
- **top_p**: Nucleus sampling (cumulative probability cutoff)
  - 0.98 → Sample from top 98% of probability mass
  - Prevents sampling from very low-probability tokens

### Reward Configuration

```yaml
rewards:
  trajectory_weight: 0.50         # Weight for trajectory quality reward
  reasoning_weight: 0.25          # Weight for reasoning quality reward
  consistency_weight: 0.25        # Weight for consistency reward
```

**Must sum to 1.0** for interpretable weighted rewards.

**Tuning guidance**:
- High `trajectory_weight` → Prioritize accurate predictions
- High `reasoning_weight` → Prioritize detailed, coherent CoC text
- High `consistency_weight` → Prioritize text-trajectory alignment

**Example alternative weightings**:
```yaml
# Focus on trajectory accuracy
trajectory_weight: 0.7
reasoning_weight: 0.2
consistency_weight: 0.1

# Balance all three equally
trajectory_weight: 0.33
reasoning_weight: 0.34
consistency_weight: 0.33
```

---

## Quick Start

### Running a Smoke Test

Test the full pipeline with minimal resources (3 samples, 1 epoch, ~5 minutes):

```bash
./scripts/run_grpo.sh --smoke
```

This runs with:
- `data.max_samples=3`
- `training.num_train_epochs=1`
- `training.num_generations=2`
- `training.per_device_train_batch_size=1`
- `training.gradient_accumulation_steps=2`
- `training.save_steps=999999` (no checkpoints)
- `training.report_to=none` (no logging)

**Output**: Model saved to `outputs/grpo_smoke/final/`

---

### Full Training Run

Run with default config (3 epochs on full training set):

```bash
./scripts/run_grpo.sh
```

Or manually:

```bash
python -m alpamayo_r1.training.train_grpo --config-name grpo_default
```

**Expected runtime**:
- Dataset size: ~1000 clips (train split)
- Rollout time: ~3-5 seconds per prompt (with num_generations=8)
- Gradient steps: ~62 steps/epoch (1000 clips / 16 effective batch size)
- Total time: ~6-10 hours for 3 epochs on A100

**Monitoring**:
```bash
tensorboard --logdir outputs/grpo
```

---

### Custom Overrides

Override any config value via CLI:

```bash
# Use custom dataset
./scripts/run_grpo.sh --max-samples 100

# Change learning rate and batch size
./scripts/run_grpo.sh training.learning_rate=5e-6 training.gradient_accumulation_steps=32

# Adjust reward weights
./scripts/run_grpo.sh rewards.trajectory_weight=0.7 rewards.reasoning_weight=0.2 rewards.consistency_weight=0.1

# Change LoRA rank
./scripts/run_grpo.sh lora.r=32 lora.alpha=64

# Use custom clip IDs
./scripts/run_grpo.sh data.clip_ids_file=/path/to/custom_clips.parquet
```

**Hydra syntax**:
```bash
# Override single value
python -m alpamayo_r1.training.train_grpo key=value

# Override nested value
python -m alpamayo_r1.training.train_grpo training.learning_rate=1e-5

# Override list
python -m alpamayo_r1.training.train_grpo lora.target_modules=[q_proj,v_proj]
```

---

### Dry Run (Print Config)

Check the resolved configuration without running training:

```bash
./scripts/run_grpo.sh --dry-run
```

Or:

```bash
python -m alpamayo_r1.training.train_grpo --config-name grpo_default --cfg job
```

---

## Design Decisions

### Why LoRA on VLM Only?

**Decision**: Apply LoRA only to VLM attention layers; freeze Expert, Diffusion, and Action Space modules.

**Rationale**:

1. **VLM generates reasoning text**: The CoC text quality directly affects rewards. By training the VLM, we optimize the reasoning process.

2. **Expert/Diffusion are already well-trained**: These components were trained in earlier stages (Stage 1: supervised trajectory prediction, Stage 2: supervised CoC generation). They don't need further updates for RL.

3. **Parameter efficiency**: LoRA adds ~30M trainable parameters (r=16) vs. 10B base model. Freezing non-VLM components saves memory and compute.

4. **Stable trajectory prediction**: Freezing Expert/Diffusion ensures trajectory quality remains high throughout RL training. Only the reasoning text adapts to reward signals.

**What gets frozen**:
```python
for name, param in model.named_parameters():
    if not name.startswith("vlm."):
        param.requires_grad = False
```

This freezes:
- `expert.*` — Expert Transformer layers
- `diffusion.*` — Flow Matching Diffusion model
- `action_in_proj.*` — Noisy action → expert token embedding projection
- `action_out_proj.*` — Expert hidden → action prediction projection
- `action_space.*` — Action space conversion utilities

**What gets trained**:
- `vlm.model.layers[*].self_attn.q_proj.lora_A` — LoRA query adapters
- `vlm.model.layers[*].self_attn.k_proj.lora_A` — LoRA key adapters
- `vlm.model.layers[*].self_attn.v_proj.lora_A` — LoRA value adapters
- `vlm.model.layers[*].self_attn.o_proj.lora_A` — LoRA output adapters
- Corresponding `lora_B` matrices

---

### Why beta=0.0 (No KL Penalty)?

**Decision**: Set `beta=0.0` in GRPO config, disabling KL divergence penalty between policy and reference model.

**What is the KL penalty**:
```
loss = -log_prob * advantage + beta * KL(policy || reference)
```
Where `KL(policy || reference)` measures how much the current policy deviates from an initial reference policy.

**Why disable it**:

1. **No reference model needed**: With `beta=0.0`, we don't need to load and maintain a separate reference model, saving ~20 GB GPU memory.

2. **Exploration encouraged**: Without KL penalty, the model can explore further from the initial policy to maximize rewards. This is beneficial when the supervised pre-training (Stage 2) may not be optimal for the RL objectives.

3. **Simpler training**: Fewer hyperparameters to tune (no beta scheduling).

**Trade-off**:
- **Pro**: More efficient, better exploration, no reference model overhead
- **Con**: Policy may diverge too far and generate off-distribution text (mode collapse risk)

**Mitigation**: The `reasoning_quality_reward` includes heuristics to prevent degenerate text (repetition detection, length bounds).

**When to use beta>0**:
- If you observe mode collapse (repetitive or nonsensical CoC text)
- If you want to stay close to the supervised checkpoint
- If you have enough GPU memory for a reference model

**Alternative**: Use `beta=0.01` or `beta=0.001` for light regularization:
```bash
./scripts/run_grpo.sh training.beta=0.01
```

---

### Why Override _generate_single_turn?

**Decision**: Subclass `GRPOTrainer` and override `_generate_single_turn` instead of using TRL's `rollout_func` hook.

**Background**: TRL provides `rollout_func` as a customization point for generating samples during GRPO training. However, this is only invoked when using vLLM for high-throughput inference.

**Why we can't use `rollout_func`**:

1. **vLLM not applicable**: Our generation involves Expert + Diffusion, not just VLM text generation. vLLM doesn't support this custom pipeline.

2. **Need trajectory predictions**: Reward functions require `pred_xyz` (predicted trajectories), which aren't available from text-only generation.

3. **Complex multi-stage pipeline**: We need to:
   - Parse clip metadata from prompt
   - Load driving data (images, egomotion)
   - Run VLM + Expert + Diffusion sequentially
   - Extract both text and trajectories
   - Compute log-probs via separate forward pass

**Solution**: Override `_generate_single_turn(prompts) -> (prompt_ids, completion_ids, logprobs, extra)` directly. This is the internal method TRL calls during training, and we can return all the data needed for rewards.

**What we keep from TRL**:
- Training loop and logging
- GRPO loss computation
- Advantage estimation (group-relative)
- Gradient accumulation and checkpointing
- Optimizer and scheduler management

**What we customize**:
- Prompt → (text, trajectories) generation logic
- Log-prob computation (teacher-forced VLM forward)
- Extra fields forwarding (`pred_xyz`, `gt_xyz`)

**Code structure**:
```python
class AlpamayoGRPOTrainer(GRPOTrainer):
    def __init__(self, full_model, avdi, rollout_temperature, ...):
        super().__init__(model=full_model.vlm, ...)  # Pass VLM to parent
        self.full_model = full_model  # Store full model for rollout
        self.avdi = avdi
        # ...

    def _generate_single_turn(self, prompts):
        # Custom generation logic
        for prompt in prompts:
            clip_id, t0_us = parse_metadata(prompt)
            data = load_data(clip_id, t0_us)
            pred_xyz, pred_rot, extra = self.full_model.sample_trajectories_...()
            logprobs = compute_logprobs(...)
        return prompt_ids, completion_ids, logprobs, extra_fields
```

---

### Input IDs Clone Pattern

**Decision**: Clone `input_ids` before passing to the model.

```python
prompt_input_ids = model_inputs["tokenized_data"]["input_ids"].clone()
```

**Why needed**: The Alpamayo model **pops** `input_ids` from the input dict during forward pass:

```python
# In alpamayo_r1/models/alpamayo_r1.py
input_ids = tokenized_data.pop("input_ids")  # Removes from dict!
```

**Problem**: After running the generation, we need `input_ids` again to compute log-probs via teacher-forced forward pass. But it's been removed from `model_inputs`.

**Solution**: Clone `input_ids` before the generation:
```python
prompt_input_ids = model_inputs["tokenized_data"]["input_ids"].clone()
# ... run generation (pops input_ids) ...
# Later: use prompt_input_ids for log-prob computation
```

**Alternative (not used)**: Modify the model to not pop `input_ids`. But this would break the existing interface and affect other code paths (inference, evaluation).

---

### Gradient Checkpointing Toggle

**Decision**: Temporarily disable gradient checkpointing during rollout, re-enable during backward pass.

**Code**:
```python
gc_enabled = getattr(self.full_model.vlm, "is_gradient_checkpointing", False)
if gc_enabled:
    self.full_model.vlm.gradient_checkpointing_disable()

# ... run generation with use_cache=True ...

if gc_enabled:
    self.full_model.vlm.gradient_checkpointing_enable()
```

**Why needed**: The Alpamayo pipeline requires `use_cache=True` for the VLM's KV cache (consumed by the Expert model). However, gradient checkpointing and `use_cache` are mutually incompatible in HuggingFace Transformers:

```
# From transformers/modeling_utils.py
if self.is_gradient_checkpointing and use_cache:
    raise ValueError("Cannot use cache with gradient checkpointing")
```

**Workflow**:
1. **During rollout** (generation + log-prob computation):
   - Disable gradient checkpointing temporarily
   - Enable `use_cache=True` for KV cache
   - Run generation in `torch.no_grad()` context (no backward pass)

2. **During backward pass** (GRPO loss):
   - Re-enable gradient checkpointing
   - Only compute gradients for LoRA parameters
   - Save memory by recomputing activations instead of storing them

**Why safe**: Rollout is always in inference mode (`torch.no_grad()`), so gradient checkpointing doesn't apply. We only need it disabled to avoid the error when `use_cache=True`.

**Memory impact**: Negligible, since we're not storing activations during rollout anyway.

---

## Testing

### CPU Tests

**File**: `tests/test_training.py`

**Purpose**: Unit tests for training components that don't require a GPU or the full model.

**Run**:
```bash
pytest tests/test_training.py -v
```

**Test coverage**:

1. **Import tests**: Verify all training modules can be imported
   ```python
   def test_import_rewards():
       from alpamayo_r1.training.rewards import trajectory_quality_reward
   ```

2. **Reward function tests**:
   - `TestTrajectoryQualityReward`: Test minADE computation, soft threshold, min-over-samples
   - `TestReasoningQualityReward`: Test causal connectors, driving terms, length, repetition
   - `TestConsistencyReward`: Test behavior detection and keyword matching

3. **Dataset utility tests**:
   - `test_build_prompt_text_format`: Verify prompt structure
   - `test_clip_metadata_roundtrip`: Encode clip_id/t0_us → parse back

4. **Rollout utility tests**:
   - `test_parse_clip_metadata`: Regex parsing
   - `test_collate_rollout_outputs`: Padding and batching logic

5. **Config tests**:
   - `test_config_loads`: YAML parsing
   - `test_config_reward_weights_sum_to_one`: Validate weights

**Example test**:
```python
def test_perfect_prediction(self):
    """Identical pred and gt should give reward close to 1.0."""
    T = 64
    gt = np.zeros((T, 3), dtype=np.float32).flatten().tolist()
    pred = np.zeros((3, T, 3), dtype=np.float32).flatten().tolist()
    rewards = trajectory_quality_reward(["dummy"], pred_xyz=[pred], gt_xyz=[gt])
    assert rewards[0] == pytest.approx(1.0, abs=0.01)
```

**Runtime**: ~5 seconds (CPU only)

---

### GPU Smoke Test

**File**: `tests/test_training_gpu.py`

**Purpose**: End-to-end integration test with the full model on GPU. Verifies the entire pipeline works.

**Requirements**:
- NVIDIA GPU with ≥24 GB VRAM
- HuggingFace authentication (model weights access)

**Run**:
```bash
python tests/test_training_gpu.py
```

**Test workflow**:

1. **Load model**: `AlpamayoR1.from_pretrained(...).to("cuda")`
2. **Get test clip**: First valid clip from dataset
3. **Test dataset builder**: `_build_prompt_text()` and metadata roundtrip
4. **Test prepare_model_inputs**: Tokenization and data formatting
5. **Run inference**: Generate trajectories + CoC text
6. **Verify output shapes**: Check `pred_xyz`, `pred_rot`, `extra["cot"]`
7. **Test reward functions**: Compute all three rewards on real outputs
8. **Test rollout function**: Run the full rollout as GRPO would use it
9. **Test parameter freezing**: Verify non-VLM params are frozen

**Sample output**:
```
============================================================
  1. Loading model
============================================================
   Model loaded in 23.4s
   GPU memory used: 21.3 GB

============================================================
  2. Getting test clip
============================================================
   Using clip: 2021-07-13-16-48-42_f1

============================================================
  5. Running inference
============================================================
   Inference done in 4.2s
   pred_xyz shape: torch.Size([1, 1, 2, 64, 3])
   pred_rot shape: torch.Size([1, 1, 2, 64, 3, 3])
   Got 2 CoC texts
   CoC[0]: The ego vehicle is approaching an intersection. Because the traffic light is green and...

============================================================
  6. Testing reward functions
============================================================
   trajectory_quality: 0.8234
   reasoning_quality:  0.7500
   consistency:        1.0000
   weighted total:     0.8234

============================================================
  ALL TESTS PASSED
============================================================
   Peak GPU memory: 23.1 GB
```

**Runtime**: ~1-2 minutes (including model download)

**Troubleshooting**:
- **Out of memory**: Reduce `num_traj_samples` in the test
- **Model not found**: Check HuggingFace authentication
- **Slow download**: First run downloads 22 GB model weights

---

## Data Flow Diagram

### Training Step Detailed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ GRPOTrainer.training_step()                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ AlpamayoGRPOTrainer._generate_single_turn(prompts)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: prompts = [P1, P1, ..., P1, P2, P2, ..., P2]          │
│          (each prompt repeated G=num_generations times)         │
│                                                                 │
│  1. De-duplicate: unique_prompts = [P1, P2]                    │
│                                                                 │
│  2. For each prompt P:                                          │
│     ┌───────────────────────────────────────────────────────┐  │
│     │ a) Parse metadata                                      │  │
│     │    prompt_text = "... [clip_id=abc] [t0_us=5100000]"  │  │
│     │    clip_id, t0_us = _parse_clip_metadata(prompt_text) │  │
│     └───────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│     ┌───────────────────────────────────────────────────────┐  │
│     │ b) Load driving data                                   │  │
│     │    data = load_physical_aiavdataset(clip_id, t0_us)   │  │
│     │    • image_frames: (4, 4, 3, H, W)                    │  │
│     │    • ego_history_xyz: (1, 1, 16, 3)                   │  │
│     │    • ego_future_xyz: (1, 1, 64, 3)                    │  │
│     └───────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│     ┌───────────────────────────────────────────────────────┐  │
│     │ c) Prepare model inputs                                │  │
│     │    model_inputs = helper.prepare_model_inputs()       │  │
│     │    • tokenized_data: {input_ids, pixel_values, ...}   │  │
│     │    • ego_history_xyz, ego_history_rot                 │  │
│     │                                                        │  │
│     │    prompt_input_ids = input_ids.clone()  ← IMPORTANT  │  │
│     └───────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│     ┌───────────────────────────────────────────────────────┐  │
│     │ d) Disable gradient checkpointing                      │  │
│     │    (needed for use_cache=True in generation)          │  │
│     └───────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│     ┌───────────────────────────────────────────────────────┐  │
│     │ e) Run full Alpamayo pipeline (G times)               │  │
│     │    pred_xyz, pred_rot, extra =                         │  │
│     │      full_model.sample_trajectories_from_data(...)    │  │
│     │                                                        │  │
│     │    ┌─────────────────────────────────────────────┐    │  │
│     │    │ e.1) VLM generates CoC text                 │    │  │
│     │    │      • Top-p/temperature sampling           │    │  │
│     │    │      • Stop at <|traj_future_start|>        │    │  │
│     │    │      • Output: token IDs + KV cache         │    │  │
│     │    └─────────────────────────────────────────────┘    │  │
│     │                    │                                   │  │
│     │                    ▼                                   │  │
│     │    ┌─────────────────────────────────────────────┐    │  │
│     │    │ e.2) Expert Transformer                     │    │  │
│     │    │      • Uses VLM's KV cache as context       │    │  │
│     │    │      • Processes noisy action tokens        │    │  │
│     │    │      • Iterative denoising (diffusion)      │    │  │
│     │    └─────────────────────────────────────────────┘    │  │
│     │                    │                                   │  │
│     │                    ▼                                   │  │
│     │    ┌─────────────────────────────────────────────┐    │  │
│     │    │ e.3) Flow Matching Diffusion                │    │  │
│     │    │      • Samples from learned distribution    │    │  │
│     │    │      • Output: action tokens                │    │  │
│     │    └─────────────────────────────────────────────┘    │  │
│     │                    │                                   │  │
│     │                    ▼                                   │  │
│     │    ┌─────────────────────────────────────────────┐    │  │
│     │    │ e.4) Action Space Converter                 │    │  │
│     │    │      • action → (xyz, rotation)             │    │  │
│     │    └─────────────────────────────────────────────┘    │  │
│     │                                                        │  │
│     │    Output shape:                                       │  │
│     │      pred_xyz: (1, 1, G, 64, 3)                       │  │
│     │      extra["cot"]: (1, 1, G) CoC text strings         │  │
│     └───────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│     ┌───────────────────────────────────────────────────────┐  │
│     │ f) Tokenize completions                                │  │
│     │    for coc_text in extra["cot"]:                       │  │
│     │        completion_ids = tokenizer.encode(coc_text)     │  │
│     └───────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│     ┌───────────────────────────────────────────────────────┐  │
│     │ g) Compute log-probs (teacher-forced VLM forward)     │  │
│     │    logprobs = _compute_batch_logprobs(                │  │
│     │        full_model,                                     │  │
│     │        model_inputs,                                   │  │
│     │        prompt_input_ids,  ← Use cloned input_ids      │  │
│     │        completion_ids_list,                            │  │
│     │        prompt_len,                                     │  │
│     │        device                                          │  │
│     │    )                                                   │  │
│     │                                                        │  │
│     │    For each completion:                                │  │
│     │      1. Cat prompt + completion tokens                 │  │
│     │      2. Forward pass: VLM(full_ids)                    │  │
│     │      3. Extract logits for completion region           │  │
│     │      4. Log-softmax and gather token log-probs         │  │
│     └───────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│     ┌───────────────────────────────────────────────────────┐  │
│     │ h) Store results                                       │  │
│     │    all_prompt_ids.append(prompt_ids_list)             │  │
│     │    all_completion_ids.append(completion_ids)          │  │
│     │    all_logprobs.append(logprobs)                      │  │
│     │    all_pred_xyz.append(pred_xyz.flatten())            │  │
│     │    all_gt_xyz.append(gt_xyz.flatten())                │  │
│     └───────────────────────────────────────────────────────┘  │
│                                                                 │
│  3. Re-enable gradient checkpointing                            │
│                                                                 │
│  4. Return to TRL:                                              │
│     • prompt_ids: List[List[int]]                              │
│     • completion_ids: List[List[int]]                          │
│     • logprobs: List[List[float]]                              │
│     • extra_fields: {pred_xyz, gt_xyz}                         │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ GRPOTrainer.compute_rewards(completions, **extra_fields)       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Call reward functions:                                      │
│     r1 = trajectory_quality_reward(completions, pred_xyz, gt)  │
│     r2 = reasoning_quality_reward(completions)                 │
│     r3 = consistency_reward(completions, pred_xyz)             │
│                                                                 │
│  2. Weighted sum:                                               │
│     rewards = w1*r1 + w2*r2 + w3*r3                            │
│                                                                 │
│  Output: rewards = [r1, r2, ..., r_{B*G}]                      │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ GRPOTrainer.compute_loss(logprobs, rewards)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Group rewards by prompt (each prompt has G samples):       │
│     rewards_grouped = rewards.reshape(B, G)                    │
│                                                                 │
│  2. Compute group baselines (mean reward per group):           │
│     baselines = rewards_grouped.mean(dim=1, keepdim=True)      │
│                                                                 │
│  3. Compute advantages (reward - baseline):                    │
│     advantages = rewards_grouped - baselines  # (B, G)         │
│                                                                 │
│  4. GRPO loss (negative log-prob weighted by advantage):       │
│     log_probs_summed = sum(logprobs per completion)            │
│     loss = -mean(log_probs_summed * advantages)                │
│                                                                 │
│  5. Backward pass:                                              │
│     loss.backward() → gradients only for VLM LoRA parameters   │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Optimizer.step() → Update VLM LoRA weights                     │
└─────────────────────────────────────────────────────────────────┘
```

### Key Observations

1. **Each prompt is processed independently** → No batch processing in rollout (batch size = 1 per unique prompt)
2. **G samples per prompt** → Group-relative advantages for stable training
3. **Log-probs computed separately** → Teacher-forced forward pass after generation
4. **Extra fields flow through** → pred_xyz/gt_xyz passed from rollout to rewards
5. **Only VLM gets gradients** → Expert/Diffusion frozen, LoRA on attention layers

---

## Troubleshooting

### Out of Memory Errors

**Symptom**: CUDA OOM during training

**Solutions**:
1. Reduce `num_generations` (e.g., 4 instead of 8)
2. Reduce `gradient_accumulation_steps` (smaller effective batch size)
3. Reduce `rollout.max_generation_length` (fewer CoC tokens)
4. Enable gradient checkpointing for VLM (if not already enabled)
5. Use a GPU with more VRAM (≥40 GB recommended for full config)

**Config overrides**:
```bash
./scripts/run_grpo.sh \
    training.num_generations=4 \
    training.gradient_accumulation_steps=8 \
    rollout.max_generation_length=128
```

---

### Slow Training

**Symptom**: Rollout takes >10 seconds per prompt

**Causes**:
1. **High `num_generations`**: More samples per prompt → longer rollout
2. **Large `max_generation_length`**: More autoregressive steps
3. **Slow data loading**: PhysicalAI-AV streaming from HuggingFace

**Solutions**:
1. Download dataset locally before training:
   ```python
   from physical_ai_av import PhysicalAIAVDatasetInterface
   avdi = PhysicalAIAVDatasetInterface(maybe_stream=False)  # Force download
   ```
2. Reduce `num_generations` or `max_generation_length`
3. Profile with `nsys` or `torch.profiler` to identify bottlenecks

---

### Reward Signals Not Improving

**Symptom**: Rewards remain flat or decrease during training

**Possible causes**:
1. **Learning rate too high**: Model diverges
2. **Learning rate too low**: Model doesn't learn
3. **Reward weights imbalanced**: One reward dominates
4. **Mode collapse**: Model generates repetitive text

**Debugging**:
1. Check tensorboard for reward breakdown:
   ```bash
   tensorboard --logdir outputs/grpo
   ```
   Look at `rewards/trajectory_quality`, `rewards/reasoning_quality`, `rewards/consistency`

2. Inspect generated text in logs:
   ```python
   # Add logging in AlpamayoGRPOTrainer._generate_single_turn
   logger.info(f"Generated CoC: {coc_text}")
   ```

3. Try different learning rates:
   ```bash
   ./scripts/run_grpo.sh training.learning_rate=5e-6  # Lower
   ./scripts/run_grpo.sh training.learning_rate=2e-5  # Higher
   ```

4. Adjust reward weights to emphasize different objectives:
   ```bash
   # Focus on trajectory quality
   ./scripts/run_grpo.sh \
       rewards.trajectory_weight=0.7 \
       rewards.reasoning_weight=0.2 \
       rewards.consistency_weight=0.1
   ```

---

### ValueError: Could not parse clip metadata

**Symptom**:
```
ValueError: Could not parse clip metadata from prompt: ...
```

**Cause**: The prompt text doesn't contain `[clip_id=...] [t0_us=...]` markers

**Check**:
1. Verify dataset was built with `_build_prompt_text()`:
   ```python
   dataset = build_alpamayo_dataset(...)
   print(dataset[0]["prompt"])
   # Should contain [clip_id=...] [t0_us=...]
   ```

2. Ensure TRL is passing prompts correctly (should be list of message dicts or concatenated strings)

---

### Flash Attention Errors

**Symptom**: Flash Attention import fails or raises errors

**Solution**: Use PyTorch's SDPA implementation instead:
```python
# In train_grpo.py, after loading model:
full_model.vlm.config.attn_implementation = "sdpa"
```

Or set at model load time:
```python
config = AlpamayoR1Config.from_pretrained(model_name)
config.attn_implementation = "sdpa"
model = AlpamayoR1.from_pretrained(model_name, config=config)
```

---

### LoRA Not Saving Correctly

**Symptom**: After training, the saved model doesn't include LoRA weights

**Check**:
1. Verify `trainer.save_model()` is called at the end
2. Check output directory contains `adapter_config.json` and `adapter_model.safetensors`
3. Load and verify:
   ```python
   from peft import PeftModel
   base_model = AlpamayoR1.from_pretrained(model_name)
   model = PeftModel.from_pretrained(base_model.vlm, "outputs/grpo/final")
   ```

---

### Dataset Too Large

**Symptom**: Training takes too long with full dataset

**Solution**: Use `max_samples` or `clip_ids_file` to train on a subset:

```bash
# Train on first 100 clips
./scripts/run_grpo.sh --max-samples 100

# Train on custom clip IDs
./scripts/run_grpo.sh data.clip_ids_file=/path/to/hard_examples.parquet
```

**Create custom clip list**:
```python
import pandas as pd
from physical_ai_av import PhysicalAIAVDatasetInterface

avdi = PhysicalAIAVDatasetInterface()
clip_index = avdi.clip_index

# Select clips with specific criteria (e.g., intersections, urban scenes)
selected_clips = clip_index[
    (clip_index["split"] == "train") &
    (clip_index["clip_is_valid"]) &
    (clip_index["scene_type"] == "urban_intersection")
].index.tolist()

# Save to parquet
df = pd.DataFrame({"clip_id": selected_clips})
df.to_parquet("custom_clips.parquet")
```

---

## References

- **Alpamayo-R1 Paper**: [arXiv:2511.00088](https://arxiv.org/abs/2511.00088)
- **HuggingFace Model Card**: [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- **PhysicalAI-AV Dataset**: [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- **TRL Documentation**: [Transformer Reinforcement Learning](https://huggingface.co/docs/trl/index)
- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)

---

## Appendix: File Locations

```
src/alpamayo_r1/training/
├── __init__.py                     # Package init
├── train_grpo.py                   # Entry point (main function)
├── rollout.py                      # AlpamayoGRPOTrainer subclass
├── rewards.py                      # 3 reward functions
├── dataset.py                      # HF Dataset builder
└── configs/
    └── grpo_default.yaml           # Default Hydra config

scripts/
└── run_grpo.sh                     # Shell script wrapper

tests/
├── test_training.py                # CPU unit tests
└── test_training_gpu.py            # GPU smoke test

docs/
└── grpo-training.md                # This document
```

---

**Last updated**: 2026-02-09
