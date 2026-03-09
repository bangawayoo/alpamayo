# GRPO Post-Training for Alpamayo-R1

## Overview

The GRPO (Group Relative Policy Optimization) module implements Stage 3 RL post-training for Alpamayo-R1, as described in Section 3.3 of the [paper](https://arxiv.org/abs/2511.00088). The VLM autoregressively generates both Chain-of-Causation (CoC) reasoning text and discrete trajectory tokens during rollouts. No Expert or Diffusion modules are used — trajectory tokens are decoded to continuous xyz waypoints via the trajectory tokenizer.

Training uses [TRL](https://huggingface.co/docs/trl/index)'s `GRPOTrainer` with a custom subclass (`AlpamayoGRPOTrainer`) that handles the multi-modal rollout pipeline.

---

## Architecture

### VLM-Only Rollout Pipeline

```
Dataset (PhysicalAI-AV clips)
    │
    │ Each sample: {prompt with [clip_id=...] [t0_us=...]}
    │
    ▼
AlpamayoGRPOTrainer._generate_single_turn
    │
    ├─ 1. Parse clip_id/t0_us from prompt metadata
    │
    ├─ 2. Load driving data (16 camera images, ego history/future)
    │      └─ ClipDataCache: lazy-loads and caches in CPU RAM
    │
    ├─ 3. Fuse history trajectory tokens into input_ids
    │      └─ full_model.fuse_traj_tokens(input_ids, traj_data)
    │
    ├─ 4. VLM generates CoC text + 64 trajectory tokens
    │      └─ vlm.generate() with temperature/top_p sampling
    │      └─ Stops at <|traj_future_end|> token
    │
    ├─ 5. Decode trajectory tokens → continuous xyz
    │      └─ extract_traj_tokens() → traj_tokenizer.decode()
    │      └─ Output: pred_xyz (num_samples, 64, 3)
    │
    ├─ 6. Compute per-token log-probs (teacher-forced VLM forward)
    │      └─ _compute_batch_logprobs() in mini-batches
    │
    └─> Return: (prompt_ids, completion_ids, logprobs, {pred_xyz, gt_xyz})
                    │
                    ▼
            TRL GRPOTrainer
                    │
                    ├─ Compute rewards (3 functions, weighted sum)
                    ├─ Group-relative advantage estimation
                    ├─ GRPO clipped loss → backprop through VLM (LoRA only)
                    └─ Optimizer step
```

**Key design choice**: The VLM directly produces discrete trajectory tokens (`<i0>`..\`<i767>`), which are decoded via `traj_tokenizer.decode()` to continuous (x, y, z) waypoints. Expert and Diffusion modules stay on CPU and are not used during GRPO training following the paper's implementation.

---

## Modules

| File | Purpose |
|------|---------|
| `train_grpo.py` | Hydra entry point. Loads model, applies LoRA, builds datasets, creates trainer. |
| `rollout.py` | `AlpamayoGRPOTrainer` — overrides `_generate_single_turn` for VLM-only rollouts, `log()` for eval metric plumbing, `_save()` for PEFT compatibility. Also contains `ClipDataCache`, `RolloutLoggingCallback`, `GpuUtilizationCallback`. |
| `rewards.py` | Three reward functions: trajectory quality, reasoning quality, consistency. |
| `dataset.py` | Builds lightweight HF Dataset with conversational prompts embedding clip metadata. |
| `configs/grpo_default.yaml` | Default Hydra config with all hyperparameters. |

### AlpamayoGRPOTrainer

Subclasses TRL's `GRPOTrainer`. The `model` passed to the parent is `full_model.vlm` (the VLM component only), which is what TRL wraps with LoRA and trains. The full `AlpamayoR1` model is stored separately for trajectory tokenizer access.

Key overrides:
- **`_generate_single_turn`**: VLM-only rollout with trajectory token decoding (see diagram above). Also supports vLLM delegation via `rollout_func` when `vllm.enabled=true`.
- **`log`**: Mutates the eval metrics dict in-place so that `_determine_best_metric` and `EarlyStoppingCallback` can find reward metrics (works around a TRL bug where `GRPOTrainer.log()` creates a new dict instead of updating the original).
- **`_save`**: Passes `save_embedding_layers=True` to PEFT because Qwen3VLConfig doesn't expose `vocab_size`, causing PEFT's auto-detection to crash.

---

## Reward Functions

All reward functions follow TRL's interface: `f(completions, **kwargs) -> list[float]`, returning values in [0, 1].

### 1. Trajectory Quality (`trajectory_quality_reward`)

Measures prediction accuracy using minADE (minimum Average Displacement Error).

```
reward = max(0, 1 - minADE / 5.0)
```

- Uses only xy coordinates (ignores z/altitude)
- `min` over trajectory samples encourages diversity (best-of-N)
- Threshold of 5.0m: perfect prediction → 1.0, 5m+ error → 0.0

### 2. Reasoning Quality (`reasoning_quality_reward`)

Rule-based heuristic scoring CoC text on 4 criteria (each worth 0.25):

| Criterion | Full score | Partial | Zero |
|-----------|-----------|---------|------|
| Causal connectors ("because", "therefore", ...) | 2+ matches | 1 match | 0 |
| Driving vocabulary ("vehicle", "lane", "brake", ...) | 4+ terms | 2-3 terms | 0-1 |
| Appropriate length (40-2000 chars) | In range | Non-empty | Empty |
| No degenerate repetition (20+ char repeated 3x) | No repeat | — | Repeat found |

**Note**: The paper uses a reasoning critic for this reward. The rule-based heuristic is a simplified stand-in.

### 3. Consistency (`consistency_reward`)

Checks whether CoC text mentions behaviors that match the predicted trajectory (turning, braking, etc.). Currently **disabled by default** (weight=0.0) because the coarse behavior extraction is too noisy.

### Default Weights

```yaml
trajectory_weight: 0.50   # minADE-based, grounded
reasoning_weight:  0.50   # rule-based heuristic
consistency_weight: 0.00  # disabled
```

---

## Configuration

All config is in `src/alpamayo_r1/training/configs/grpo_default.yaml`. Override any value via CLI Hydra overrides.

### Key Parameters

```yaml
# LoRA (applied to VLM attention layers only)
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

# Training
training:
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 1
  learning_rate: 1e-5
  num_generations: 8              # G in GRPO (group size)
  max_completion_length: 384      # CoC (≤256) + 64 traj tokens + overhead
  gradient_checkpointing: true
  beta: 0.0                       # no KL penalty → no reference model

# Rollout
rollout:
  temperature: 0.6
  top_p: 0.98
  max_generation_length: 256      # max CoC text tokens
  logprob_mini_batch_size: 2      # completions per forward pass

# Dataset
data:
  split: train
  t0_us: 5100000                  # 5.1s into each clip
  max_samples: 3000
  exclude_clip_ids_file: notebooks/clip_ids.parquet  # exclude eval clips

# Early stopping
early_stopping:
  enabled: true
  patience: 5
  eval_steps: 100
  eval_max_samples: 30
  metric: rewards/trajectory_quality_reward/mean
```


---

## Quick Start

```bash
# Smoke test (3 samples, 1 epoch)
./scripts/run_grpo.sh --smoke

# Full training run (DDP, auto-detected GPUs)
./scripts/run_grpo.sh --no-fsdp

# Single-GPU mode
./scripts/run_grpo.sh --no-fsdp --num-gpus 1

# Custom overrides
./scripts/run_grpo.sh --no-fsdp training.learning_rate=5e-6 rewards.trajectory_weight=0.7

# Dry run (print resolved config)
./scripts/run_grpo.sh --dry-run

# Monitor training
tensorboard --logdir outputs/grpo
```

> **Note**: FSDP is currently broken (conflicts with LoRA + Qwen3-VL tied embeddings during checkpoint saving). Use `--no-fsdp` for DDP mode.

---

## Design Decisions

### Why VLM-Only Rollouts?

The paper's full pipeline runs VLM → Expert → Diffusion → Action Space, but during GRPO the VLM is trained to generate trajectory tokens directly.

### Why Override `_generate_single_turn`?

TRL's `rollout_func` hook is only invoked in vLLM code paths. For HuggingFace generation, we override `_generate_single_turn` directly to:
- Parse clip metadata from prompts and load driving data
- Run the VLM with fused history trajectory tokens
- Extract and decode trajectory tokens for reward computation
- Compute per-token log-probs via a separate teacher-forced forward pass

### Why Clone `input_ids`?

The Alpamayo model **pops** `input_ids` from the input dict during forward pass (`tokenized_data.pop("input_ids")`). We clone it before generation so it's available for the log-prob computation afterward.

---

## Troubleshooting

**OOM errors**: Reduce `num_generations` (e.g., 4), `logprob_mini_batch_size` (e.g., 1), or `max_generation_length`.

**Slow rollouts**: Download the PhysicalAI-AV dataset locally (streaming is slow) through [download_eval_clips.py](../scripts/data/download_eval_clips.py). `ClipDataCache` caches loaded clips in CPU RAM to avoid redundant I/O.

**FSDP crashes**: Use `--no-fsdp` for DDP mode. FSDP conflicts with LoRA + Qwen3-VL during checkpoint saving.

---

## References

- [Alpamayo-R1 Paper (arXiv:2511.00088)](https://arxiv.org/abs/2511.00088)
- [TRL GRPOTrainer Documentation](https://huggingface.co/docs/trl/index)
- [GRPO Paper (arXiv:2402.03300)](https://arxiv.org/abs/2402.03300)
- [PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
