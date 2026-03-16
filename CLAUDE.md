# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alpamayo-R1 is a Vision-Language-Action (VLA) model for autonomous driving that bridges reasoning and action prediction. It uses a Qwen3-VL backbone to generate Chain-of-Causation (CoC) text reasoning 
followed by discrete trajectory tokens, trained via GRPO (Group Relative Policy Optimization).

When writing new code, do not put any license statements like below:
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

## Environment

- Development environment uses MIG notebook (10GB VRAM)
- Run enviornment uses multi-gpu
- Python enviornment is in conda's .venv environment
- When the conversation history includes requests for working in a worktree or the user has asked to do so, always work in the worktree unless otherwise requested. If unsure, ask the user.

## Running

```bash
# Inference
python src/alpamayo_r1/test_inference.py

# GRPO training
./scripts/run_grpo.sh                     # full run (FSDP enabled by default)
./scripts/run_grpo.sh --smoke             # smoke test (3 samples, 1 epoch)
./scripts/run_grpo.sh --dry-run           # print resolved Hydra config
./scripts/run_grpo.sh --no-fsdp           # single-GPU mode
./scripts/run_grpo.sh --fsdp --num-gpus 2 # explicit GPU count

# Reward signal evaluation
python scripts/evaluate/evaluate_reward_signal.py --num-samples 50
```

Training uses Hydra for config management. Extra args to `run_grpo.sh` are passed as Hydra overrides (e.g., `training.num_train_epochs=3`). The default config is `src/alpamayo_r1/training/configs/grpo_default.yaml`.

## Tests

```bash
pytest tests/test_training.py       # CPU-only tests
pytest tests/test_training_gpu.py   # GPU-dependent tests
pytest tests/test_training.py -k "test_name"  # single test
```

No CI/CD pipeline — all testing is manual.

## Linting

```bash
ruff check src/                     # lint
ruff format src/                    # format
```

Ruff config in pyproject.toml: line-length = 100. No other linting tools configured.

## Architecture

### Model Hierarchy

`AlpamayoR1` → `ReasoningVLA` (base_model.py) → includes `TrajectoryFusionMixin`

- **VLM** (Qwen3-VL): Generates CoC reasoning text + 64 discrete trajectory tokens
- **Trajectory Tokenizers**: Encode/decode between continuous (x,y,z) waypoints and 768 discrete tokens (`<i0>`..`<i767>`)
  - `hist_traj_tokenizer` — history (16 tokens)
  - `traj_tokenizer` — future (64 tokens)
- **Expert Transformer + Diffusion**: Post-processing components, kept on CPU during GRPO to save memory
- **Config**: `AlpamayoR1Config` extends `ReasoningVLAConfig` (config.py)

### Special Tokens

History: `<|traj_history_start|>`, `<|traj_history|>`, `<|traj_history_end|>`
Future: `<|traj_future_start|>`, `<|traj_future|>`, `<|traj_future_end|>`
CoC: `<|cot_start|>`, `<|cot_end|>`

### GRPO Training Pipeline (src/alpamayo_r1/training/)

1. **train_grpo.py** — Hydra entry point, builds model/dataset/trainer
2. **rollout.py** — `AlpamayoGRPOTrainer` (extends TRL's GRPOTrainer), `ClipDataCache` for lazy data loading
3. **rewards.py** — Three reward functions: `trajectory_quality_reward` (minADE, 50%), `reasoning_quality_reward` (rule-based, 25%), `consistency_reward` (CoC/trajectory agreement, 25%)
4. **dataset.py** — Builds dataset from PhysicalAI-AV with clip_id/t0_us metadata embedded in prompts

**Key design**: VLM-only rollouts during GRPO — the VLM generates both CoC text and trajectory tokens directly (no Expert/Diffusion). Generation stops at `<|traj_future_end|>`. Only VLM is trained via LoRA (r=16, alpha=32, targets: q/k/v/o_proj).

### Token Processing (models/token_utils.py)

- `extract_traj_tokens()` normalizes by subtracting `future_token_start_idx`
- `extract_text_tokens()` extracts CoC text between special token boundaries
- `StopAfterEOS` deliberately generates one extra token after EOS (by design)
- `fuse_traj_tokens()` (in base_model.py) replaces placeholder token IDs at the token ID level

## Code Conventions

- Imperative mood for commit messages
- Follow existing conventions in each file/module

## Response Conventions

- You can freely make revisions to code to improve them. When you do, always tell which files have changed at the end of your response.