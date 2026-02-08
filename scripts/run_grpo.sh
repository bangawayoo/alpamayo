#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GRPO Post-Training for Alpamayo-R1
#
# Usage:
#   ./scripts/run_grpo.sh                    # full training run
#   ./scripts/run_grpo.sh --smoke            # quick smoke test (3 samples, 1 epoch)
#   ./scripts/run_grpo.sh --dry-run          # print config and exit
#   ./scripts/run_grpo.sh --max-samples 50   # limit dataset size

set -euo pipefail

# ---------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------
SMOKE=0
DRY_RUN=0
MAX_SAMPLES=""
EXTRA_OVERRIDES=()

# ---------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            # Pass through any other args as Hydra overrides
            EXTRA_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: venv python not found at $VENV_PYTHON"
    echo "Run 'uv sync' first to create the virtual environment."
    exit 1
fi

# ---------------------------------------------------------------
# Build Hydra overrides
# ---------------------------------------------------------------
OVERRIDES=()

if [[ "$SMOKE" -eq 1 ]]; then
    echo "=== SMOKE TEST MODE ==="
    OVERRIDES+=(
        "data.max_samples=3"
        "training.num_train_epochs=1"
        "training.num_generations=2"
        "training.per_device_train_batch_size=1"
        "training.gradient_accumulation_steps=2"
        "training.logging_steps=1"
        "training.save_steps=999999"
        "training.output_dir=outputs/grpo_smoke"
        "training.report_to=none"
        "rollout.num_traj_samples=2"
        "rollout.max_generation_length=128"
    )
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    OVERRIDES+=("data.max_samples=$MAX_SAMPLES")
fi

OVERRIDES+=("${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}")

# ---------------------------------------------------------------
# Environment
# ---------------------------------------------------------------
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# ---------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------
echo "================================================================"
echo "  Alpamayo-R1 GRPO Post-Training"
echo "================================================================"

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "N/A")
    echo "GPU: $GPU_INFO"
else
    echo "WARNING: nvidia-smi not found. Training requires a GPU."
fi

echo "Python: $VENV_PYTHON"
echo "Project: $PROJECT_ROOT"
echo ""

# ---------------------------------------------------------------
# Print config
# ---------------------------------------------------------------
echo "Hydra overrides:"
if [[ ${#OVERRIDES[@]} -eq 0 ]]; then
    echo "  (none — using grpo_default.yaml as-is)"
else
    for ov in "${OVERRIDES[@]}"; do
        echo "  $ov"
    done
fi
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "=== DRY RUN — printing resolved config ==="
    "$VENV_PYTHON" -m alpamayo_r1.training.train_grpo \
        --config-name grpo_default \
        --cfg job \
        "${OVERRIDES[@]+"${OVERRIDES[@]}"}"
    exit 0
fi

# ---------------------------------------------------------------
# Run training
# ---------------------------------------------------------------
echo "Starting training..."
echo "================================================================"

"$VENV_PYTHON" -m alpamayo_r1.training.train_grpo \
    --config-name grpo_default \
    "${OVERRIDES[@]+"${OVERRIDES[@]}"}"

echo ""
echo "================================================================"
echo "  Training complete!"
echo "================================================================"
