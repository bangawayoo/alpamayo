#!/usr/bin/env bash
# Run Flow Matching Action Expert training.
#
# Usage:
#   ./scripts/run_expert.sh                        # full training
#   ./scripts/run_expert.sh --smoke                 # smoke test (3 samples, 1 epoch)
#   ./scripts/run_expert.sh --dry-run               # print resolved config
#   ./scripts/run_expert.sh --num-gpus 4            # multi-GPU with torchrun
#   ./scripts/run_expert.sh training.num_epochs=5   # Hydra overrides
#
# Environment:
#   Expects conda's .venv environment with alpamayo_r1 installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Parse CLI flags
SMOKE=false
DRY_RUN=false
NUM_GPUS=1
HYDRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            HYDRA_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

# Smoke test overrides
if $SMOKE; then
    HYDRA_OVERRIDES+=(
        "data.max_samples=3"
        "training.num_epochs=1"
        "training.num_noisy_samples=2"
        "training.gradient_accumulation_steps=1"
        "training.logging_steps=1"
        "training.save_steps=999999"
        "eval.enabled=false"
        "training.output_dir=outputs/expert_smoke"
    )
fi

# Dry-run: resolve and print config
if $DRY_RUN; then
    python -m alpamayo_r1.training.train_expert \
        --config-name expert_default \
        --cfg job \
        "${HYDRA_OVERRIDES[@]}"
    exit 0
fi

# Set up logging
export PYTHONUNBUFFERED=1

if [[ "$NUM_GPUS" -gt 1 ]]; then
    echo "Launching expert training with torchrun ($NUM_GPUS GPUs)..."
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29501 \
        -m alpamayo_r1.training.train_expert \
        --config-name expert_default \
        "${HYDRA_OVERRIDES[@]}"
else
    echo "Launching expert training (single GPU)..."
    python -m alpamayo_r1.training.train_expert \
        --config-name expert_default \
        "${HYDRA_OVERRIDES[@]}"
fi
