#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GRPO Post-Training for Alpamayo-R1
#
# Usage:
#   ./scripts/run_grpo.sh                              # full training run (single GPU)
#   ./scripts/run_grpo.sh --smoke                      # quick smoke test (3 samples, 1 epoch)
#   ./scripts/run_grpo.sh --dry-run                    # print config and exit
#   ./scripts/run_grpo.sh --max-samples 50             # limit dataset size
#   ./scripts/run_grpo.sh --vllm                       # enable vLLM colocate mode for fast generation
#   ./scripts/run_grpo.sh --vllm --smoke               # vLLM smoke test
#   ./scripts/run_grpo.sh --vllm-server                # enable vLLM server mode (requires trl vllm-serve)
#   ./scripts/run_grpo.sh --vllm-server --smoke        # vLLM server smoke test
#   ./scripts/run_grpo.sh --fsdp                       # multi-GPU FSDP training (auto-detect GPUs)
#   ./scripts/run_grpo.sh --fsdp --num-gpus 2          # FSDP with explicit GPU count
#   ./scripts/run_grpo.sh --fsdp --num-gpus 2 --smoke  # FSDP smoke test
#   ./scripts/run_grpo.sh --no-fsdp                    # DDP mode (no FSDP sharding, auto-detect GPUs)
#   ./scripts/run_grpo.sh --no-fsdp --num-gpus 1       # force single-GPU via accelerate
#   ./scripts/run_grpo.sh --no-lora                    # full-parameter training (no LoRA adapters)
#   ./scripts/run_grpo.sh --no-lora --fsdp             # full-parameter with FSDP multi-GPU
#   ./scripts/run_grpo.sh --fsdp --accelerate-config path/to/config.yaml  # custom accelerate config

set -euo pipefail
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
# ---------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------
SMOKE=0
DRY_RUN=0
FSDP=1
NO_LORA=0
VLLM=0
VLLM_SERVER=0
NUM_GPUS=""
ACCELERATE_CONFIG=""
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
        --vllm)
            VLLM=1
            shift
            ;;
        --vllm-server)
            VLLM_SERVER=1
            shift
            ;;
        --fsdp)
            FSDP=1
            shift
            ;;
        --no-fsdp)
            FSDP=0
            shift
            ;;
        --no-lora)
            NO_LORA=1
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --accelerate-config)
            ACCELERATE_CONFIG="$2"
            shift 2
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
VENV_PYTHON="/home/jovyan/conda/.venv/bin/python"


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

if [[ "$NO_LORA" -eq 1 ]]; then
    echo "=== FULL-PARAMETER TRAINING (no LoRA) ==="
    OVERRIDES+=("lora.enabled=false")
fi

if [[ "$VLLM" -eq 1 ]]; then
    echo "=== vLLM COLOCATE MODE ==="
    OVERRIDES+=("vllm.enabled=true")
fi

if [[ "$VLLM_SERVER" -eq 1 ]]; then
    echo "=== vLLM SERVER MODE ==="
    OVERRIDES+=("vllm.enabled=true" "vllm.mode=server")
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

# Select accelerate config
if [[ -z "$ACCELERATE_CONFIG" ]]; then
    if [[ "$FSDP" -eq 1 ]]; then
        ACCELERATE_CONFIG="$PROJECT_ROOT/src/alpamayo_r1/training/configs/accelerate_fsdp.yaml"
    else
        ACCELERATE_CONFIG="$PROJECT_ROOT/src/alpamayo_r1/training/configs/accelerate_ddp.yaml"
    fi
fi
if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
    echo "Error: accelerate config not found at $ACCELERATE_CONFIG"
    exit 1
fi

# Auto-detect GPU count if not specified
if [[ -z "$NUM_GPUS" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        # Count comma-separated device IDs
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    else
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    if [[ "$NUM_GPUS" -lt 1 ]]; then
        NUM_GPUS=1
    fi
fi

if [[ "$FSDP" -eq 1 ]]; then
    echo "FSDP enabled: $NUM_GPUS GPU(s), config=$ACCELERATE_CONFIG"
else
    echo "DDP enabled: $NUM_GPUS GPU(s), config=$ACCELERATE_CONFIG"
fi

"$VENV_PYTHON" -m accelerate.commands.launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_GPUS" \
    -m alpamayo_r1.training.train_grpo \
    --config-name grpo_default \
    "${OVERRIDES[@]+"${OVERRIDES[@]}"}"

echo ""
echo "================================================================"
echo "  Training complete!"
echo "================================================================"
