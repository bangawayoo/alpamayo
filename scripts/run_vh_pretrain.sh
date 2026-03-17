#!/usr/bin/env bash
# Value Head Pre-Training (Stage 0) for Alpamayo-R1
#
# Generates rollouts from the base policy, extracts segment hidden states,
# and trains the three-level value head (obs/coc/traj) for multiple epochs.
# The saved checkpoint can then be loaded by the full SFT pipeline via
# value_head.load_path=.
#
# Usage:
#   ./scripts/run_vh_pretrain.sh                          # default: 500 samples, 50 epochs, 1 GPU
#   ./scripts/run_vh_pretrain.sh --max-samples 200        # custom sample count
#   ./scripts/run_vh_pretrain.sh --epochs 100             # custom training epochs
#   ./scripts/run_vh_pretrain.sh --num-gpus 4             # multi-GPU DDP
#   ./scripts/run_vh_pretrain.sh --smoke                  # quick test (5 samples, 2 epochs)
#   ./scripts/run_vh_pretrain.sh --dry-run                # print config and exit
#   ./scripts/run_vh_pretrain.sh --output outputs/vh_v2   # custom output directory

set -euo pipefail
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"

# ---------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------
SMOKE=0
DRY_RUN=0
MAX_SAMPLES=""
NUM_EPOCHS=""
NUM_GPUS=""
ACCELERATE_CONFIG=""
OUTPUT_DIR=""
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
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --accelerate-config)
            ACCELERATE_CONFIG="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
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
SAMPLES="${MAX_SAMPLES:-500}"
EPOCHS="${NUM_EPOCHS:-50}"

OVERRIDES=(
    # Skip the self-play loop — only run Stage 0 (value head pre-training)
    "advantage_conditioning.num_iterations=0"
    # Ensure value head and segment-level are enabled
    "value_head.enabled=true"
    "value_head.segment_level=true"
)

if [[ "$SMOKE" -eq 1 ]]; then
    echo "=== SMOKE TEST MODE ==="
    OVERRIDES+=(
        "data.max_samples=10"
        "value_head.pretrain_scenes=5"
        "value_head.pretrain_epochs=2"
        "training.output_dir=outputs/vh_pretrain_smoke"
        "training.report_to=none"
        "rollout.max_generation_length=128"
        "advantage_conditioning.completions_per_scene=2"
    )
else
    OVERRIDES+=(
        "data.max_samples=$SAMPLES"
        "value_head.pretrain_scenes=$SAMPLES"
        "value_head.pretrain_epochs=$EPOCHS"
    )
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    OVERRIDES+=("training.output_dir=$OUTPUT_DIR")
fi

OVERRIDES+=("${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}")

# ---------------------------------------------------------------
# Environment
# ---------------------------------------------------------------
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# ---------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------
# Auto-detect GPU count if not specified
if [[ -z "$NUM_GPUS" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    else
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    if [[ "$NUM_GPUS" -lt 1 ]]; then
        NUM_GPUS=1
    fi
fi

# Select accelerate config (DDP for multi-GPU rollout sharding)
if [[ -z "$ACCELERATE_CONFIG" ]]; then
    ACCELERATE_CONFIG="$PROJECT_ROOT/src/alpamayo_r1/training/configs/accelerate_ddp.yaml"
fi
if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
    echo "Error: accelerate config not found at $ACCELERATE_CONFIG"
    exit 1
fi

echo "================================================================"
echo "  Alpamayo-R1 Value Head Pre-Training (Stage 0)"
echo "================================================================"

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "N/A")
    echo "GPU: $GPU_INFO"
else
    echo "WARNING: nvidia-smi not found. Training requires a GPU."
fi

echo "Python: $VENV_PYTHON"
echo "Project: $PROJECT_ROOT"
echo "GPUs: $NUM_GPUS"
echo ""

# ---------------------------------------------------------------
# Print config
# ---------------------------------------------------------------
echo "Hydra overrides:"
for ov in "${OVERRIDES[@]}"; do
    echo "  $ov"
done
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "=== DRY RUN — printing resolved config ==="
    "$VENV_PYTHON" -m alpamayo_r1.training.train_sft \
        --config-name sft_default \
        --cfg job \
        "${OVERRIDES[@]}"
    exit 0
fi

# ---------------------------------------------------------------
# Run value head pre-training
# ---------------------------------------------------------------
echo "Starting value head pre-training..."
echo "================================================================"

"$VENV_PYTHON" -m accelerate.commands.launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_GPUS" \
    -m alpamayo_r1.training.train_sft \
    --config-name sft_default \
    "${OVERRIDES[@]}"

echo ""
echo "================================================================"
echo "  Value head pre-training complete!"
echo "  Checkpoint saved to: $(grep -o 'training.output_dir=[^ ]*' <<< "${OVERRIDES[*]}" | cut -d= -f2 || echo 'outputs/sft_advcond')/value_head_pretrained.pt"
echo ""
echo "  To use in SFT training:"
echo "    ./scripts/run_sft.sh value_head.load_path=<path>/value_head_pretrained.pt value_head.pretrain_scenes=0"
echo "================================================================"
