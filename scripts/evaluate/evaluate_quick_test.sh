#!/bin/bash
# Curated test set evaluation (1,181 clips from notebooks/clip_ids.parquet)
#
# Usage:
#   ./evaluate_quick_test.sh                          # auto-detect GPUs
#   ./evaluate_quick_test.sh --num-gpus 4             # multi-GPU data parallelism (1 trial)
#   ./evaluate_quick_test.sh --seed 123               # custom seed
#   ./evaluate_quick_test.sh --temperature 0          # greedy decoding
#   ./evaluate_quick_test.sh --num-traj-samples 20    # more trajectory samples
#   ./evaluate_quick_test.sh --top-p 0.95             # nucleus sampling threshold
#   ./evaluate_quick_test.sh --traj-mode vlm          # VLM-only discrete trajectory tokens
#   ./evaluate_quick_test.sh --num-trials 4 --num-gpus 4   # 4 trials in parallel, 1 GPU each
#
# Multi-trial parallelism:
#   When --num-trials > 1, trials run in parallel batches of --num-gpus.
#   Each trial is assigned one GPU (CUDA_VISIBLE_DEVICES=i).
#   Wall clock ≈ ceil(num_trials / num_gpus) × single_trial_time.
#
#   When --num-trials 1 (default), all --num-gpus are used for clip sharding
#   within a single trial (original behavior).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
set -a; source "$REPO_ROOT/.env" 2>/dev/null || true; set +a

NUM_GPUS=""
MODEL="nvidia/Alpamayo-R1-10B"
SEED=42
OUTPUT_DIR=""
NUM_TRAJ_SAMPLES=5
TEMPERATURE=0.6
TOP_P=0.98
TRAJ_MODE="expert"
NUM_TRIALS=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num-traj-samples)
            NUM_TRAJ_SAMPLES="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --traj-mode)
            TRAJ_MODE="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default output dir derived from model path if not specified
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="evaluation_results/curated_set-$(basename "$MODEL")"
fi
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



echo "Running curated test set evaluation (1,181 clips)..."
echo "  Model: ${MODEL}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Traj samples: ${NUM_TRAJ_SAMPLES}"
echo "  Traj mode: ${TRAJ_MODE}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Top-p: ${TOP_P}"
if [[ "${NUM_TRIALS}" -gt 1 ]]; then
    echo "  Trials: ${NUM_TRIALS} (seeds ${SEED}..$((SEED + NUM_TRIALS - 1)), ${NUM_GPUS} parallel)"
else
    echo "  Trials: 1 (seed ${SEED})"
fi
echo "  Output: ${OUTPUT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# run_trial OUTPUT_DIR SEED
#   Runs a single-GPU evaluation. Caller sets CUDA_VISIBLE_DEVICES as needed.
# ---------------------------------------------------------------------------
run_trial() {
    local trial_output_dir="$1"
    local trial_seed="$2"
    python src/alpamayo_r1/evaluate_test_set.py \
        --model-name "${MODEL}" \
        --num-traj-samples "${NUM_TRAJ_SAMPLES}" \
        --traj-mode "${TRAJ_MODE}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --output-dir "${trial_output_dir}" \
        --use-clip-ids-file \
        --num-workers 4 \
        --prefetch-factor 2 \
        --seed "${trial_seed}" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

# ---------------------------------------------------------------------------
# Single-trial path: use all GPUs for clip sharding (original behavior)
# ---------------------------------------------------------------------------
if [[ "${NUM_TRIALS}" -eq 1 ]]; then
    if [[ "${NUM_GPUS}" -eq 1 ]]; then
        run_trial "${OUTPUT_DIR}" "${SEED}"
    else
        echo "Launching ${NUM_GPUS} shards in parallel..."
        PIDS=()
        for ((i=0; i<NUM_GPUS; i++)); do
            CUDA_VISIBLE_DEVICES="${i}" python src/alpamayo_r1/evaluate_test_set.py \
                --model-name "${MODEL}" \
                --num-traj-samples "${NUM_TRAJ_SAMPLES}" \
                --traj-mode "${TRAJ_MODE}" \
                --temperature "${TEMPERATURE}" \
                --top-p "${TOP_P}" \
                --output-dir "${OUTPUT_DIR}" \
                --use-clip-ids-file \
                --num-workers 8 \
                --prefetch-factor 2 \
                --seed "${SEED}" \
                --shard-id "${i}" \
                --num-shards "${NUM_GPUS}" \
                "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
            PIDS+=($!)
        done

        echo "Waiting for all shards to complete..."
        FAILED=0
        for pid in "${PIDS[@]}"; do
            if ! wait "${pid}"; then FAILED=$((FAILED + 1)); fi
        done
        if [[ "${FAILED}" -gt 0 ]]; then
            echo "ERROR: ${FAILED} shard(s) failed."
            exit 1
        fi

        echo "All shards complete. Merging results..."
        python src/alpamayo_r1/evaluate_test_set.py \
            --merge-shards \
            --output-dir "${OUTPUT_DIR}"
    fi

# ---------------------------------------------------------------------------
# Multi-trial path: 1 GPU per trial, run NUM_GPUS trials in parallel per batch
# ---------------------------------------------------------------------------
else
    TRIALS_DIR="${OUTPUT_DIR}/trials"
    mkdir -p "${TRIALS_DIR}"

    for ((batch_start=0; batch_start<NUM_TRIALS; batch_start+=NUM_GPUS)); do
        batch_end=$((batch_start + NUM_GPUS))
        if [[ "${batch_end}" -gt "${NUM_TRIALS}" ]]; then
            batch_end="${NUM_TRIALS}"
        fi
        batch_size=$((batch_end - batch_start))

        echo "=== Batch trials ${batch_start}..$((batch_end-1)) (${batch_size} parallel) ==="
        PIDS=()
        for ((t=batch_start; t<batch_end; t++)); do
            GPU=$((t - batch_start))
            TRIAL_SEED=$((SEED + t))
            TRIAL_DIR="${TRIALS_DIR}/trial_${t}"
            echo "  Trial ${t} → GPU ${GPU} (seed=${TRIAL_SEED})"
            CUDA_VISIBLE_DEVICES="${GPU}" run_trial "${TRIAL_DIR}" "${TRIAL_SEED}" &
            PIDS+=($!)
        done

        FAILED=0
        for pid in "${PIDS[@]}"; do
            if ! wait "${pid}"; then FAILED=$((FAILED + 1)); fi
        done
        if [[ "${FAILED}" -gt 0 ]]; then
            echo "ERROR: ${FAILED} trial(s) in batch failed."
            exit 1
        fi
        echo "  Batch complete."
        echo ""
    done

    # Aggregate across all trials
    echo "=== Aggregating ${NUM_TRIALS} trials ==="
    python src/alpamayo_r1/evaluate_test_set.py \
        --aggregate-trials "${NUM_TRIALS}" \
        --output-dir "${OUTPUT_DIR}"
fi

echo ""
echo "Done."
