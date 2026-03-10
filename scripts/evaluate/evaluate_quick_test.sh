#!/bin/bash
# Curated test set evaluation (1,181 clips from notebooks/clip_ids.parquet)
#
# Usage:
#   ./evaluate_quick_test.sh                          # single GPU
#   ./evaluate_quick_test.sh --num-gpus 4             # multi-GPU data parallelism
#   ./evaluate_quick_test.sh --seed 123               # custom seed
#   ./evaluate_quick_test.sh --temperature 0          # greedy decoding
#   ./evaluate_quick_test.sh --num-traj-samples 20    # more trajectory samples
#   ./evaluate_quick_test.sh --top-p 0.95             # nucleus sampling threshold
#   ./evaluate_quick_test.sh --num-trials 3           # repeat with seeds 42,43,44 and aggregate

set -euo pipefail

NUM_GPUS=1
MODEL="nvidia/Alpamayo-R1-10B"
SEED=42
OUTPUT_DIR=""
NUM_TRAJ_SAMPLES=5
TEMPERATURE=0.6
TOP_P=0.98
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
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"

echo "Running curated test set evaluation (1,181 clips)..."
echo "  Model: ${MODEL}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Traj samples: ${NUM_TRAJ_SAMPLES}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Top-p: ${TOP_P}"
echo "  Trials: ${NUM_TRIALS} (seeds ${SEED}...$((SEED + NUM_TRIALS - 1)))"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# run_trial TRIAL_OUTPUT_DIR TRIAL_SEED
#   Runs a single evaluation (single- or multi-GPU) into TRIAL_OUTPUT_DIR.
# ---------------------------------------------------------------------------
run_trial() {
    local trial_output_dir="$1"
    local trial_seed="$2"

    if [[ "${NUM_GPUS}" -eq 1 ]]; then
        python src/alpamayo_r1/evaluate_test_set.py \
            --model-name "${MODEL}" \
            --num-traj-samples "${NUM_TRAJ_SAMPLES}" \
            --temperature "${TEMPERATURE}" \
            --top-p "${TOP_P}" \
            --output-dir "${trial_output_dir}" \
            --use-clip-ids-file \
            --num-workers 4 \
            --prefetch-factor 2 \
            --seed "${trial_seed}" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    else
        echo "Launching ${NUM_GPUS} shards in parallel..."
        local PIDS=()
        for ((i=0; i<NUM_GPUS; i++)); do
            CUDA_VISIBLE_DEVICES="${i}" python src/alpamayo_r1/evaluate_test_set.py \
                --model-name "${MODEL}" \
                --num-traj-samples "${NUM_TRAJ_SAMPLES}" \
                --temperature "${TEMPERATURE}" \
                --top-p "${TOP_P}" \
                --output-dir "${trial_output_dir}" \
                --use-clip-ids-file \
                --num-workers 8 \
                --prefetch-factor 2 \
                --seed "${trial_seed}" \
                --shard-id "${i}" \
                --num-shards "${NUM_GPUS}" \
                "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
            PIDS+=($!)
        done

        echo "Waiting for all shards to complete..."
        local FAILED=0
        for pid in "${PIDS[@]}"; do
            if ! wait "${pid}"; then
                FAILED=$((FAILED + 1))
            fi
        done

        if [[ "${FAILED}" -gt 0 ]]; then
            echo "ERROR: ${FAILED} shard(s) failed."
            exit 1
        fi

        echo ""
        echo "All shards complete. Merging shard results..."
        python -c "
import pandas as pd, json, numpy as np
from pathlib import Path

output_dir = Path('${trial_output_dir}')
dfs = []
for p in sorted(output_dir.glob('results_shard*.csv')):
    dfs.append(pd.read_csv(p))
if not dfs:
    print('No shard results found!'); exit(1)

df = pd.concat(dfs, ignore_index=True)
df.to_csv(output_dir / 'results.csv', index=False)

ok = df[df['success'] == True]
stats = {
    'total_samples': len(df),
    'successful_samples': len(ok),
    'failed_samples': len(df) - len(ok),
}
if len(ok) > 0:
    for m in ['minADE', 'minFDE']:
        v = ok[m].values
        stats[m] = {'mean': float(np.mean(v)), 'median': float(np.median(v)),
                     'std': float(np.std(v)), 'min': float(np.min(v)), 'max': float(np.max(v))}
with open(output_dir / 'statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f'Merged {len(dfs)} shards: {len(df)} total, {len(ok)} successful')
if len(ok) > 0:
    print(f'  minADE mean: {stats[\"minADE\"][\"mean\"]:.4f}')
    print(f'  minFDE mean: {stats[\"minFDE\"][\"mean\"]:.4f}')
"
    fi
}

# ---------------------------------------------------------------------------
# Run trials
# ---------------------------------------------------------------------------
for ((t=0; t<NUM_TRIALS; t++)); do
    TRIAL_SEED=$((SEED + t))

    if [[ "${NUM_TRIALS}" -gt 1 ]]; then
        TRIAL_DIR="${OUTPUT_DIR}/trials/trial_${t}"
        echo "=== Trial $((t+1))/${NUM_TRIALS} (seed=${TRIAL_SEED}) -> ${TRIAL_DIR} ==="
    else
        TRIAL_DIR="${OUTPUT_DIR}"
    fi

    run_trial "${TRIAL_DIR}" "${TRIAL_SEED}"
done

# ---------------------------------------------------------------------------
# Aggregate across trials (only when NUM_TRIALS > 1)
# ---------------------------------------------------------------------------
if [[ "${NUM_TRIALS}" -gt 1 ]]; then
    echo ""
    echo "=== Aggregating ${NUM_TRIALS} trials ==="
    python -c "
import pandas as pd, json, numpy as np
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
trials_dir = output_dir / 'trials'
num_trials = ${NUM_TRIALS}

# Load per-clip results from every trial
trial_dfs = []
for t in range(num_trials):
    p = trials_dir / f'trial_{t}' / 'results.csv'
    df = pd.read_csv(p)
    df['trial'] = t
    trial_dfs.append(df)

all_trials = pd.concat(trial_dfs, ignore_index=True)
all_trials.to_csv(output_dir / 'trials' / 'all_trials.csv', index=False)

# Per-clip average across trials (only successful rows)
ok = all_trials[all_trials['success'] == True]
per_clip_mean = ok.groupby('clip_id')[['minADE', 'minFDE']].mean().reset_index()
per_clip_mean['success'] = True
per_clip_mean.to_csv(output_dir / 'results.csv', index=False)

# Trial-level means (to report variance across trials)
trial_means = ok.groupby('trial')[['minADE', 'minFDE']].mean()

stats = {
    'num_trials': num_trials,
    'total_clips': per_clip_mean['clip_id'].nunique(),
}
for m in ['minADE', 'minFDE']:
    vals = trial_means[m].values
    stats[m] = {
        'mean_of_trials': float(np.mean(vals)),
        'std_of_trials':  float(np.std(vals)),
        'trial_values':   [float(v) for v in vals],
    }

with open(output_dir / 'statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f'Aggregated {num_trials} trials over {stats[\"total_clips\"]} clips')
for m in ['minADE', 'minFDE']:
    print(f'  {m}: {stats[m][\"mean_of_trials\"]:.4f} +/- {stats[m][\"std_of_trials\"]:.4f}  (per-trial: {[f\"{v:.4f}\" for v in stats[m][\"trial_values\"]]})')
"
fi

echo ""
echo "Done."
