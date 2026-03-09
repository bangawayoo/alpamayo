#!/bin/bash
# Curated test set evaluation (1,181 clips from notebooks/clip_ids.parquet)
#
# Usage:
#   ./evaluate_quick_test.sh                     # single GPU
#   ./evaluate_quick_test.sh --num-gpus 4        # multi-GPU data parallelism

set -euo pipefail

NUM_GPUS=1
MODEL="nvidia/Alpamayo-R1-10B"
OUTPUT_DIR=""
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
echo "  Output: ${OUTPUT_DIR}"
echo ""

if [[ "${NUM_GPUS}" -eq 1 ]]; then
    python src/alpamayo_r1/evaluate_test_set.py \
        --model-name "${MODEL}" \
        --num-traj-samples 5 \
        --temperature 0.6 \
        --top-p 0.98 \
        --output-dir "${OUTPUT_DIR}" \
        --use-clip-ids-file \
        --num-workers 4 \
        --prefetch-factor 2 \
        --seed 42 \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
    echo "Launching ${NUM_GPUS} shards in parallel..."
    PIDS=()
    for ((i=0; i<NUM_GPUS; i++)); do
        CUDA_VISIBLE_DEVICES="${i}" python src/alpamayo_r1/evaluate_test_set.py \
            --model-name "${MODEL}" \
            --num-traj-samples 5 \
            --temperature 0.6 \
            --top-p 0.98 \
            --output-dir "${OUTPUT_DIR}" \
            --use-clip-ids-file \
            --num-workers 4 \
            --prefetch-factor 2 \
            --seed 42 \
            --shard-id "${i}" \
            --num-shards "${NUM_GPUS}" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
        PIDS+=($!)
    done

    echo "Waiting for all shards to complete..."
    FAILED=0
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
    echo "All shards complete. Merging results..."
    python -c "
import pandas as pd, json, numpy as np
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
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

echo "Done."
