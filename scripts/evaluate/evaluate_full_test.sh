#!/bin/bash
# Full test set evaluation (~61,599 valid test clips from PhysicalAI-AV)
#
# Usage:
#   ./evaluate_full_test.sh                      # auto-detect GPUs
#   ./evaluate_full_test.sh --num-gpus 4         # multi-GPU data parallelism

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
set -a; source "$REPO_ROOT/.env" 2>/dev/null || true; set +a

NUM_GPUS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

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

OUTPUT_DIR="evaluation_results/full_test_set"

echo "Running full test set evaluation..."
echo "  Note: This evaluates ALL ~61,599 valid test clips and will take a long time."
echo "  Estimated time: ~17-34 hours on a single A100 (1-2s per sample)"
echo "  GPUs: ${NUM_GPUS}"
echo "  Output: ${OUTPUT_DIR}"
echo ""
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled."
    exit 1
fi

if [[ "${NUM_GPUS}" -eq 1 ]]; then
    python src/alpamayo_r1/evaluate_test_set.py \
        --num-traj-samples 5 \
        --temperature 0.6 \
        --top-p 0.98 \
        --output-dir "${OUTPUT_DIR}" \
        --num-workers 8 \
        --prefetch-factor 3 \
        --seed 42 \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
    echo "Launching ${NUM_GPUS} shards in parallel..."
    PIDS=()
    for ((i=0; i<NUM_GPUS; i++)); do
        CUDA_VISIBLE_DEVICES="${i}" python src/alpamayo_r1/evaluate_test_set.py \
            --num-traj-samples 5 \
            --temperature 0.6 \
            --top-p 0.98 \
            --output-dir "${OUTPUT_DIR}" \
            --num-workers 8 \
            --prefetch-factor 3 \
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
