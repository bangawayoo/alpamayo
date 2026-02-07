#!/bin/bash
# Quick test evaluation on 10 samples (optimized with parallel data loading)

echo "Running quick test evaluation on 10 samples..."
python src/alpamayo_r1/evaluate_test_set.py \
    --num-samples 10 \
    --num-traj-samples 5 \
    --temperature 0.6 \
    --top-p 0.98 \
    --output-dir evaluation_results/quick_test \
    --use-clip-ids-file \
    --num-workers 4 \
    --prefetch-factor 2 \
    --seed 42
