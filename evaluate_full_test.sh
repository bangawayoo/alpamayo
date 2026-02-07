#!/bin/bash
# Full test set evaluation

echo "Running full test set evaluation..."
echo "Note: This will evaluate ALL 61,599 test samples and will take a long time!"
echo "Estimated time: ~17-34 hours on A100 (assuming 1-2s per sample)"
echo ""
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Evaluation cancelled."
    exit 1
fi

python src/alpamayo_r1/evaluate_test_set.py \
    --num-traj-samples 5 \
    --temperature 0.6 \
    --top-p 0.98 \
    --output-dir evaluation_results/curated_set \
    --use-clip-ids-file \
    --num-workers 8 \
    --prefetch-factor 3 \
    --seed 42
