export CUDA_VISIBLE_DEVICES=0

bash scripts/evaluate/evaluate_quick_test.sh \
    --output-dir eval_results/sft-overfit \
    --adv-obs \
    --num-trials 1 \
    --clip-ids outputs/sft-overfit-3/iter_0/rollout_clip_ids.json \
    --num-traj-samples 1 \
    --visualize \
    --temperature 0 \
    --model-name outputs/sft-overfit-3/iter_0/final/full_model

    # --iteration-dir outputs/sft-overfit-3 \
# --iteration 0