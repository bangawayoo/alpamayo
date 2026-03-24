export CUDA_VISIBLE_DEVICES=1


MODEL_DIR="sft-overfit/text-adv"
OUTPUT_DIR="sft-overfit/text-adv"

bash scripts/evaluate/evaluate_quick_test.sh \
    --output-dir eval_results/$OUTPUT_DIR \
    --num-trials 1 \
    --clip-ids outputs/$MODEL_DIR/iter_0/rollout_clip_ids.json \
    --num-traj-samples 1 \
    --visualize \
    --temperature 0 \
    --adv-obs \
    --adv-mode text \
    --model-name outputs/$MODEL_DIR/iter_0/final/full_model 


    
# --iteration-dir outputs/sft-overfit-3 \
# --iteration 0