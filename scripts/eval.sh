export CUDA_VISIBLE_DEVICES=0


MODEL_DIR="grpo_smoke/checkpoint-1/"
OUTPUT_DIR="grpo-debug"

bash scripts/evaluate/evaluate_quick_test.sh \
    --output-dir eval_results/$OUTPUT_DIR \
    --num-trials 1 \
    --num-traj-samples 5 \
    --temperature 0.6 \
    --adv-mode text \
    --model-name outputs/$MODEL_DIR 

    # --adv-obs \
    # --adv-traj 



    
# --iteration-dir outputs/sft-overfit-3 \
# --iteration 0