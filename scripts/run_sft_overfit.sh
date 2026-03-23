export CUDA_VISIBLE_DEVICES=0
OUTPUT_DIR="outputs/sft-overfit-3"

bash ./scripts/run_sft.sh \
    --no-fsdp \
    data.max_samples=2 \
    training.num_train_epochs=20 \
    advantage_conditioning.num_iterations=1 \
    advantage_conditioning.completions_per_scene=1 \
    training.output_dir=$OUTPUT_DIR \
    value_head.pretrain_scenes=0 \
    training.learning_rate=5e-3 \
    rollout.use_artificial_data=true \
    advantage_conditioning.p_drop=1 \
    advantage_conditioning.enabled=true