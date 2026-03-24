export CUDA_VISIBLE_DEVICES=0
OUTPUT_DIR="outputs/sft-overfit/text-adv"

bash ./scripts/run_sft.sh \
    --no-fsdp \
    data.max_samples=4 \
    training.num_train_epochs=50 \
    advantage_conditioning.num_iterations=1 \
    advantage_conditioning.completions_per_scene=1 \
    training.output_dir=$OUTPUT_DIR \
    value_head.pretrain_scenes=0 \
    training.learning_rate=5e-4 \
    rollout.use_artificial_data=true \
    advantage_conditioning.p_drop=0.3 \
    advantage_conditioning.enabled=true \
    advantage_conditioning.adv_mode=text \
    expert_finetune.lr=1e-3 \
    expert_finetune.num_noisy_samples=2 \
    training.per_device_train_batch_size=4
