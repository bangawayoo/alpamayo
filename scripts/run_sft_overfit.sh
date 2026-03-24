export CUDA_VISIBLE_DEVICES=0,1
OUTPUT_DIR="outputs/test/gt-augment"

bash ./scripts/run_sft.sh \
    --no-fsdp \
    data.max_samples=10 \
    training.num_train_epochs=1 \
    advantage_conditioning.num_iterations=1 \
    advantage_conditioning.completions_per_scene=4 \
    training.output_dir=$OUTPUT_DIR \
    value_head.pretrain_scenes=0 \
    training.learning_rate=5e-4 \
    rollout.use_artificial_data=false \
    advantage_conditioning.p_drop=0.3 \
    advantage_conditioning.enabled=true \
    advantage_conditioning.adv_mode=text \
    advantage_conditioning.augment_with_gt=true \
    expert_finetune.lr=1e-3 \
    expert_finetune.num_noisy_samples=2 \
    training.per_device_train_batch_size=2 
