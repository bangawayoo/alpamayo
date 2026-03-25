#!/usr/bin/env bash
# Experiment: validate KV cache reuse optimization
#
# Runs a smoke SFT training with reuse_training_kv=true and validate_kv_reuse=true.
# Compares stashed KV caches against serial extraction and reports memory/latency.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/experiment_kv_reuse.sh

set -euo pipefail

echo "============================================"
echo "  KV Cache Reuse Experiment"
echo "============================================"
echo ""

bash scripts/run_sft.sh \
    --smoke \
    --no-fsdp \
    expert_finetune.reuse_training_kv=true \
    expert_finetune.validate_kv_reuse=true \
    training.per_device_train_batch_size=2 \
    training.gradient_checkpointing=true
