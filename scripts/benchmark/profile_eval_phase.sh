#!/bin/bash
# Profile the evaluate phase (Phase 2) of the SFT selfplay loop.
#
# Runs a minimal SFT iteration with 20 scenes × 4 completions to measure
# per-stage latency in the evaluate phase:
#   Stage 1: compute_rewards (CPU, negligible)
#   Stage 2: extract_segment_hidden (GPU, dominant — sequential VLM TF forwards)
#   Stage 3: compute_advantages (GPU, value head forward)
#   Stage 4: train_value_head (GPU, small MLP)
#
# Usage:
#   bash scripts/benchmark/profile_eval_phase.sh [GPU_ID]
#
# Results are logged to outputs/profile-eval-phase/train_rank0.log
# Look for "Stage N" lines for per-stage timings.

GPU_ID="${1:-1}"
OUTPUT_DIR="outputs/profile-eval-phase"

CUDA_VISIBLE_DEVICES=$GPU_ID python -m alpamayo_r1.training.train_sft \
    --config-name sft_default \
    data.max_samples=20 \
    training.num_train_epochs=1 \
    training.output_dir=$OUTPUT_DIR \
    advantage_conditioning.num_iterations=1 \
    advantage_conditioning.completions_per_scene=4 \
    advantage_conditioning.augment_with_gt=false \
    rollout.mode=vlm_only \
    rollout.scene_batch_size=1 \
    value_head.load_path=null \
    value_head.pretrain_scenes=0 \
    expert_finetune.enabled=false

echo ""
echo "=== Phase 2 Stage Timings ==="
grep -E "Stage [0-9]|Phase [0-9]|Evaluate phase" "$OUTPUT_DIR/train_rank0.log"
