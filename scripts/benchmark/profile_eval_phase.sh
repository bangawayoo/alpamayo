#!/bin/bash
# Profile the evaluate phase (Phase 2) of the SFT selfplay loop.
#
# Measures per-stage latency with segment hidden stashing enabled
# (TF forwards happen during rollout, Phase 2 reads from stash).
#
# Usage:
#   bash scripts/benchmark/profile_eval_phase.sh [GPU_ID] [NUM_SCENES]
#
# Results are logged to the output dir's train_rank0.log.
# Look for "Phase" and "Stage" lines for timings.

set -e

GPU_ID="${1:-1}"
NUM_SCENES="${2:-20}"
OUTPUT_DIR="outputs/benchmark-eval-phase"

echo "=== Evaluate Phase Benchmark ==="
echo "GPU: $GPU_ID, Scenes: $NUM_SCENES, Completions/scene: 4"
echo "Output: $OUTPUT_DIR"
echo ""

PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m alpamayo_r1.training.train_sft \
    --config-name sft_default \
    data.max_samples=$NUM_SCENES \
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
echo "=== Phase Timings ==="
grep -E "Phase [0-9]" "$OUTPUT_DIR/train_rank0.log"
echo ""
echo "=== Phase 2 Stage Breakdown ==="
grep -E "Stage [0-9]" "$OUTPUT_DIR/train_rank0.log"
echo ""
echo "=== Rollout Per-Scene Timings ==="
grep -E "stash_hidden|Rollout complete" "$OUTPUT_DIR/train_rank0.log"
