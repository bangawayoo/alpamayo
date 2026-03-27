# Experiment: KV Cache Reuse During Training Forward (Optimization #3)

## Date: 2025-03-25

## Objective
Eliminate the redundant VLM forward pass during expert CFM training by capturing the KV cache during the training forward pass (`compute_loss`) and reusing it in `_expert_cfm_step`, skipping the serial per-sample VLM extraction (Phase A).

## Setup
- **GPU**: NVIDIA A100-SXM4-80GB (single GPU, no FSDP)
- **Model**: Qwen3-VL 10B + Expert Transformer + LoRA (r=16, alpha=32)
- **Dataset**: 1030 samples (30 fresh scenes after pretrain exclusion, artificial rollout data)
- **Training**: 78 steps, 1 epoch, batch_size=1, gradient_checkpointing=true (baseline) / disabled during KV capture (new)
- **Expert**: every_n_steps=1, num_noisy_samples=2

## Implementation
When `reuse_training_kv=true` and an expert step is due:
1. `compute_loss()` disables gradient checkpointing, runs VLM forward with `use_cache=True`
2. Per-sample KV caches are cropped to the `traj_future_start` boundary and stashed
3. `_expert_cfm_step()` skips Phase A (serial extraction) and uses stashed caches directly
4. Gradient checkpointing is re-enabled in a `finally` block

## Results

### Correctness (KV Equivalence)
- Steps 1-2: **PASS** — max_diff keys=0.000000, values=0.000000 (36 layers, prefill=3031)
- Steps 3+: Divergence due to LoRA dropout randomness (stochastic dropout masks differ between two independent forward passes). Not a correctness issue — stashed KV is the correct one for that training step.

### Per-Step Timing (steady state, batch_size=1)

| Metric        | Baseline (serial extraction) | KV Reuse (stashed) | Speedup    |
|---------------|-----------------------------|--------------------|------------|
| fwd+bwd       | 1.25s                       | 1.01s              | 1.24x      |
| expert step   | 0.78s                       | 0.27s              | 2.89x      |
| **total/step**| **2.03s**                   | **1.28s**           | **1.59x**  |

### Memory

| Metric     | Baseline | KV Reuse | Delta       |
|------------|----------|----------|-------------|
| peak (bs=1)| 29.5 GB  | 51.5 GB  | +22.0 GB    |
| peak (bs=2)| ~40 GB*  | 80.4 GB  | OOM on A100 |

*Baseline bs=2 not measured directly; estimated from bs=1 scaling.

### Observations
1. **37% faster per training step** at batch_size=1
2. **Expert step 2.9x faster** — serial VLM forward elimination is the main win
3. **fwd+bwd itself 19% faster** — gradient checkpointing disabled means no recomputation during backward (memory-compute tradeoff)
4. **+22 GB memory** — gradient checkpointing disabled stores all intermediate activations + KV cache
5. **batch_size=2 OOMs** on A100-80GB (80.4 GB peak) — the activation memory without gradient checkpointing is too large

## Conclusion
The optimization is correct and provides significant speedup, but the memory overhead from disabling gradient checkpointing limits it to batch_size=1 on 80GB GPUs. FSDP (multi-GPU sharding) could reduce per-GPU memory enough to support batch_size=2.
