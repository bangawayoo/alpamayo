  1. Serial KV Cache Extraction (Major Bottleneck)
  In src/alpamayo_r1/training/sft_trainer.py, the Phase A: Extract KV caches step for the action expert runs a sequential loop over the training batch.

   * Current State: It processes each sample one-by-one, running a teacher-forced VLM forward pass for each clip_id. If your batch size is 16, you are essentially doing 16 sequential
     prefill operations.
   * Optimization: Batch these forward passes by left-padding the input_ids and completion_prefix.
   * Expected Gain: 8x–16x speedup in the expert training phase (Phase A), significantly reducing the time the VLM occupies the GPU without performing actual gradient updates.

  2. Lack of Overlap between Rollout and Training
  The SelfPlayLoop in src/alpamayo_r1/training/selfplay_loop.py operates in a strictly synchronous fashion: ROLLOUT -> EVALUATE -> TRAIN.

   * Current State: The GPU is fully occupied by the RolloutEngine during Phase 1. Then, it sits idle during Phase 2 (Reward computation/binarization) while the CPU processes rewards.
     Then it switches back to training.
   * Optimization: Implement a "double-buffer" or asynchronous rollout. While the GPU is training on Iteration $N$, the CPU/other-ranks can be preparing/prefetching or even running
     rewards for the samples of Iteration $N+1$.
   * Expected Gain: Reduction in total wall time by 15–25%, as the "reward bubble" between rollout and training is eliminated.

  3. Redundant VLM Prefills for Expert Training
  The action expert requires the VLM's KV cache as conditioning.

   * Current State: Currently, the RolloutEngine runs a forward pass to generate data. Then, during compute_cfm_loss, the SFTTrainer runs another teacher-forced forward pass to get the
     exact same KV cache for the same completion.
   * Optimization: Stash the KV caches (or at least the hidden states) during the Rollout phase or the Reward extraction phase. If memory is tight, compute them once during the VLM's own
     training forward pass and pass them to the expert.
   * Expected Gain: Eliminates the need for a second VLM forward pass during expert training, effectively halving the VLM's compute overhead during the expert finetuning steps.

  4. GPU-CPU Transfer Overhead — **DONE** (PR #2: `perf/remove-per-step-gc-collect`)
  The code contained commented-out logic for moving the VLM to CPU and Expert to GPU.

   * Fix: Removed per-step `gc.collect()` + `cuda.empty_cache()`, guarded `expert.to(device)` behind device check, deleted dead offload code.
   * Result: Both models stay co-resident on GPU. No per-step overhead.

  5. Dataloader/Collator Bottlenecks
  In src/alpamayo_r1/training/sft_trainer.py, the adv_cond_collator and AdvCondDataset perform several CPU-bound operations (token concatenation, masking) on the fly.

   * Current State: While these are relatively fast, for small batches or very fast expert steps, the dataloader can become a bottleneck.
   * Optimization: Move sequence construction (advantage token insertion) into the dataset.map or pre-compute it during the "EVALUATE" phase of the self-play loop.
   * Expected Gain: Higher samples-per-second during the VLM training phase, ensuring the GPU is never waiting for the next batch.

  Summary of Impact:
  By batching the KV cache extraction (Point 1) and stashing/reusing hidden states (Point 3), you could likely reduce the wall-clock time of the expert finetuning step by over 90%, which
  is currently the slowest part of your overfit experiment.

-----
# Value Head
  1. Serial Value Head Inference (Major Bottleneck)
  In src/alpamayo_r1/training/advantage_conditioning.py, compute_segment_advantages_from_rollouts iterates through each completion and performs two sequential value head forward passes
  per sample.

   * Current State: For a batch of 128 completions, you're running 128 value_head(h_obs) and 128 value_head(h_traj) calls sequentially.
   * Optimization: Batch all h_obs and h_traj tokens across the entire rollout set into two large batched forward passes.
   * Expected Gain: 10x–50x speedup in advantage computation, as the MLP forward pass is much more efficient on a single large batch than many tiny ones.

  2. Redundant Data Transfers
  The hidden states (h_obs, h_traj) are being moved from the CPU (where they were stashed) to the GPU one sample at a time within the same serial loop.

   * Current State: h_obs = seg["h_obs"].to(vh_device) is called inside the for loop.
   * Optimization: Concatenate all hidden states on the CPU first, then perform a single bulk transfer to the GPU.
   * Expected Gain: Significant reduction in PCIe overhead and synchronization latency, especially for large rollout sets.

  3. Redundant Hidden State Extraction — **DONE** (`eee701f`)
  The EVALUATE phase ran a teacher-forced VLM forward pass specifically to extract hidden states for the value head.

   * Fix: Stash segment hidden states during rollout via `stash_segment_hidden_in_results()`, called inside `_generate_batch_vlm_only` and `_generate_batch_expert` where model inputs are already on GPU. Phase 2 reads from the stash instead of re-running TF forwards.
   * Result: Phase 2 Stage 2 dropped from 45s → 0s (80 completions). Phase 2 total: 52s → 6s (8.4x speedup).
   * Benchmark: `scripts/benchmark/profile_eval_phase.sh`, results in `scripts/benchmark/results/`.

  4. CPU-Bound Reward Logic
  Reward functions like trajectory_quality_reward and consistency_reward currently run on the CPU using NumPy.

   * Current State: The loop over completions for reward scoring can become a "bubble" where the GPU sits idle waiting for the CPU to finish L2 distance calculations and regex-based CoC
     checks.
   * Optimization: Vectorize the trajectory rewards (L2 ADE/FDE) using PyTorch on the GPU.
   * Expected Gain: Reduced "reward bubble" time between Rollout and Training phases.