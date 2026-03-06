# AlpamayoGRPOTrainer: db31d0e → HEAD Diff Analysis

## Architecture Change

### Old (`db31d0e`)

- Completions = CoC text tokens **only**
- Trajectory prediction came from **Expert + Diffusion** (`sample_trajectories_from_data_with_vlm_rollout`)
- `prompt_ids` = raw unfused tokenized input
- Reward input was decoupled: CoC text optimized by GRPO, trajectory from the deterministic Expert pipeline

### New (HEAD)

- Completions = CoC text + `<|traj_future_start|>` + **64 trajectory tokens** + `<|traj_future_end|>`
- Trajectory prediction comes from VLM output tokens decoded via `traj_tokenizer.decode()`
- `prompt_ids` = input with history trajectory tokens **fused in** via `fuse_traj_tokens`
- GRPO jointly optimizes reasoning text AND trajectory token generation

---

## Potential Bugs That Could Cause Malfunctioning Outputs

### 1. `num_return_sequences` + multimodal pixel_values (HF path — most likely)

In `_generate_single_turn` (`rollout.py:654`):

```python
vlm_output = self.full_model.vlm.generate(
    input_ids=input_ids,          # shape: (1, L)
    generation_config=gen_config, # num_return_sequences=4
    **tokenized,                  # pixel_values: (N_patches, channels) — for batch_size=1
)
```

HF's `generate()` internally expands `input_ids` from `(1, L)` → `(4, L)` for
`num_return_sequences=4`. It does **not** automatically expand `pixel_values` for Qwen3-VL. On
the first forward pass (which processes the full prompt to build the KV cache), the model receives
`input_ids=(4, L)` but `pixel_values` sized for batch_size=1. This causes Qwen3-VL to assign
images to only a fraction of the 4 sequences — the remaining sequences get incorrect/zero visual
embeddings and effectively generate **without image context**.

This would explain malfunctioning: generations look plausible (the model has strong text priors)
but don't reflect the actual driving scene.

**Fix**: Either loop `num_generations` times with `num_return_sequences=1`, or manually expand
`pixel_values` and `image_grid_thw` by `num_return_sequences` before calling `generate()`.

### 2. `extract_traj_tokens` silently produces zeros on truncation

If the VLM doesn't generate a complete trajectory token sequence (e.g., truncated at
`max_new_tokens`, or `<|traj_future_start|>` never emitted), `extract_traj_tokens` logs a warning
but returns a **zero tensor** (`token_utils.py:58-60`). These zeros decode to a default trajectory
that could look reasonable (e.g., driving straight) but is meaningless. You'd see
`trajectory_quality_reward` scores that are suspiciously uniform across samples.

Watch training logs for:
```
Batch N: Number of tokens is not equal to the expected number. Expected: 64, Got: 0.
```

### 3. `fuse_traj_tokens` / stale `tokenized` dict (not a bug, but confusing)

In `_generate_single_turn` (`rollout.py:631–636`):

```python
tokenized = model_inputs["tokenized_data"]
input_ids = tokenized.pop("input_ids")          # removes from dict
input_ids = self.full_model.fuse_traj_tokens(input_ids, traj_data)
prompt_input_ids = input_ids.clone()
```

`_compute_batch_logprobs` later uses `prompt_input_ids` (correctly fused) but
`tokenized["attention_mask"]` from the pre-fused tokenization. Since `fuse_traj_tokens` only
replaces token values without changing sequence length, the mask remains valid — not a bug, but
easy to mistake for one. The cache is safe because `helper.to_device` creates new dicts.

### 4. vLLM path: logprob format inconsistency

In `make_vllm_rollout_func` (colocate mode, `rollout.py:376-391`), `all_logprobs[i]` is
`list[list[float]]` (singleton lists per token: `[[lp0], [lp1], ...]`). This matches vLLM's
native logprob format. If TRL's internal processing of `rollout_func` results expects the flat
`list[float]` format (same as the HF `_generate_single_turn` return), it would silently misread
logprobs — treating `[lp0]` as a single-element list instead of a scalar logprob per token.

### 5. Config regressions

| Config | db31d0e | HEAD | Impact |
|---|---|---|---|
| `num_generations` | 8 | 4 | Fewer rollout samples → higher variance advantage estimates |
| `lora.r` | 16 | 4 | Reduced VLM capacity — harder task (generating traj tokens) needs more capacity, not less |
| `gradient_accumulation_steps` | 16 | 1 | Effective batch size drops 16x |
| `consistency_weight` | 0.25 | 0.0 | No signal enforcing CoC/trajectory agreement |
| `reasoning_weight` | 0.25 | 0.5 | Doubled reasoning weight without consistency counterpart |

LoRA rank halved twice (16→4) is particularly risky: generating 64 discrete trajectory tokens
autoregressively is a much harder task than just generating CoC text. The old architecture
offloaded trajectory generation entirely to Expert + Diffusion.

---

## Other Notable Changes

### `ClipDataCache` added
Data is now cached in CPU RAM and moved to device per call. Avoids repeated disk I/O for the same
clip across `num_generations` rollouts. Safe — `helper.to_device` creates new dicts each call, so
`tokenized.pop()` doesn't corrupt the cache.

### `_calculate_rewards` override (vLLM path only)
New override decodes trajectory tokens from `completion_ids` in the vLLM path. For the HF path,
`pred_xyz` is already set by `_generate_single_turn` and passes through unchanged.

### FSDP `summon_full_params(recurse=False)`
`rollout.py:608` uses `recurse=False` which only unshards the root FSDP module. Sufficient because
`unwrap_model_for_generation` also handles unsharding via accelerate.

### vLLM Qwen3-VL monkey-patch
`_patch_vllm_qwen3vl_embed()` fixes a vLLM crash where Qwen3-VL's `get_image_features` returns a
`(image_embeds, deepstack_image_embeds)` tuple instead of a single tensor. The generic vLLM
`MultiModalMixin.embed_multimodal` doesn't handle tuples, crashing during profiling.
