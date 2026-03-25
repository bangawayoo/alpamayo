"""Rollout engine for the SFT pipeline, decoupled from GRPOTrainer.

Generates completions from the current policy and extracts segment-level
hidden states for value head scoring. Supports two rollout modes:
- "vlm_only": VLM generates both CoC text and discrete trajectory tokens
- "expert": VLM generates CoC text, action expert produces trajectories via diffusion

Adapted from AlpamayoGRPOTrainer._generate_single_turn() in rollout.py.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import einops
import torch
import torch.nn.functional as F
from alpamayo_r1.inference import decode_vlm_trajectories, generate_coc, prepare_vlm_inputs
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.training.rollout_utils import ClipDataCache

logger = logging.getLogger(__name__)


def run_batched_expert_diffusion(
    model: AlpamayoR1,
    tf_input_ids: torch.Tensor,
    tf_attention_mask: torch.Tensor,
    full_lens: list[int],
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    adv_traj_token_id: int | list[int] | None = None,
    expert_non_causal: bool = True,
    diffusion_steps: int | None = None,
) -> torch.Tensor:
    """Run batched teacher-forced VLM forward + optional adv_traj injection + expert diffusion.

    Shared implementation used by ``RolloutEngine._generate_batch_expert`` (training)
    and ``evaluate_test_set._evaluate_expert_with_adv_traj`` (evaluation).

    Takes right-padded teacher-forced input sequences, runs a single batched VLM
    forward to build the KV cache, optionally injects adv_traj conditioning tokens,
    then runs batched expert diffusion sampling.

    Args:
        model: AlpamayoR1 instance with VLM, expert, action projections on the
            appropriate devices.
        tf_input_ids: Right-padded token sequences, ``(n_valid, max_full_len)``.
        tf_attention_mask: Attention mask for tf_input_ids, ``(n_valid, max_full_len)``.
        full_lens: Actual (unpadded) sequence length for each sample.
        pixel_values: Concatenated pixel patches for all samples.
        image_grid_thw: Image grid dimensions for all samples.
        adv_traj_token_id: If set, inject this advantage conditioning token
            into the KV cache after teacher-forcing.
        expert_non_causal: Whether to use non-causal attention in expert.
        diffusion_steps: Override default diffusion step count (default 10).

    Returns:
        Diffusion-sampled actions, ``(n_valid, *action_dims)``.
    """
    device = tf_input_ids.device
    n_valid = tf_input_ids.shape[0]
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]  # 64
    vlm = model.vlm

    tf_kwargs: dict[str, torch.Tensor] = {"attention_mask": tf_attention_mask}
    if pixel_values is not None:
        tf_kwargs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        tf_kwargs["image_grid_thw"] = image_grid_thw

    # Disable gradient checkpointing for TF forward + adv injection
    gc_modules = [m for m in vlm.modules() if getattr(m, "gradient_checkpointing", False)]
    for m in gc_modules:
        m.gradient_checkpointing = False

    try:
        # Batched teacher-forced VLM forward
        with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
            tf_out = vlm(
                input_ids=tf_input_ids,
                use_cache=True,
                **tf_kwargs,
            )

        prompt_cache = tf_out.past_key_values
        # Navigate through PeftModel wrapper for rope_deltas
        _inner = vlm.model
        if not hasattr(_inner, "rope_deltas"):
            _inner = _inner.model
        rope_deltas = _inner.rope_deltas  # (n_valid,)
        prefill_seq_len = prompt_cache.get_seq_length()

        # Per-sample offsets = actual sequence lengths
        offsets = torch.tensor(full_lens, device=device)  # (n_valid,)

        # Inject adv_traj into KV cache (optional)
        if adv_traj_token_id is not None:
            traj_ids = (
                [adv_traj_token_id]
                if isinstance(adv_traj_token_id, int)
                else list(adv_traj_token_id)
            )
            adv_tensor = torch.tensor(
                [traj_ids] * n_valid, device=device, dtype=torch.long
            )  # (n_valid, n_traj_tokens)
            with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
                adv_out = vlm(
                    input_ids=adv_tensor,
                    past_key_values=prompt_cache,
                    use_cache=True,
                )
            prompt_cache = adv_out.past_key_values
            prefill_seq_len = prompt_cache.get_seq_length()
            offsets = offsets + len(traj_ids)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Build expert position_ids and attention_mask
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=n_valid).clone()
        delta = rope_deltas + offsets[:, None]  # (n_valid, 1)
        position_ids += delta.to(position_ids.device)

        expert_attn_mask = torch.zeros(
            (n_valid, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
            dtype=torch.float32,
            device=device,
        )
        min_val = torch.finfo(expert_attn_mask.dtype).min
        for i in range(n_valid):
            # Mask padding region in KV cache (between real prefix and diffusion tokens)
            expert_attn_mask[i, :, :, offsets[i] : -n_diffusion_tokens] = min_val

        forward_kwargs = {}
        if expert_non_causal:
            forward_kwargs["is_causal"] = False

        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            future_token_embeds = model.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b, n_diffusion_tokens, -1)
            expert_out = model.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,  # noqa: F821
                attention_mask=expert_attn_mask,
                use_cache=True,
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_seq_len)  # noqa: F821
            last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
            return model.action_out_proj(last_hidden).view(
                -1, *model.action_space.get_action_space_dims()
            )

        diff_kwargs = {}
        if diffusion_steps is not None and diffusion_steps != 10:
            diff_kwargs["num_steps"] = diffusion_steps

        with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
            sampled_action = model.diffusion.sample(
                batch_size=n_valid,
                step_fn=step_fn,
                device=device,
                return_all_steps=False,
                **diff_kwargs,
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        del prompt_cache
        return sampled_action

    finally:
        for m in gc_modules:
            m.gradient_checkpointing = True


class RolloutEngine:
    """Generate completions and extract segment hidden states for SFT.

    Decoupled from GRPOTrainer — can be used standalone or by SelfPlayLoop.
    Supports HuggingFace generation and optional vLLM server mode.

    Args:
        full_model: AlpamayoR1 instance (VLM on GPU, expert/diffusion on CPU).
        processor: HuggingFace processor with tokenizer.
        data_cache: ClipDataCache for lazy-loading driving data.
        rollout_cfg: Dict with temperature, top_p, max_generation_length, etc.
        device: CUDA device for generation.
    """

    def __init__(
        self,
        full_model: AlpamayoR1,
        processor: Any,
        data_cache: ClipDataCache,
        rollout_cfg: dict,
        device: torch.device | None = None,
        adv_token_ids: dict[str, int] | None = None,
    ) -> None:
        self.full_model = full_model
        self.processor = processor
        self.data_cache = data_cache
        self.rollout_cfg = rollout_cfg
        self.adv_token_ids = adv_token_ids or {}
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            # Respect accelerate/torchrun rank assignment
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        # Config
        self.temperature = float(rollout_cfg.get("temperature", 1.2))
        self.top_p = float(rollout_cfg.get("top_p", 0.98))
        self.max_generation_length = int(rollout_cfg.get("max_generation_length", 256))
        self.logprob_mini_batch_size = int(rollout_cfg.get("logprob_mini_batch_size", 2))

        # Rollout mode: "vlm_only" or "expert"
        self.mode = str(rollout_cfg.get("mode", "vlm_only"))
        self.expert_diffusion_steps = int(rollout_cfg.get("expert_diffusion_steps", 10))
        self.expert_non_causal = bool(rollout_cfg.get("expert_non_causal", True))
        self.use_adv_conditioning = bool(rollout_cfg.get("use_adv_conditioning", False))
        self.scene_batch_size = int(rollout_cfg.get("scene_batch_size", 1))
        self.use_artificial_data = bool(rollout_cfg.get("use_artificial_data", False))

        # Model constants
        self.traj_token_start_idx = full_model.future_token_start_idx
        self.tokens_per_future_traj = full_model.config.tokens_per_future_traj
        self.traj_vocab_size = full_model.config.traj_vocab_size
        self.special_token_ids = full_model.special_token_ids
        self.traj_tokenizer = full_model.traj_tokenizer
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.tokenizer = full_model.tokenizer

        # Expert component references (kept on CPU by default, moved to GPU per-scene)
        self.expert = full_model.expert
        self.action_in_proj = full_model.action_in_proj
        self.action_out_proj = full_model.action_out_proj
        self.action_space = full_model.action_space
        self.diffusion = full_model.diffusion

    def generate_completions(
        self,
        clip_ids: list[str],
        t0_us: int,
        G: int,
    ) -> list[dict]:
        """Generate G completions per scene.

        When running with multiple GPUs (via accelerate/torchrun), each rank
        processes a disjoint shard of scenes. Results from all ranks should be
        gathered by the caller if needed.

        Args:
            clip_ids: List of clip IDs for fresh scenes.
            t0_us: Timestamp in microseconds.
            G: Number of completions per scene.

        Returns:
            List of dicts per completion with:
            {prompt_ids, completion_ids, pred_xyz, gt_xyz, coc_text,
             clip_id, t0_us, completion_prefix, hist_xyz, hist_rot}
        """
        traj_future_end_id = self.special_token_ids["traj_future_end"]
        traj_future_start_id = self.special_token_ids["traj_future_start"]
        device = self.device
        results = []

        # Shard scenes across ranks for parallel generation
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_clip_ids = clip_ids[rank::world_size]
        if world_size > 1:
            logger.info(
                "Rank %d/%d: generating for %d/%d scenes",
                rank,
                world_size,
                len(local_clip_ids),
                len(clip_ids),
            )

        # Artificial data mode: skip real generation entirely
        if self.use_artificial_data:
            logger.warning("ARTIFICIAL DATA MODE: replacing all rollouts with fixed data")
            for clip_id in local_clip_ids:
                try:
                    scene_results = self._make_artificial_results(
                        clip_id, t0_us, G, device, traj_future_end_id, traj_future_start_id
                    )
                    results.extend(scene_results)
                except Exception as e:
                    logger.warning("Artificial data failed for %s: %s", clip_id, e)
            return results

        # Dispatch generation method based on mode
        from tqdm import tqdm

        use_expert = self.mode == "expert"
        batch_fn = self._generate_batch_expert if use_expert else self._generate_batch_vlm_only
        S = self.scene_batch_size
        logger.info("Rollout mode: %s, scene_batch_size: %d", self.mode, S)

        # Ensure VLM is on the target device for generation
        self.full_model.vlm.to(device)
        self.full_model.vlm.eval()
        n_local = len(local_clip_ids)
        n_failed = 0

        if use_expert:
            self._move_expert_to_device(device)

        # Pre-fetch first chunk so it's ready when the loop starts
        for cid in local_clip_ids[:S]:
            self.data_cache.prefetch(cid, t0_us)

        rollout_start = time.time()
        pbar = tqdm(
            total=n_local,
            desc=f"Rollout (rank {rank})",
            unit="scene",
            disable=rank != 0,
        )
        with torch.no_grad():
            for chunk_start in range(0, n_local, S):
                chunk_ids = local_clip_ids[chunk_start : chunk_start + S]
                chunk_end = chunk_start + len(chunk_ids)

                # Prefetch next chunk
                for cid in local_clip_ids[chunk_end : chunk_end + S]:
                    self.data_cache.prefetch(cid, t0_us)

                try:
                    chunk_results = batch_fn(
                        chunk_ids,
                        t0_us,
                        G,
                        device,
                        traj_future_end_id,
                        traj_future_start_id,
                    )
                    results.extend(chunk_results)
                except Exception as e:
                    logger.warning(
                        "Batch generation failed for %d scenes, falling back to per-scene: %s",
                        len(chunk_ids),
                        e,
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    for clip_id in chunk_ids:
                        try:
                            scene_results = batch_fn(
                                [clip_id],
                                t0_us,
                                G,
                                device,
                                traj_future_end_id,
                                traj_future_start_id,
                            )
                            results.extend(scene_results)
                        except Exception as e2:
                            n_failed += 1
                            logger.warning("Generation failed for %s: %s", clip_id, e2)
                            if device.type == "cuda":
                                torch.cuda.empty_cache()

                pbar.update(len(chunk_ids))
                pbar.set_postfix(completions=len(results), failed=n_failed)

        pbar.close()

        if use_expert:
            self._move_expert_to_cpu()

        total_time = time.time() - rollout_start
        logger.info(
            "[Rank %d] Rollout complete: %d completions from %d scenes (%d failed) in %.1fs (%.2f scenes/s)",
            rank,
            len(results),
            n_local,
            n_failed,
            total_time,
            n_local / max(total_time, 0.001),
        )
        return results

    def _make_artificial_results(
        self,
        clip_id: str,
        t0_us: int,
        G: int,
        device: torch.device,
        traj_future_end_id: int,
        traj_future_start_id: int,
    ) -> list[dict]:
        """Generate artificial rollout results for overfit sanity checking.

        Produces a fixed artificial CoC text but uses the REAL trajectory from
        the dataset. This lets us verify:
        - VLM overfits to the artificial CoC text
        - Expert overfits to the real trajectory (which round-trips correctly
          through the unicycle action space, unlike synthetic sinusoids)
        """

        # Fixed CoC text → token IDs (include <cot_end> to match real VLM output format)
        # Add leading space to match standard VLM generation which typically starts with a space
        # after the <|cot_start|> trigger token.
        artificial_coc = " This is artificial CoC text for overfit testing."
        coc_token_ids = self.tokenizer.encode(artificial_coc, add_special_tokens=False)
        cot_end_id = self.special_token_ids["cot_end"]

        # Load real scene data — use real trajectory as the expert target
        model_inputs, ego_future_xyz = self.data_cache.get(clip_id, t0_us, device)
        ego_future_rot = self.data_cache.get_ego_future_rot(clip_id, t0_us)
        hist_xyz = model_inputs["ego_history_xyz"][:, -1].cpu()  # (1, T, 3)
        hist_rot = model_inputs["ego_history_rot"][:, -1].cpu()  # (1, T, 3, 3)

        # Use real GT trajectory for discrete token encoding
        pred_xyz_t = ego_future_xyz[:, 0].cpu()  # (1, T_fut, 3)
        pred_rot_t = ego_future_rot[:, 0].cpu()  # (1, T_fut, 3, 3)
        discrete_tokens = self.traj_tokenizer.encode(hist_xyz, hist_rot, pred_xyz_t, pred_rot_t)
        discrete_tokens = discrete_tokens.clamp(0, self.traj_vocab_size - 1)
        traj_token_ids = (discrete_tokens + self.traj_token_start_idx).squeeze(0).tolist()

        # Build prompt_ids (same for all G completions)
        input_ids, _ = prepare_vlm_inputs(self.full_model, model_inputs)
        prompt_ids_list = input_ids[0].cpu().tolist()

        gt_traj = ego_future_xyz[0, 0].numpy().flatten().tolist()

        completion_ids = (
            coc_token_ids
            + [cot_end_id, traj_future_start_id]
            + traj_token_ids
            + [traj_future_end_id]
        )
        completion_prefix_ids = coc_token_ids + [cot_end_id, traj_future_start_id]

        # Expert CFM GT: real trajectory from dataset
        expert_fut_xyz = pred_xyz_t[0].cpu()  # (T_fut, 3)
        expert_fut_rot = pred_rot_t[0].cpu()  # (T_fut, 3, 3)

        import numpy as np

        pred_xyz_np = pred_xyz_t.numpy()  # (1, T_fut, 3)
        results = []
        for g in range(G):
            # Add small noise to pred_xyz so rewards vary across completions,
            # preventing all advantages from collapsing to zero during binarization.
            noise = np.random.normal(0, 0.01, size=pred_xyz_np.shape).astype(np.float32)
            noisy_pred = (pred_xyz_np + noise).flatten().tolist()
            results.append(
                {
                    "prompt_ids": prompt_ids_list,
                    "completion_ids": completion_ids,
                    "pred_xyz": noisy_pred,
                    "gt_xyz": gt_traj,
                    "coc_text": artificial_coc,
                    "clip_id": clip_id,
                    "t0_us": t0_us,
                    "completion_prefix": completion_prefix_ids,
                    "hist_xyz": hist_xyz[0].cpu(),
                    "hist_rot": hist_rot[0].cpu(),
                    "expert_fut_xyz": expert_fut_xyz,
                    "expert_fut_rot": expert_fut_rot,
                }
            )

        logger.info(
            "[artificial] Scene %s: %d completions, CoC=%r, traj_tokens=%d",
            clip_id,
            G,
            artificial_coc[:50],
            len(traj_token_ids),
        )
        return results

    def _move_expert_to_device(self, device: torch.device) -> None:
        """Move expert components to the specified device."""
        self.expert.to(device)
        self.action_in_proj.to(device)
        self.action_out_proj.to(device)
        self.action_space.to(device)

    def _move_expert_to_cpu(self) -> None:
        """Move expert components back to CPU to free GPU memory."""
        self.expert.cpu()
        self.action_in_proj.cpu()
        self.action_out_proj.cpu()
        self.action_space.cpu()

    # ------------------------------------------------------------------
    # Batched generation methods (scene_batch_size > 1)
    # ------------------------------------------------------------------

    def _prepare_scene_batch(
        self,
        clip_ids: list[str],
        t0_us: int,
        device: torch.device,
    ) -> tuple[dict, list[dict]]:
        """Load S scenes, pad/stack inputs into batched tensors.

        Returns:
            batched_inputs: {input_ids, attention_mask, pixel_values, image_grid_thw}
            per_scene_meta: list of per-scene metadata dicts
        """
        scenes = []
        for clip_id in clip_ids:
            model_inputs, ego_future_xyz = self.data_cache.get(clip_id, t0_us, device)
            ego_future_rot = self.data_cache.get_ego_future_rot(clip_id, t0_us)
            input_ids, gen_kwargs = prepare_vlm_inputs(self.full_model, model_inputs)
            scenes.append(
                {
                    "input_ids": input_ids,
                    "tokenized": gen_kwargs,
                    "ego_future_xyz": ego_future_xyz,
                    "ego_future_rot": ego_future_rot,
                    "hist_xyz": model_inputs["ego_history_xyz"][:, -1],  # (1, T, 3)
                    "hist_rot": model_inputs["ego_history_rot"][:, -1],  # (1, T, 3, 3)
                    "clip_id": clip_id,
                }
            )

        S = len(scenes)
        prompt_lens = [s["input_ids"].shape[1] for s in scenes]
        max_prompt_len = max(prompt_lens)

        # Left-pad input_ids and attention_mask to max_prompt_len
        padded_input_ids = torch.full(
            (S, max_prompt_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        padded_attention_mask = torch.zeros(
            S,
            max_prompt_len,
            dtype=torch.long,
            device=device,
        )
        for i, (scene, plen) in enumerate(zip(scenes, prompt_lens)):
            offset = max_prompt_len - plen
            padded_input_ids[i, offset:] = scene["input_ids"][0]
            if "attention_mask" in scene["tokenized"]:
                padded_attention_mask[i, offset:] = scene["tokenized"]["attention_mask"][0]
            else:
                padded_attention_mask[i, offset:] = 1

        batched_inputs: dict[str, torch.Tensor] = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
        }

        # Concatenate pixel_values and image_grid_thw across scenes
        # Qwen3-VL format: pixel_values (N_patches, C), image_grid_thw (N_images, 3)
        pv_list = [s["tokenized"].get("pixel_values") for s in scenes]
        igt_list = [s["tokenized"].get("image_grid_thw") for s in scenes]

        if all(pv is not None for pv in pv_list):
            batched_inputs["pixel_values"] = torch.cat(pv_list, dim=0)
        if all(igt is not None for igt in igt_list):
            batched_inputs["image_grid_thw"] = torch.cat(igt_list, dim=0)

        per_scene_meta = []
        for i, scene in enumerate(scenes):
            gt_traj = scene["ego_future_xyz"][0, 0].numpy().flatten().tolist()
            # Expert CFM GT: real dataset trajectory (fut_xyz shape: (T, 3), fut_rot: (T, 3, 3))
            expert_fut_xyz = scene["ego_future_xyz"][0, 0].cpu()  # (T_fut, 3)
            expert_fut_rot = scene["ego_future_rot"][0, 0].cpu()  # (T_fut, 3, 3)
            per_scene_meta.append(
                {
                    "prompt_len": prompt_lens[i],
                    "prompt_ids": scene["input_ids"][0].cpu().tolist(),
                    "gt_xyz": gt_traj,
                    "clip_id": scene["clip_id"],
                    "hist_xyz": scene["hist_xyz"],  # (1, T, 3) on device
                    "hist_rot": scene["hist_rot"],  # (1, T, 3, 3) on device
                    "pixel_values": pv_list[i],
                    "image_grid_thw": igt_list[i],
                    "expert_fut_xyz": expert_fut_xyz,
                    "expert_fut_rot": expert_fut_rot,
                }
            )

        return batched_inputs, per_scene_meta

    def _expand_for_generation(
        self,
        batched_inputs: dict[str, torch.Tensor],
        per_scene_meta: list[dict],
        G: int,
    ) -> dict[str, torch.Tensor]:
        """Expand S-scene batch to S*G for generate() with num_return_sequences=1.

        HF's internal expansion via repeat_interleave on concatenated pixel_values
        interleaves individual patch *rows* G times rather than repeating each
        scene's *block* of patches G times — producing garbled vision embeddings
        for multi-scene batches. We expand manually to avoid this.

        Output ordering: [s0_g0, s0_g1, ..., s0_gG-1, s1_g0, ...] matching
        the order HF would produce with num_return_sequences=G.
        """
        expanded: dict[str, torch.Tensor] = {}

        # input_ids / attention_mask: standard repeat_interleave along batch dim
        expanded["input_ids"] = batched_inputs["input_ids"].repeat_interleave(G, dim=0)
        expanded["attention_mask"] = batched_inputs["attention_mask"].repeat_interleave(G, dim=0)

        # pixel_values: repeat each scene's patch block G times (not interleave rows)
        if "pixel_values" in batched_inputs:
            pv_parts = []
            for meta in per_scene_meta:
                pv = meta["pixel_values"]
                if pv is not None:
                    # pv shape: (N_patches, C) — repeat gives G copies of the full block
                    pv_parts.append(pv.repeat(G, *([1] * (pv.dim() - 1))))
            if pv_parts:
                expanded["pixel_values"] = torch.cat(pv_parts, dim=0)

        # image_grid_thw: repeat each scene's grid entries G times
        if "image_grid_thw" in batched_inputs:
            igt_parts = []
            for meta in per_scene_meta:
                igt = meta["image_grid_thw"]
                if igt is not None:
                    igt_parts.append(igt.repeat(G, 1))
            if igt_parts:
                expanded["image_grid_thw"] = torch.cat(igt_parts, dim=0)

        return expanded

    def _generate_batch_vlm_only(
        self,
        clip_ids: list[str],
        t0_us: int,
        G: int,
        device: torch.device,
        traj_future_end_id: int,
        traj_future_start_id: int,
    ) -> list[dict]:
        """Generate G completions for S scenes using batched VLM-only rollout."""
        S = len(clip_ids)
        timings: dict[str, float] = {}

        # 1. Prepare batched inputs
        t_ = time.time()
        batched_inputs, per_scene_meta = self._prepare_scene_batch(clip_ids, t0_us, device)
        max_prompt_len = batched_inputs["input_ids"].shape[1]
        timings["prepare_batch"] = time.time() - t_

        # 2. Batched VLM generation: S scenes × G completions
        # Manually expand to S*G with num_return_sequences=1 to avoid HF's
        # repeat_interleave garbling concatenated pixel_values across scenes.
        t_ = time.time()
        expanded = self._expand_for_generation(batched_inputs, per_scene_meta, G)
        gen_kwargs = {
            k: expanded[k]
            for k in ("attention_mask", "pixel_values", "image_grid_thw")
            if k in expanded
        }
        vlm_output = generate_coc(
            self.full_model,
            expanded["input_ids"],
            gen_kwargs,
            mode="vlm",
            temperature=self.temperature,
            top_p=self.top_p,
            num_samples=1,
            max_new_tokens=self.max_generation_length + self.tokens_per_future_traj + 10,
            pad_token_id=self.pad_token_id,
            autocast_device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings["vlm_generate"] = time.time() - t_

        # vlm_output: (S*G, max_seq_len). Generated tokens start at max_prompt_len.
        generated_seqs = vlm_output[:, max_prompt_len:]

        # 3. Batched trajectory extraction and decoding
        t_ = time.time()
        hist_xyz_list = []
        hist_rot_list = []
        for meta in per_scene_meta:
            hist_xyz_list.append(meta["hist_xyz"].expand(G, -1, -1))  # (G, T, 3)
            hist_rot_list.append(meta["hist_rot"].expand(G, -1, -1, -1))  # (G, T, 3, 3)
        hist_xyz_batch = torch.cat(hist_xyz_list, dim=0)  # (S*G, T, 3)
        hist_rot_batch = torch.cat(hist_rot_list, dim=0)  # (S*G, T, 3, 3)

        pred_xyz_tensor, pred_rot_tensor = decode_vlm_trajectories(
            self.full_model, vlm_output, hist_xyz_batch, hist_rot_batch
        )
        timings["traj_decode"] = time.time() - t_

        # 4. Build per-completion result dicts
        t_ = time.time()
        results = []
        for s_idx, meta in enumerate(per_scene_meta):
            for g_idx in range(G):
                flat_idx = s_idx * G + g_idx
                raw_completion = generated_seqs[flat_idx].cpu().tolist()

                # Trim to traj_future_end
                completion_ids: list[int] = []
                for tid in raw_completion:
                    completion_ids.append(tid)
                    if tid == traj_future_end_id:
                        break
                if traj_future_end_id not in completion_ids:
                    while completion_ids and completion_ids[-1] == self.pad_token_id:
                        completion_ids.pop()
                if not completion_ids:
                    completion_ids = [self.processor.tokenizer.eos_token_id]

                # Extract CoC text
                try:
                    traj_start_pos = raw_completion.index(traj_future_start_id)
                except ValueError:
                    traj_start_pos = len(raw_completion)
                coc_text = self.tokenizer.decode(
                    raw_completion[:traj_start_pos],
                    skip_special_tokens=True,
                ).strip()

                # Completion prefix for expert KV cache
                try:
                    prefix_end = raw_completion.index(traj_future_start_id) + 1
                except ValueError:
                    prefix_end = len(raw_completion)

                pred_traj = pred_xyz_tensor[flat_idx].cpu().numpy().flatten().tolist()

                results.append(
                    {
                        "prompt_ids": meta["prompt_ids"],
                        "completion_ids": completion_ids,
                        "pred_xyz": pred_traj,
                        "gt_xyz": meta["gt_xyz"],
                        "coc_text": coc_text,
                        "clip_id": meta["clip_id"],
                        "t0_us": t0_us,
                        "completion_prefix": raw_completion[:prefix_end],
                        "hist_xyz": meta["hist_xyz"][0].cpu(),
                        "hist_rot": meta["hist_rot"][0].cpu(),
                        "expert_fut_xyz": pred_xyz_tensor[flat_idx].cpu(),
                        "expert_fut_rot": pred_rot_tensor[flat_idx].cpu(),
                    }
                )
        timings["postprocess"] = time.time() - t_

        # 5. Stash segment hidden states (TF forward reuses cached model inputs)
        t_ = time.time()
        self.stash_segment_hidden_in_results(results, per_scene_meta)
        timings["stash_hidden"] = time.time() - t_

        logger.debug(
            "[vlm_only_batch] S=%d, G=%d (%d results) timings: %s | total=%.2fs",
            S,
            G,
            len(results),
            ", ".join(f"{k}={v:.3f}s" for k, v in timings.items()),
            sum(timings.values()),
        )
        return results

    def _generate_batch_expert(
        self,
        clip_ids: list[str],
        t0_us: int,
        G: int,
        device: torch.device,
        traj_future_end_id: int,
        traj_future_start_id: int,
    ) -> list[dict]:
        """Generate G completions for S scenes using batched expert pipeline.

        Phases:
        1. Batched VLM CoC generation (S scenes × G completions)
        2. Batched teacher-forced VLM forward (valid completions only)
        3. Batched expert diffusion sampling
        4. Batched trajectory encoding
        """
        S = len(clip_ids)
        timings: dict[str, float] = {}

        # Phase 1a: Prepare batched inputs
        t_ = time.time()
        batched_inputs, per_scene_meta = self._prepare_scene_batch(clip_ids, t0_us, device)
        max_prompt_len = batched_inputs["input_ids"].shape[1]

        # Optionally append <adv_obs_pos> to input_ids
        if self.use_adv_conditioning and "adv_obs_pos" in self.adv_token_ids:
            adv_obs_tokens = self.adv_token_ids["adv_obs_pos"]
            adv_obs_tokens = (
                [adv_obs_tokens] if isinstance(adv_obs_tokens, int) else list(adv_obs_tokens)
            )
            n_obs = len(adv_obs_tokens)
            adv_col = torch.tensor(
                [adv_obs_tokens] * S, device=device, dtype=batched_inputs["input_ids"].dtype
            )  # (S, n_obs)
            batched_inputs["input_ids"] = torch.cat([batched_inputs["input_ids"], adv_col], dim=1)
            adv_mask = torch.ones(S, n_obs, device=device, dtype=torch.long)
            batched_inputs["attention_mask"] = torch.cat(
                [batched_inputs["attention_mask"], adv_mask], dim=1
            )
            max_prompt_len = batched_inputs["input_ids"].shape[1]

        timings["prepare_batch"] = time.time() - t_

        # Phase 1b: Batched VLM CoC generation
        # Manually expand to S*G with num_return_sequences=1 to avoid HF's
        # repeat_interleave garbling concatenated pixel_values across scenes.
        t_ = time.time()
        expanded = self._expand_for_generation(batched_inputs, per_scene_meta, G)
        gen_kwargs = {
            k: expanded[k]
            for k in ("attention_mask", "pixel_values", "image_grid_thw")
            if k in expanded
        }
        vlm_output = generate_coc(
            self.full_model,
            expanded["input_ids"],
            gen_kwargs,
            mode="expert",
            temperature=self.temperature,
            top_p=self.top_p,
            num_samples=1,
            max_new_tokens=self.max_generation_length + 10,
            pad_token_id=self.pad_token_id,
            autocast_device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        generated_seqs = vlm_output[:, max_prompt_len:]
        timings["vlm_coc_generate"] = time.time() - t_

        # Parse completions and filter to valid ones (those with <traj_future_start>)
        valid_items: list[dict] = []
        for s_idx in range(S):
            meta = per_scene_meta[s_idx]
            for g_idx in range(G):
                flat_idx = s_idx * G + g_idx
                raw = generated_seqs[flat_idx].cpu().tolist()
                while raw and raw[-1] == self.pad_token_id:
                    raw.pop()
                try:
                    traj_pos = raw.index(traj_future_start_id)
                except ValueError:
                    logger.warning(
                        "No <traj_future_start> in batch output for %s sample %d, skipping",
                        meta["clip_id"],
                        g_idx,
                    )
                    continue
                coc_tokens = raw[:traj_pos]
                valid_items.append(
                    {
                        "scene_idx": s_idx,
                        "coc_tokens": coc_tokens,
                        "coc_prefix": coc_tokens + [traj_future_start_id],
                        "coc_text": self.tokenizer.decode(
                            coc_tokens,
                            skip_special_tokens=True,
                        ).strip(),
                    }
                )

        if not valid_items:
            logger.warning("No valid completions in batch of %d scenes", S)
            return []

        n_valid = len(valid_items)

        # Phase 2: Build right-padded TF sequences
        t_ = time.time()
        full_lens = []
        for item in valid_items:
            s_idx = item["scene_idx"]
            meta = per_scene_meta[s_idx]
            full_len = meta["prompt_len"] + len(item["coc_prefix"])
            full_lens.append(full_len)

        max_full_len = max(full_lens)
        tf_input_ids = torch.full(
            (n_valid, max_full_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        tf_attention_mask = torch.zeros(
            n_valid,
            max_full_len,
            dtype=torch.long,
            device=device,
        )

        for i, item in enumerate(valid_items):
            s_idx = item["scene_idx"]
            meta = per_scene_meta[s_idx]
            prompt = torch.tensor(meta["prompt_ids"], device=device, dtype=torch.long)
            prefix = torch.tensor(item["coc_prefix"], device=device, dtype=torch.long)
            actual_len = prompt.shape[0] + prefix.shape[0]
            tf_input_ids[i, : prompt.shape[0]] = prompt
            tf_input_ids[i, prompt.shape[0] : actual_len] = prefix
            tf_attention_mask[i, :actual_len] = 1

        # Replicate pixel_values and image_grid_thw for valid items
        pv_parts = []
        igt_parts = []
        for item in valid_items:
            meta = per_scene_meta[item["scene_idx"]]
            if meta["pixel_values"] is not None:
                pv_parts.append(meta["pixel_values"])
            if meta["image_grid_thw"] is not None:
                igt_parts.append(meta["image_grid_thw"])
        pv_tensor = torch.cat(pv_parts, dim=0) if pv_parts else None
        igt_tensor = torch.cat(igt_parts, dim=0) if igt_parts else None
        timings["build_tf_inputs"] = time.time() - t_

        # Phase 3: Batched TF forward + adv_traj injection + expert diffusion
        t_ = time.time()
        adv_traj_id = None
        if self.use_adv_conditioning and "adv_traj_pos" in self.adv_token_ids:
            adv_traj_id = self.adv_token_ids["adv_traj_pos"]

        sampled_action = run_batched_expert_diffusion(
            model=self.full_model,
            tf_input_ids=tf_input_ids,
            tf_attention_mask=tf_attention_mask,
            full_lens=full_lens,
            pixel_values=pv_tensor,
            image_grid_thw=igt_tensor,
            adv_traj_token_id=adv_traj_id,
            expert_non_causal=self.expert_non_causal,
            diffusion_steps=self.expert_diffusion_steps,
        )
        timings["tf_diffusion"] = time.time() - t_

        # Phase 4: Batched trajectory encoding
        t_ = time.time()
        # Build batched hist_xyz/hist_rot for valid items
        hist_xyz_list = []
        hist_rot_list = []
        for item in valid_items:
            meta = per_scene_meta[item["scene_idx"]]
            hist_xyz_list.append(meta["hist_xyz"])  # (1, T, 3)
            hist_rot_list.append(meta["hist_rot"])  # (1, T, 3, 3)
        hist_xyz_batch = torch.cat(hist_xyz_list, dim=0)  # (n_valid, T, 3)
        hist_rot_batch = torch.cat(hist_rot_list, dim=0)  # (n_valid, T, 3, 3)

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action,
            hist_xyz_batch,
            hist_rot_batch,
        )
        discrete_tokens = self.traj_tokenizer.encode(
            hist_xyz_batch,
            hist_rot_batch,
            pred_xyz,
            pred_rot,
        )
        discrete_tokens = discrete_tokens.clamp(0, self.traj_vocab_size - 1)
        # discrete_tokens: (n_valid, n_traj_tokens)

        # Build result dicts
        results = []
        for i, item in enumerate(valid_items):
            meta = per_scene_meta[item["scene_idx"]]
            traj_token_ids = (discrete_tokens[i] + self.traj_token_start_idx).tolist()
            completion_ids = (
                item["coc_tokens"] + [traj_future_start_id] + traj_token_ids + [traj_future_end_id]
            )
            pred_traj = pred_xyz[i].cpu().numpy().flatten().tolist()

            results.append(
                {
                    "prompt_ids": meta["prompt_ids"],
                    "completion_ids": completion_ids,
                    "pred_xyz": pred_traj,
                    "gt_xyz": meta["gt_xyz"],
                    "coc_text": item["coc_text"],
                    "clip_id": meta["clip_id"],
                    "t0_us": t0_us,
                    "completion_prefix": item["coc_prefix"],
                    "hist_xyz": meta["hist_xyz"][0].cpu(),
                    "hist_rot": meta["hist_rot"][0].cpu(),
                    # Use rolled-out trajectory as expert CFM target so the
                    # expert trains on the same data the advantages were computed on.
                    "expert_fut_xyz": pred_xyz[i].cpu(),
                    "expert_fut_rot": pred_rot[i].cpu(),
                }
            )
        timings["traj_encode"] = time.time() - t_

        # Stash segment hidden states (TF forward reuses cached model inputs)
        t_ = time.time()
        self.stash_segment_hidden_in_results(results, per_scene_meta)
        timings["stash_hidden"] = time.time() - t_

        logger.debug(
            "[expert_batch] S=%d, G=%d (%d valid, %d results) timings: %s | total=%.2fs",
            S,
            G,
            n_valid,
            len(results),
            ", ".join(f"{k}={v:.3f}s" for k, v in timings.items()),
            sum(timings.values()),
        )
        return results

    def extract_segment_hidden(
        self,
        rollout_results: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Extract segment-level hidden states via teacher-forced VLM forward.

        Runs a forward pass for each completion to get hidden states at
        h_obs, h_coc, and h_traj positions. Also returns the segment map
        for each completion.

        Args:
            rollout_results: List of rollout dicts with prompt_ids and completion_ids.

        Returns:
            (segment_hidden_stash, completion_segment_map) — same format as
            AlpamayoGRPOTrainer._extract_and_stash_segment_hidden output.
        """
        device = self.device
        segment_hidden_stash = []
        completion_segment_map = []

        # Group by clip_id to share model_inputs across completions from same scene
        from collections import defaultdict

        scene_groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rollout_results):
            scene_groups[r["clip_id"]].append(i)

        # Ensure VLM is on the target device
        self.full_model.vlm.to(device)
        self.full_model.vlm.eval()
        with torch.no_grad():
            for clip_id, indices in scene_groups.items():
                t0_us = rollout_results[indices[0]]["t0_us"]
                model_inputs, _ = self.data_cache.get(clip_id, t0_us, device)

                # Prepare prompt input_ids with fused history
                input_ids, _ = prepare_vlm_inputs(self.full_model, model_inputs)
                prompt_len = input_ids.shape[1]
                prompt_input_ids = input_ids.clone()

                # Process each completion from this scene
                comp_ids_list = [rollout_results[i]["completion_ids"] for i in indices]
                logprob_result = _compute_batch_logprobs(
                    self.full_model,
                    model_inputs,
                    prompt_input_ids,
                    comp_ids_list,
                    prompt_len,
                    device,
                    mini_batch_size=self.logprob_mini_batch_size,
                    output_hidden_states=True,
                )
                _, batch_hidden = logprob_result

                for local_idx, global_idx in enumerate(indices):
                    hidden_states = batch_hidden[local_idx]
                    completion_ids = rollout_results[global_idx]["completion_ids"]
                    seg_hidden, seg_map = _extract_segment_hidden(
                        hidden_states,
                        completion_ids,
                        self.special_token_ids,
                        self.traj_token_start_idx,
                        self.traj_vocab_size,
                    )
                    segment_hidden_stash.append(seg_hidden)
                    completion_segment_map.append(seg_map)

        logger.debug(
            "Extracted segment hidden states for %d completions", len(segment_hidden_stash)
        )
        return segment_hidden_stash, completion_segment_map

    def stash_segment_hidden_in_results(
        self,
        results: list[dict],
        per_scene_meta: list[dict] | None = None,
    ) -> None:
        """Extract segment hidden states and stash them in each result dict.

        Runs teacher-forced VLM forward passes and stores segment_hidden +
        segment_map directly in each result dict. This allows the evaluate
        phase to skip the expensive re-forward by reading from the stash.

        Args:
            results: List of rollout result dicts (mutated in-place).
            per_scene_meta: If provided, reuse already-prepared model inputs
                from _prepare_scene_batch (avoids re-loading from cache).
                Must have one entry per scene, with results ordered as
                [scene0_g0, scene0_g1, ..., scene1_g0, ...].
        """
        from collections import defaultdict

        device = self.device

        if per_scene_meta is not None:
            # Fast path: reuse already-prepared inputs from batch rollout
            G = len(results) // len(per_scene_meta) if per_scene_meta else 0
            for s_idx, meta in enumerate(per_scene_meta):
                # Indices into results for this scene
                scene_indices = list(range(s_idx * G, (s_idx + 1) * G))
                if not scene_indices or scene_indices[-1] >= len(results):
                    continue

                clip_id = meta["clip_id"]
                t0_us = results[scene_indices[0]].get("t0_us", 5_100_000)
                model_inputs, _ = self.data_cache.get(clip_id, t0_us, device)
                prompt_ids = torch.tensor([meta["prompt_ids"]], device=device, dtype=torch.long)
                prompt_len = len(meta["prompt_ids"])

                comp_ids_list = [results[i]["completion_ids"] for i in scene_indices]
                logprob_result = _compute_batch_logprobs(
                    self.full_model,
                    model_inputs,
                    prompt_ids,
                    comp_ids_list,
                    prompt_len,
                    device,
                    mini_batch_size=self.logprob_mini_batch_size,
                    output_hidden_states=True,
                )
                _, batch_hidden = logprob_result

                for local_idx, global_idx in enumerate(scene_indices):
                    hidden_states = batch_hidden[local_idx]
                    seg_hidden, seg_map = _extract_segment_hidden(
                        hidden_states,
                        results[global_idx]["completion_ids"],
                        self.special_token_ids,
                        self.traj_token_start_idx,
                        self.traj_vocab_size,
                    )
                    results[global_idx]["segment_hidden"] = seg_hidden
                    results[global_idx]["segment_map"] = seg_map
        else:
            # Fallback: group by clip_id and load from cache
            scene_groups: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(results):
                scene_groups[r["clip_id"]].append(i)

            for clip_id, indices in scene_groups.items():
                t0_us = results[indices[0]]["t0_us"]
                model_inputs, _ = self.data_cache.get(clip_id, t0_us, device)

                input_ids, _ = prepare_vlm_inputs(self.full_model, model_inputs)
                prompt_len = input_ids.shape[1]
                prompt_input_ids = input_ids.clone()

                comp_ids_list = [results[i]["completion_ids"] for i in indices]
                logprob_result = _compute_batch_logprobs(
                    self.full_model,
                    model_inputs,
                    prompt_input_ids,
                    comp_ids_list,
                    prompt_len,
                    device,
                    mini_batch_size=self.logprob_mini_batch_size,
                    output_hidden_states=True,
                )
                _, batch_hidden = logprob_result

                for local_idx, global_idx in enumerate(indices):
                    hidden_states = batch_hidden[local_idx]
                    seg_hidden, seg_map = _extract_segment_hidden(
                        hidden_states,
                        results[global_idx]["completion_ids"],
                        self.special_token_ids,
                        self.traj_token_start_idx,
                        self.traj_vocab_size,
                    )
                    results[global_idx]["segment_hidden"] = seg_hidden
                    results[global_idx]["segment_map"] = seg_map

    def compute_rewards(self, rollout_results: list[dict]) -> list[dict]:
        """Score completions using reward functions.

        Args:
            rollout_results: List of rollout dicts with pred_xyz, gt_xyz, coc_text.

        Returns:
            List of dicts per completion: {r_traj, r_reason, r_consist, r_traj_per_step}
        """
        from alpamayo_r1.training.rewards import (
            consistency_reward,
            reasoning_quality_reward,
            trajectory_quality_reward,
        )

        completions = [r["coc_text"] for r in rollout_results]
        pred_xyzs = [r["pred_xyz"] for r in rollout_results]
        gt_xyzs = [r["gt_xyz"] for r in rollout_results]

        r_traj = trajectory_quality_reward(completions, pred_xyzs, gt_xyzs)
        r_reason = reasoning_quality_reward(completions)
        r_consist = consistency_reward(completions, pred_xyzs)

        results = []
        for i in range(len(rollout_results)):
            # Per-timestep trajectory rewards (uniform split as fallback)
            n_traj = self.tokens_per_future_traj
            r_per_step = [r_traj[i] / max(n_traj, 1)] * n_traj if n_traj > 0 else []

            results.append(
                {
                    "r_traj": r_traj[i],
                    "r_reason": r_reason[i],
                    "r_consist": r_consist[i],
                    "r_traj_per_step": r_per_step,
                }
            )

        return results


# ---------------------------------------------------------------------------
# Standalone helpers (extracted from rollout.py methods)
# ---------------------------------------------------------------------------


def _extract_segment_hidden(
    hidden_states: torch.Tensor,
    completion_ids: list[int],
    special_token_ids: dict,
    traj_token_start_idx: int,
    traj_vocab_size: int,
) -> tuple[dict, dict]:
    """Extract hidden states at segment boundaries.

    Standalone version of AlpamayoGRPOTrainer._extract_and_stash_segment_hidden.

    Args:
        hidden_states: (1, 1+comp_len, D) from teacher-forced pass.
        completion_ids: List of completion token IDs.
        special_token_ids: Dict of special token IDs.
        traj_token_start_idx: Start index for trajectory tokens.
        traj_vocab_size: Number of trajectory token IDs.

    Returns:
        (segment_hidden, segment_map) dicts.
    """
    cot_end_id = special_token_ids["cot_end"]
    traj_future_start_id = special_token_ids["traj_future_start"]

    # h_obs: last prompt token
    h_obs = hidden_states[0, 0:1, :]  # (1, D)

    # Find <cot_end> position
    cot_end_offset = None
    for idx, tid in enumerate(completion_ids):
        if tid == cot_end_id:
            cot_end_offset = idx
            break

    if cot_end_offset is not None:
        h_coc = hidden_states[0, cot_end_offset + 1 : cot_end_offset + 2, :]
    else:
        h_coc = h_obs

    # Find trajectory token positions
    traj_positions = []
    for idx, tid in enumerate(completion_ids):
        if traj_token_start_idx <= tid < traj_token_start_idx + traj_vocab_size:
            traj_positions.append(idx)

    if traj_positions:
        traj_indices = [p + 1 for p in traj_positions]
        h_traj = hidden_states[0, traj_indices, :]
    else:
        h_traj = torch.zeros(0, hidden_states.shape[-1])

    # CoC length
    traj_start_offset = None
    for idx, tid in enumerate(completion_ids):
        if tid == traj_future_start_id:
            traj_start_offset = idx
            break
    coc_len = traj_start_offset if traj_start_offset is not None else len(completion_ids)

    segment_hidden = {"h_obs": h_obs, "h_coc": h_coc, "h_traj": h_traj}
    segment_map = {
        "coc_len": coc_len,
        "traj_len": len(traj_positions),
        "traj_positions": traj_positions,
        "total_len": len(completion_ids),
    }
    return segment_hidden, segment_map


def _compute_batch_logprobs(
    full_model: AlpamayoR1,
    model_inputs: dict,
    prompt_input_ids: torch.Tensor,
    completion_ids_list: list[list[int]],
    prompt_len: int,
    device: torch.device,
    mini_batch_size: int = 4,
    output_hidden_states: bool = False,
) -> list[list[float]] | tuple[list[list[float]], list[torch.Tensor]]:
    """Compute per-token log-probs for a batch of completions.

    Standalone version of _compute_batch_logprobs from rollout.py.
    """
    results: list[list[float]] = []
    hidden_results: list[torch.Tensor] = []
    tokenized = model_inputs["tokenized_data"]

    for batch_start in range(0, len(completion_ids_list), mini_batch_size):
        batch_comp_ids = completion_ids_list[batch_start : batch_start + mini_batch_size]
        B = len(batch_comp_ids)

        comp_tensors = [
            torch.tensor(ids, dtype=torch.long, device=device)
            if ids
            else torch.tensor([0], dtype=torch.long, device=device)
            for ids in batch_comp_ids
        ]
        comp_lens = [len(ids) for ids in batch_comp_ids]
        max_comp_len = max(t.shape[0] for t in comp_tensors)

        comp_padded = torch.zeros(B, max_comp_len, dtype=torch.long, device=device)
        for i, t in enumerate(comp_tensors):
            comp_padded[i, : t.shape[0]] = t

        prompt_expanded = prompt_input_ids.expand(B, -1)
        full_ids = torch.cat([prompt_expanded, comp_padded], dim=1)

        forward_kwargs = {}
        if "attention_mask" in tokenized:
            orig_mask = tokenized["attention_mask"]
            comp_mask = torch.zeros(B, max_comp_len, device=device, dtype=orig_mask.dtype)
            for i, comp_len in enumerate(comp_lens):
                if comp_len > 0:
                    comp_mask[i, :comp_len] = 1
            forward_kwargs["attention_mask"] = torch.cat(
                [orig_mask.expand(B, -1), comp_mask], dim=1
            )
        if "pixel_values" in tokenized:
            pv = tokenized["pixel_values"]
            forward_kwargs["pixel_values"] = pv.repeat(B, *([1] * (pv.dim() - 1)))
        if "image_grid_thw" in tokenized:
            igt = tokenized["image_grid_thw"]
            forward_kwargs["image_grid_thw"] = igt.repeat(B, 1)

        with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
            outputs = full_model.vlm(
                input_ids=full_ids,
                output_hidden_states=output_hidden_states,
                **forward_kwargs,
            )

        for i, (comp_ids, comp_len) in enumerate(zip(batch_comp_ids, comp_lens)):
            if not comp_ids:
                results.append([])
                if output_hidden_states:
                    hidden_results.append(torch.zeros(1, 0, 1))
                continue
            logits = outputs.logits[i, prompt_len - 1 : prompt_len - 1 + comp_len]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            comp_target = comp_tensors[i][:comp_len]
            token_log_probs = log_probs.gather(1, comp_target.unsqueeze(-1)).squeeze(-1)
            results.append(token_log_probs.cpu().tolist())

            if output_hidden_states:
                hs = outputs.hidden_states[-1][i, prompt_len - 1 : prompt_len + comp_len]
                hidden_results.append(hs.float().cpu().unsqueeze(0))

    if output_hidden_states:
        return results, hidden_results
    return results
