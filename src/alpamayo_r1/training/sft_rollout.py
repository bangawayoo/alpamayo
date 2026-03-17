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
from transformers import GenerationConfig, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1, ExpertLogitsProcessor
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_traj_tokens,
)
from alpamayo_r1.training.rollout_utils import ClipDataCache

logger = logging.getLogger(__name__)


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

        # Dispatch per-scene generation method based on mode
        use_expert = self.mode == "expert"
        generate_fn = (
            self._generate_for_scene_expert if use_expert else self._generate_for_scene_vlm_only
        )
        logger.info("Rollout mode: %s", self.mode)

        # Ensure VLM is on the target device for generation
        self.full_model.vlm.to(device)
        self.full_model.vlm.eval()
        n_local = len(local_clip_ids)
        n_failed = 0

        if use_expert:
            self._move_expert_to_device(device)

        rollout_start = time.time()
        with torch.no_grad():
            for scene_idx, clip_id in enumerate(local_clip_ids):
                if scene_idx % max(1, n_local // 10) == 0 or scene_idx == n_local - 1:
                    elapsed = time.time() - rollout_start
                    mem_alloc = torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0
                    mem_reserved = torch.cuda.memory_reserved(device) / 1e9 if device.type == "cuda" else 0
                    scenes_per_sec = (scene_idx + 1) / max(elapsed, 0.001)
                    eta = (n_local - scene_idx - 1) / max(scenes_per_sec, 0.001)
                    logger.info(
                        "[Rank %d] Rollout progress: %d/%d scenes (%d completions, %d failed) "
                        "| %.1fs elapsed, ETA %.0fs | GPU mem: %.1f/%.1f GB",
                        rank,
                        scene_idx + 1,
                        n_local,
                        len(results),
                        n_failed,
                        elapsed,
                        eta,
                        mem_alloc,
                        mem_reserved,
                    )
                try:
                    scene_start = time.time()
                    scene_results = generate_fn(
                        clip_id,
                        t0_us,
                        G,
                        device,
                        traj_future_end_id,
                        traj_future_start_id,
                    )
                    results.extend(scene_results)
                    logger.debug(
                        "Scene %s: %d completions in %.2fs",
                        clip_id,
                        len(scene_results),
                        time.time() - scene_start,
                    )
                except Exception as e:
                    n_failed += 1
                    logger.warning("Generation failed for %s: %s", clip_id, e)
                    continue

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

    def _generate_for_scene_vlm_only(
        self,
        clip_id: str,
        t0_us: int,
        G: int,
        device: torch.device,
        traj_future_end_id: int,
        traj_future_start_id: int,
    ) -> list[dict]:
        """Generate G completions for a single scene using VLM-only rollout."""
        # 1. Load driving data
        model_inputs, ego_future_xyz = self.data_cache.get(clip_id, t0_us, device)

        # 2. Fuse history trajectory tokens
        tokenized = {k: v for k, v in model_inputs["tokenized_data"].items()}
        input_ids = tokenized.pop("input_ids")
        traj_data = {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        }
        input_ids = self.full_model.fuse_traj_tokens(input_ids, traj_data)
        prompt_len = input_ids.shape[1]
        prompt_input_ids = input_ids.clone()

        # 3. VLM generation
        gen_config = GenerationConfig(
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=G,
            max_new_tokens=self.max_generation_length + self.tokens_per_future_traj + 10,
            pad_token_id=self.pad_token_id,
        )
        stopping = StoppingCriteriaList(
            [
                StopAfterEOS(eos_token_id=traj_future_end_id),
            ]
        )

        with torch.autocast(str(device), dtype=torch.bfloat16):
            vlm_output = self.full_model.vlm.generate(
                input_ids=input_ids,
                generation_config=gen_config,
                stopping_criteria=stopping,
                **tokenized,
            )

        generated_seqs = vlm_output[:, prompt_len:]

        # 4. Extract trajectory tokens and decode to continuous xyz
        traj_tokens = extract_traj_tokens(
            vlm_output,
            self.special_token_ids,
            self.tokens_per_future_traj,
            self.traj_token_start_idx,
            self.traj_vocab_size,
        )
        hist_xyz = model_inputs["ego_history_xyz"][:, -1]  # (1, T, 3)
        hist_rot = model_inputs["ego_history_rot"][:, -1]  # (1, T, 3, 3)
        hist_xyz_rep = hist_xyz.expand(G, -1, -1)
        hist_rot_rep = hist_rot.expand(G, -1, -1, -1)

        with torch.no_grad():
            pred_xyz_tensor, pred_rot_tensor, _ = self.traj_tokenizer.decode(
                hist_xyz_rep,
                hist_rot_rep,
                traj_tokens,
            )

        # 5. Build per-sample outputs
        prompt_ids_list = prompt_input_ids[0].cpu().tolist()
        results = []

        for sample_idx in range(G):
            raw_completion = generated_seqs[sample_idx].cpu().tolist()

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
                raw_completion[:traj_start_pos], skip_special_tokens=True
            ).strip()

            # Completion prefix for expert KV cache
            try:
                prefix_end = raw_completion.index(traj_future_start_id) + 1
            except ValueError:
                prefix_end = len(raw_completion)

            # Trajectory data
            pred_traj = pred_xyz_tensor[sample_idx].cpu().numpy().flatten().tolist()
            gt_traj = ego_future_xyz[0, 0].numpy().flatten().tolist()

            results.append(
                {
                    "prompt_ids": prompt_ids_list,
                    "completion_ids": completion_ids,
                    "pred_xyz": pred_traj,
                    "gt_xyz": gt_traj,
                    "coc_text": coc_text,
                    "clip_id": clip_id,
                    "t0_us": t0_us,
                    "completion_prefix": raw_completion[:prefix_end],
                    "hist_xyz": hist_xyz[0].cpu(),  # (T, 3) for reward computation
                    "hist_rot": hist_rot[0].cpu(),  # (T, 3, 3)
                }
            )

        return results

    def _generate_for_scene_expert(
        self,
        clip_id: str,
        t0_us: int,
        G: int,
        device: torch.device,
        traj_future_end_id: int,
        traj_future_start_id: int,
    ) -> list[dict]:
        """Generate G completions for a scene using VLM CoC + action expert diffusion.

        Optimized flow:
        1. Batch VLM CoC generation for all G samples at once (num_return_sequences=G)
        2. Per sample: teacher-forced VLM forward to reconstruct KV cache (fast, non-AR)
        3. Per sample: expert diffusion produces trajectory conditioned on KV cache
        4. Encode trajectories into discrete tokens for training

        The batch generation avoids redundant image/prompt prefill across G samples.
        Teacher-forced KV reconstruction is a single forward pass (~1s) vs autoregressive
        generation (~15-20s), giving ~3x overall speedup per scene.
        """
        t_scene = time.time()

        # 1. Load driving data
        model_inputs, ego_future_xyz = self.data_cache.get(clip_id, t0_us, device)

        # 2. Fuse history trajectory tokens
        tokenized = {k: v for k, v in model_inputs["tokenized_data"].items()}
        input_ids = tokenized.pop("input_ids")
        traj_data = {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        }
        input_ids = self.full_model.fuse_traj_tokens(input_ids, traj_data)

        # 3. Optionally append <adv_obs_pos> to input_ids (before VLM generation)
        if self.use_adv_conditioning and "adv_obs_pos" in self.adv_token_ids:
            adv_obs_id = self.adv_token_ids["adv_obs_pos"]
            adv_obs_tensor = torch.tensor([[adv_obs_id]], device=device, dtype=input_ids.dtype)
            prompt_ids_list = input_ids[0].cpu().tolist()
            input_ids = torch.cat([input_ids, adv_obs_tensor], dim=1)
        else:
            prompt_ids_list = input_ids[0].cpu().tolist()

        prompt_len = input_ids.shape[1]

        hist_xyz = model_inputs["ego_history_xyz"][:, -1]  # (1, T, 3)
        hist_rot = model_inputs["ego_history_rot"][:, -1]  # (1, T, 3, 3)
        gt_traj = ego_future_xyz[0, 0].numpy().flatten().tolist()

        # ---- 4. Batch VLM CoC generation for all G samples at once ----
        t_gen = time.time()
        gen_config = GenerationConfig(
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=G,
            max_new_tokens=self.max_generation_length + 10,
            pad_token_id=self.pad_token_id,
        )
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.traj_token_start_idx,
                    traj_vocab_size=self.traj_vocab_size,
                )
            ]
        )
        stopping = StoppingCriteriaList([StopAfterEOS(eos_token_id=traj_future_start_id)])

        with torch.autocast(str(device), dtype=torch.bfloat16):
            vlm_output = self.full_model.vlm.generate(
                input_ids=input_ids,
                generation_config=gen_config,
                stopping_criteria=stopping,
                logits_processor=logits_processor,
                **tokenized,
            )
        # vlm_output: (G, seq_len) — padded to longest sequence
        generated_seqs = vlm_output[:, prompt_len:]
        logger.debug(
            "Batch VLM CoC gen for %s: G=%d in %.2fs", clip_id, G, time.time() - t_gen
        )

        # ---- 5. Per-sample: teacher-forced KV cache + expert diffusion ----
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]  # 64
        results = []

        # Disable gradient checkpointing for all teacher-forced forwards + adv injection
        vlm = self.full_model.vlm
        gc_modules = [
            m for m in vlm.modules() if getattr(m, "gradient_checkpointing", False)
        ]
        for m in gc_modules:
            m.gradient_checkpointing = False

        try:
            for sample_idx in range(G):
                raw_completion = generated_seqs[sample_idx].cpu().tolist()

                # Strip padding tokens
                while raw_completion and raw_completion[-1] == self.pad_token_id:
                    raw_completion.pop()

                # Find <traj_future_start> position
                try:
                    traj_start_pos = raw_completion.index(traj_future_start_id)
                except ValueError:
                    logger.warning(
                        "No <traj_future_start> in generated sequence for %s sample %d, skipping",
                        clip_id,
                        sample_idx,
                    )
                    continue

                coc_tokens = raw_completion[:traj_start_pos]
                coc_text = self.tokenizer.decode(
                    coc_tokens, skip_special_tokens=True
                ).strip()
                completion_prefix_ids = coc_tokens + [traj_future_start_id]

                # ---- Teacher-forced VLM forward to reconstruct KV cache ----
                prefix_tensor = torch.tensor(
                    [completion_prefix_ids], device=device, dtype=torch.long
                )
                full_ids = torch.cat([input_ids, prefix_tensor], dim=1)

                tf_kwargs = {}
                if "attention_mask" in tokenized:
                    orig_mask = tokenized["attention_mask"]
                    prefix_mask = torch.ones(
                        1, len(completion_prefix_ids), device=device, dtype=orig_mask.dtype
                    )
                    tf_kwargs["attention_mask"] = torch.cat([orig_mask, prefix_mask], dim=1)
                for k in ("pixel_values", "image_grid_thw"):
                    if k in tokenized:
                        tf_kwargs[k] = tokenized[k]

                with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
                    tf_out = vlm(
                        input_ids=full_ids,
                        use_cache=True,
                        **tf_kwargs,
                    )

                prompt_cache = tf_out.past_key_values
                rope_deltas = vlm.model.rope_deltas
                prefill_seq_len = prompt_cache.get_seq_length()
                b_star = 1
                offset = torch.tensor([full_ids.shape[1]], device=device)

                # ---- Inject <adv_traj_pos> into KV cache (optional) ----
                if self.use_adv_conditioning and "adv_traj_pos" in self.adv_token_ids:
                    adv_traj_id = self.adv_token_ids["adv_traj_pos"]
                    adv_tensor = torch.tensor(
                        [[adv_traj_id]], device=device, dtype=torch.long
                    )
                    with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
                        adv_out = vlm(
                            input_ids=adv_tensor,
                            past_key_values=prompt_cache,
                            use_cache=True,
                        )
                    prompt_cache = adv_out.past_key_values
                    prefill_seq_len = prompt_cache.get_seq_length()
                    offset = offset + 1

                # ---- Build expert position_ids and attention_mask ----
                position_ids = torch.arange(n_diffusion_tokens, device=device)
                position_ids = einops.repeat(
                    position_ids, "l -> 3 b l", b=b_star
                ).clone()
                delta = rope_deltas + offset[:, None]
                position_ids += delta.to(position_ids.device)

                attention_mask = torch.zeros(
                    (b_star, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
                    dtype=torch.float32,
                    device=device,
                )
                for i in range(b_star):
                    attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                        attention_mask.dtype
                    ).min

                forward_kwargs = {}
                if self.expert_non_causal:
                    forward_kwargs["is_causal"] = False

                # ---- Diffusion sampling via expert ----
                def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                    b = x.shape[0]
                    future_token_embeds = self.action_in_proj(x, t)
                    if future_token_embeds.dim() == 2:
                        future_token_embeds = future_token_embeds.view(
                            b, n_diffusion_tokens, -1
                        )
                    expert_out = self.expert(
                        inputs_embeds=future_token_embeds,
                        position_ids=position_ids,
                        past_key_values=prompt_cache,  # noqa: F821
                        attention_mask=attention_mask,
                        use_cache=True,
                        **forward_kwargs,
                    )
                    prompt_cache.crop(prefill_seq_len)  # noqa: F821
                    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
                    return self.action_out_proj(last_hidden).view(
                        -1, *self.action_space.get_action_space_dims()
                    )

                diffusion_kwargs = {}
                if self.expert_diffusion_steps != 10:  # 10 is the default
                    diffusion_kwargs["num_steps"] = self.expert_diffusion_steps

                with torch.autocast(str(device), dtype=torch.bfloat16):
                    sampled_action = self.diffusion.sample(
                        batch_size=b_star,
                        step_fn=step_fn,
                        device=device,
                        return_all_steps=False,
                        **diffusion_kwargs,
                    )

                # ---- Convert action → trajectory → discrete tokens ----
                pred_xyz, pred_rot = self.action_space.action_to_traj(
                    sampled_action, hist_xyz, hist_rot
                )
                discrete_tokens = self.traj_tokenizer.encode(
                    hist_xyz, hist_rot, pred_xyz, pred_rot
                )
                discrete_tokens = discrete_tokens.clamp(0, self.traj_vocab_size - 1)
                traj_token_ids = (
                    (discrete_tokens + self.traj_token_start_idx).squeeze(0).tolist()
                )

                # ---- Build completion_ids and result dict ----
                completion_ids = (
                    coc_tokens
                    + [traj_future_start_id]
                    + traj_token_ids
                    + [traj_future_end_id]
                )

                pred_traj = pred_xyz[0].cpu().numpy().flatten().tolist()

                results.append(
                    {
                        "prompt_ids": prompt_ids_list,
                        "completion_ids": completion_ids,
                        "pred_xyz": pred_traj,
                        "gt_xyz": gt_traj,
                        "coc_text": coc_text,
                        "clip_id": clip_id,
                        "t0_us": t0_us,
                        "completion_prefix": completion_prefix_ids,
                        "hist_xyz": hist_xyz[0].cpu(),
                        "hist_rot": hist_rot[0].cpu(),
                    }
                )

                del prompt_cache

        finally:
            for m in gc_modules:
                m.gradient_checkpointing = True

        logger.debug(
            "Scene %s: %d/%d completions in %.2fs (gen=%.2fs)",
            clip_id,
            len(results),
            G,
            time.time() - t_scene,
            time.time() - t_gen,
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
                tokenized = {k: v for k, v in model_inputs["tokenized_data"].items()}
                input_ids = tokenized.pop("input_ids")
                traj_data = {
                    "ego_history_xyz": model_inputs["ego_history_xyz"],
                    "ego_history_rot": model_inputs["ego_history_rot"],
                }
                input_ids = self.full_model.fuse_traj_tokens(input_ids, traj_data)
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

        logger.info("Extracted segment hidden states for %d completions", len(segment_hidden_stash))
        return segment_hidden_stash, completion_segment_map

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
