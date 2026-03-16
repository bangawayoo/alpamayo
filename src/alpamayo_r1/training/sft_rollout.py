"""Rollout engine for the SFT pipeline, decoupled from GRPOTrainer.

Generates completions from the current policy and extracts segment-level
hidden states for value head scoring. Supports HuggingFace generation
(default) and optional vLLM server mode for faster rollouts.

Adapted from AlpamayoGRPOTrainer._generate_single_turn() in rollout.py.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import GenerationConfig, StoppingCriteriaList

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.token_utils import StopAfterEOS, extract_traj_tokens
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
    ) -> None:
        self.full_model = full_model
        self.processor = processor
        self.data_cache = data_cache
        self.rollout_cfg = rollout_cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Config
        self.temperature = float(rollout_cfg.get("temperature", 1.2))
        self.top_p = float(rollout_cfg.get("top_p", 0.98))
        self.max_generation_length = int(rollout_cfg.get("max_generation_length", 256))
        self.logprob_mini_batch_size = int(rollout_cfg.get("logprob_mini_batch_size", 2))

        # Model constants
        self.traj_token_start_idx = full_model.future_token_start_idx
        self.tokens_per_future_traj = full_model.config.tokens_per_future_traj
        self.traj_vocab_size = full_model.config.traj_vocab_size
        self.special_token_ids = full_model.special_token_ids
        self.traj_tokenizer = full_model.traj_tokenizer
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.tokenizer = full_model.tokenizer

    def generate_completions(
        self,
        clip_ids: list[str],
        t0_us: int,
        G: int,
    ) -> list[dict]:
        """Generate G completions per scene.

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

        # Ensure VLM and trajectory tokenizers are on the target device
        self.full_model.vlm.to(device)
        if self.traj_tokenizer is not None:
            self.traj_tokenizer.to(device)
        if hasattr(self.full_model, "hist_traj_tokenizer") and self.full_model.hist_traj_tokenizer is not None:
            self.full_model.hist_traj_tokenizer.to(device)
        self.full_model.vlm.eval()
        with torch.no_grad():
            for clip_id in clip_ids:
                try:
                    scene_results = self._generate_for_scene(
                        clip_id,
                        t0_us,
                        G,
                        device,
                        traj_future_end_id,
                        traj_future_start_id,
                    )
                    results.extend(scene_results)
                except Exception as e:
                    logger.warning("Generation failed for %s: %s", clip_id, e)
                    continue

        logger.info("Generated %d completions from %d scenes", len(results), len(clip_ids))
        return results

    def _generate_for_scene(
        self,
        clip_id: str,
        t0_us: int,
        G: int,
        device: torch.device,
        traj_future_end_id: int,
        traj_future_start_id: int,
    ) -> list[dict]:
        """Generate G completions for a single scene."""
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
