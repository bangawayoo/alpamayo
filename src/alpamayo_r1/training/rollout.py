# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom GRPOTrainer subclass for Alpamayo-R1.

TRL's ``rollout_func`` is only invoked in vLLM code paths. To use our custom
VLM + Expert + Diffusion pipeline without requiring vLLM, we override
``_generate_single_turn`` directly. This gives us full control over generation
while keeping TRL's training loop, loss computation, logging, and checkpointing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch
import torch.nn.functional as F
from physical_ai_av import PhysicalAIAVDatasetInterface
from transformers import AutoProcessor
from trl import GRPOTrainer

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

logger = logging.getLogger(__name__)


def _parse_clip_metadata(prompt_text: str) -> tuple[str, int]:
    """Extract clip_id and t0_us from the prompt system message.

    The dataset builder encodes ``[clip_id=...] [t0_us=...]`` in the system
    content.  We parse them back out here.

    Args:
        prompt_text: Full prompt string (all roles concatenated by TRL).

    Returns:
        (clip_id, t0_us) tuple.

    Raises:
        ValueError: If clip_id or t0_us cannot be parsed.
    """
    clip_match = re.search(r"\[clip_id=([^\]]+)\]", prompt_text)
    t0_match = re.search(r"\[t0_us=(\d+)\]", prompt_text)
    if clip_match is None or t0_match is None:
        raise ValueError(f"Could not parse clip metadata from prompt: {prompt_text[:200]}")
    return clip_match.group(1), int(t0_match.group(1))


class AlpamayoGRPOTrainer(GRPOTrainer):
    """GRPOTrainer subclass that uses the full Alpamayo-R1 pipeline for generation.

    Overrides ``_generate_single_turn`` to run the VLM → CoC → Expert → Diffusion
    pipeline instead of standard HF ``model.generate()``. This is necessary because:

    1. Standard generation only produces text — we also need trajectory predictions
       for the trajectory quality and consistency reward functions.
    2. The full pipeline involves non-VLM components (expert transformer, diffusion)
       that are not part of the model passed to GRPOTrainer.
    3. TRL's ``rollout_func`` is only called in vLLM code paths, not in the regular
       generation path.

    Args:
        full_model: The complete AlpamayoR1 model (VLM + expert + diffusion).
            The VLM (``full_model.vlm``) should be passed as the ``model`` arg
            to the parent GRPOTrainer.
        avdi: PhysicalAI-AV dataset interface for loading driving data.
        rollout_temperature: Sampling temperature for VLM generation.
        rollout_top_p: Nucleus sampling threshold.
        rollout_max_generation_length: Maximum CoC tokens.
        **kwargs: All other arguments forwarded to GRPOTrainer.
    """

    def __init__(
        self,
        *args,
        full_model: AlpamayoR1,
        avdi: PhysicalAIAVDatasetInterface,
        rollout_temperature: float = 0.6,
        rollout_top_p: float = 0.98,
        rollout_max_generation_length: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.full_model = full_model
        self.avdi = avdi
        self.rollout_temperature = rollout_temperature
        self.rollout_top_p = rollout_top_p
        self.rollout_max_generation_length = rollout_max_generation_length

    def _generate_single_turn(self, prompts: list):
        """Override generation to use the full Alpamayo pipeline.

        Runs the VLM + Expert + Diffusion pipeline for each unique prompt,
        extracts CoC text and trajectory predictions, computes per-token
        log-probs via teacher-forced VLM forward, and returns everything
        in TRL's expected format.

        Args:
            prompts: List of prompt strings or message lists (B*G items,
                with ``num_generations`` duplicates per unique prompt).

        Returns:
            Tuple of (prompt_ids, completion_ids, logprobs, extra_fields)
            matching TRL's internal interface.
        """
        device = self.accelerator.device
        num_generations = self.num_generations if self.model.training else self.num_generations_eval

        # Temporarily disable gradient checkpointing during rollout.
        # The Alpamayo pipeline needs use_cache=True for KV cache in the
        # expert model, which is incompatible with gradient checkpointing.
        gc_enabled = getattr(self.full_model.vlm, "is_gradient_checkpointing", False)
        if gc_enabled:
            self.full_model.vlm.gradient_checkpointing_disable()

        # De-duplicate prompts (TRL repeats each prompt num_generations times)
        unique_prompts = prompts[::num_generations]

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_pred_xyz: list[list[float]] = []
        all_gt_xyz: list[list[float]] = []

        for prompt in unique_prompts:
            # Resolve prompt to string if conversational
            if isinstance(prompt, list):
                prompt_text = " ".join(
                    m.get("content", "") if isinstance(m.get("content"), str)
                    else " ".join(c.get("text", "") for c in m.get("content", []) if isinstance(c, dict))
                    for m in prompt
                )
            else:
                prompt_text = prompt

            clip_id, t0_us = _parse_clip_metadata(prompt_text)

            # 1. Load driving data
            data = load_physical_aiavdataset(
                clip_id=clip_id, t0_us=t0_us, avdi=self.avdi, maybe_stream=True
            )

            # 2. Prepare model inputs
            model_inputs = helper.prepare_model_inputs(
                data, self.processing_class, device
            )
            prompt_len = model_inputs["tokenized_data"]["input_ids"].shape[1]
            prompt_input_ids = model_inputs["tokenized_data"]["input_ids"].clone()

            # 3. Run full pipeline
            with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = (
                    self.full_model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=self.rollout_top_p,
                        temperature=self.rollout_temperature,
                        num_traj_samples=num_generations,
                        max_generation_length=self.rollout_max_generation_length,
                        return_extra=True,
                    )
                )

            # 4. Extract CoC text
            coc_texts_raw = extra.get("cot", None)
            if coc_texts_raw is not None:
                coc_texts = coc_texts_raw.flatten().tolist()
            else:
                coc_texts = [""] * num_generations

            # 5. Build per-sample outputs (num_generations per unique prompt)
            prompt_ids_list = prompt_input_ids[0].cpu().tolist()

            for sample_idx in range(num_generations):
                coc_text = coc_texts[sample_idx] if sample_idx < len(coc_texts) else ""

                # Tokenize CoC completion. Guarantee at least one token —
                # TRL's _generate() accesses ids[-1] and crashes on empty lists.
                coc_token_ids = self.processing_class.tokenizer.encode(
                    coc_text, add_special_tokens=False
                )
                if not coc_token_ids:
                    coc_token_ids = [self.processing_class.tokenizer.eos_token_id]

                all_prompt_ids.append(prompt_ids_list)
                all_completion_ids.append(coc_token_ids)

                # Trajectory data
                pred_traj = pred_xyz[0, 0, sample_idx].cpu().numpy().flatten().tolist()
                gt_traj = data["ego_future_xyz"][0, 0].numpy().flatten().tolist()
                all_pred_xyz.append(pred_traj)
                all_gt_xyz.append(gt_traj)

            # 6. Compute log-probs via teacher-forced VLM forward
            batch_logprobs = _compute_batch_logprobs(
                self.full_model,
                model_inputs,
                prompt_input_ids,
                all_completion_ids[-num_generations:],
                prompt_len,
                device,
            )
            all_logprobs.extend(batch_logprobs)

        # Re-enable gradient checkpointing if it was active
        if gc_enabled:
            self.full_model.vlm.gradient_checkpointing_enable()

        extra_fields = {
            "pred_xyz": all_pred_xyz,
            "gt_xyz": all_gt_xyz,
        }

        return all_prompt_ids, all_completion_ids, all_logprobs, extra_fields


def _compute_batch_logprobs(
    full_model: AlpamayoR1,
    model_inputs: dict,
    prompt_input_ids: torch.Tensor,
    completion_ids_list: list[list[int]],
    prompt_len: int,
    device: torch.device,
) -> list[list[float]]:
    """Compute per-token log-probs for a batch of completions.

    Args:
        full_model: The full AlpamayoR1 model.
        model_inputs: Dict with tokenized_data (input_ids may be popped).
        prompt_input_ids: Saved prompt input_ids, shape (1, L_prompt).
        completion_ids_list: List of completion token ID lists.
        prompt_len: Number of prompt tokens.
        device: CUDA device.

    Returns:
        List of per-token log-prob lists, one per completion.
    """
    results: list[list[float]] = []

    for comp_ids in completion_ids_list:
        if not comp_ids:
            results.append([])
            continue

        comp_tensor = torch.tensor(comp_ids, dtype=torch.long, device=device)

        # Concatenate prompt + completion for teacher-forced forward
        full_ids = torch.cat(
            [prompt_input_ids[0], comp_tensor], dim=0
        ).unsqueeze(0)

        # Prepare forward kwargs (attention_mask, pixel_values, etc.)
        forward_kwargs = {}
        tokenized = model_inputs["tokenized_data"]
        if "attention_mask" in tokenized:
            orig_mask = tokenized["attention_mask"]
            ext_mask = torch.ones(
                1, comp_tensor.shape[0], device=device, dtype=orig_mask.dtype
            )
            forward_kwargs["attention_mask"] = torch.cat([orig_mask, ext_mask], dim=1)
        if "pixel_values" in tokenized:
            forward_kwargs["pixel_values"] = tokenized["pixel_values"]
        if "image_grid_thw" in tokenized:
            forward_kwargs["image_grid_thw"] = tokenized["image_grid_thw"]

        with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
            outputs = full_model.vlm(input_ids=full_ids, **forward_kwargs)

        # Logits at position t predict token t+1
        comp_len = comp_tensor.shape[0]
        logits = outputs.logits[0, prompt_len - 1 : prompt_len - 1 + comp_len]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(
            1, comp_tensor.unsqueeze(-1)
        ).squeeze(-1)
        results.append(token_log_probs.cpu().tolist())

    return results


def _collate_rollout_outputs(
    all_prompt_ids: list[torch.Tensor],
    all_completion_ids: list[torch.Tensor],
    all_logprobs: list[torch.Tensor],
    all_pred_xyz: list[list[float]],
    all_gt_xyz: list[list[float]],
    all_coc_texts: list[str],
    pad_token_id: int,
) -> dict[str, Any]:
    """Pad and collate rollout outputs into a batch dict.

    Args:
        all_prompt_ids: List of prompt token tensors (varying lengths).
        all_completion_ids: List of completion token tensors.
        all_logprobs: List of log-prob tensors.
        all_pred_xyz: Flattened predicted trajectories.
        all_gt_xyz: Flattened ground-truth trajectories.
        all_coc_texts: Decoded CoC strings.
        pad_token_id: Token ID used for padding.

    Returns:
        Dict matching TRL's expected rollout_func output format.
    """
    # Pad prompt_ids to same length
    max_prompt_len = max(t.shape[0] for t in all_prompt_ids)
    prompt_ids_padded = torch.full(
        (len(all_prompt_ids), max_prompt_len), pad_token_id, dtype=torch.long
    )
    for i, t in enumerate(all_prompt_ids):
        prompt_ids_padded[i, max_prompt_len - t.shape[0] :] = t  # left-pad prompts

    # Pad completion_ids to same length
    max_comp_len = max(t.shape[0] for t in all_completion_ids) if all_completion_ids else 1
    max_comp_len = max(max_comp_len, 1)  # at least 1
    completion_ids_padded = torch.full(
        (len(all_completion_ids), max_comp_len), pad_token_id, dtype=torch.long
    )
    for i, t in enumerate(all_completion_ids):
        if t.shape[0] > 0:
            completion_ids_padded[i, : t.shape[0]] = t

    # Pad logprobs to same length (pad with 0.0)
    logprobs_padded = torch.zeros(
        (len(all_logprobs), max_comp_len), dtype=torch.float32
    )
    for i, t in enumerate(all_logprobs):
        if t.shape[0] > 0:
            logprobs_padded[i, : t.shape[0]] = t

    return {
        "prompt_ids": prompt_ids_padded,
        "completion_ids": completion_ids_padded,
        "logprobs": logprobs_padded,
        # Extra fields forwarded to reward functions
        "pred_xyz": all_pred_xyz,
        "gt_xyz": all_gt_xyz,
        "completions": all_coc_texts,
    }
