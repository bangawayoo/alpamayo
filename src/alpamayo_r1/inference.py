"""Shared building blocks for VLM inference, rollout, and evaluation.

These functions consolidate the duplicated boilerplate for input preparation,
CoC generation, and trajectory decoding that was previously reimplemented
across alpamayo_r1.py, sft_rollout.py, and evaluate_test_set.py.
"""

from __future__ import annotations

from typing import Literal

import torch
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList

from alpamayo_r1.models.alpamayo_r1 import ExpertLogitsProcessor
from alpamayo_r1.models.token_utils import StopAfterEOS, extract_traj_tokens


def prepare_vlm_inputs(
    model,
    model_inputs: dict,
    adv_obs_token_id: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Fuse history trajectory tokens and build generation kwargs.

    Handles the repeated pattern of: pop input_ids from tokenized_data,
    call fuse_traj_tokens(), optionally append an advantage-conditioning
    observation token, and return the remaining tokenized fields as gen_kwargs.

    Args:
        model: AlpamayoR1 model (needs ``fuse_traj_tokens`` method).
        model_inputs: Dict with ``tokenized_data`` (containing ``input_ids``,
            ``attention_mask``, and optionally ``pixel_values``,
            ``image_grid_thw``), ``ego_history_xyz``, ``ego_history_rot``.
        adv_obs_token_id: If not None, append this token to input_ids and
            extend the attention mask by one position.

    Returns:
        input_ids: Tensor of shape ``(1, seq_len)`` with history tokens fused.
        gen_kwargs: Dict with ``attention_mask`` and any vision tensors
            (``pixel_values``, ``image_grid_thw``). Ready to be unpacked
            as ``**gen_kwargs`` into ``model.vlm.generate()``.
    """
    # Shallow copy so we don't mutate the caller's dict
    tokenized = {k: v for k, v in model_inputs["tokenized_data"].items()}
    input_ids = tokenized.pop("input_ids")

    traj_data = {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data)

    # Optionally append <adv_obs_pos> token
    if adv_obs_token_id is not None:
        device = input_ids.device
        adv_obs_tensor = torch.tensor([[adv_obs_token_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, adv_obs_tensor], dim=1)
        if "attention_mask" in tokenized:
            adv_mask = torch.ones(1, 1, device=device, dtype=tokenized["attention_mask"].dtype)
            tokenized["attention_mask"] = torch.cat([tokenized["attention_mask"], adv_mask], dim=1)

    return input_ids, tokenized


def generate_coc(
    model,
    input_ids: torch.Tensor,
    gen_kwargs: dict,
    *,
    mode: Literal["vlm", "expert"],
    temperature: float,
    top_p: float,
    num_samples: int,
    max_new_tokens: int,
    pad_token_id: int,
    return_dict: bool = False,
    autocast_device: torch.device | str | None = None,
) -> torch.Tensor:
    """Unified VLM Chain-of-Causation generation.

    Args:
        model: AlpamayoR1 model.
        input_ids: Fused input token IDs from ``prepare_vlm_inputs()``.
        gen_kwargs: Remaining tokenized fields (attention_mask, pixel_values,
            image_grid_thw) from ``prepare_vlm_inputs()``.
        mode: ``"vlm"`` stops at ``traj_future_end`` with no logits processor
            (VLM generates trajectory tokens directly). ``"expert"`` stops at
            ``traj_future_start`` and uses ``ExpertLogitsProcessor`` to mask
            trajectory token logits.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        num_samples: Number of return sequences (``num_return_sequences``).
        max_new_tokens: Maximum new tokens to generate.
        pad_token_id: Pad token ID for the generation config.
        return_dict: If True, set ``output_logits=True`` and
            ``return_dict_in_generate=True`` so the output is a
            ``GenerateOutput`` with logits and KV cache (needed for the
            expert diffusion pipeline in alpamayo_r1.py).
        autocast_device: If not None, wrap generation in ``torch.autocast``
            with bfloat16 on this device.

    Returns:
        VLM output — a plain tensor ``(B * num_samples, seq_len)`` by default,
        or a ``GenerateOutput`` if ``return_dict=True``.
    """
    gen_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_samples,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
    )

    if mode == "vlm":
        stop_token_id = model.special_token_ids["traj_future_end"]
        logits_processor = None
    else:
        stop_token_id = model.special_token_ids["traj_future_start"]
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=model.config.traj_token_start_idx,
                    traj_vocab_size=model.config.traj_vocab_size,
                )
            ]
        )

    if return_dict:
        gen_config.output_logits = True
        gen_config.return_dict_in_generate = True

    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=stop_token_id)])

    generate_kwargs = {
        "input_ids": input_ids,
        "generation_config": gen_config,
        "stopping_criteria": stopping_criteria,
        **gen_kwargs,
    }
    if logits_processor is not None:
        generate_kwargs["logits_processor"] = logits_processor

    if autocast_device is not None:
        with torch.autocast(str(autocast_device), dtype=torch.bfloat16):
            return model.vlm.generate(**generate_kwargs)
    return model.vlm.generate(**generate_kwargs)


def decode_vlm_trajectories(
    model,
    vlm_output: torch.Tensor,
    hist_xyz: torch.Tensor,
    hist_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract discrete trajectory tokens and decode to continuous waypoints.

    Consolidates the repeated pattern of ``extract_traj_tokens()`` followed
    by ``traj_tokenizer.decode()``.

    The trajectory tokenizer decode is pure math (dequantization + cumsum),
    so no ``torch.no_grad()`` wrapper is needed.

    Args:
        model: AlpamayoR1 model (needs ``special_token_ids``,
            ``future_token_start_idx``, ``config``, ``traj_tokenizer``).
        vlm_output: Raw VLM output sequences of shape ``(B, seq_len)``.
        hist_xyz: History positions, shape ``(B, T, 3)``.
        hist_rot: History rotations, shape ``(B, T, 3, 3)``.

    Returns:
        pred_xyz: Predicted future waypoints, shape ``(B, T_future, 3)``.
        pred_rot: Predicted future rotations, shape ``(B, T_future, 3, 3)``.
    """
    traj_tokens = extract_traj_tokens(
        vlm_output,
        model.special_token_ids,
        model.config.tokens_per_future_traj,
        model.future_token_start_idx,
        model.config.traj_vocab_size,
    )

    pred_xyz, pred_rot, _ = model.traj_tokenizer.decode(hist_xyz, hist_rot, traj_tokens)
    return pred_xyz, pred_rot
