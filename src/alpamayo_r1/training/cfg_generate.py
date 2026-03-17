"""Classifier-Free Guidance (CFG) generation for advantage-conditioned models.

At inference time, we use two forward passes to steer generation toward
high-advantage behavior:

    logits_final = logits_uncond + beta * (logits_cond - logits_uncond)

Pass 1 (unconditional): No advantage tokens — baseline generation
Pass 2 (conditioned): All-positive advantage tokens prepended

A higher beta amplifies the signal from positive conditioning, producing
trajectories that more strongly reflect high-advantage patterns learned
during SFT training.

See docs/advantage-conditioning.md Section 5 for details.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def cfg_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids: list[int],
    adv_token_ids: dict[str, int],
    special_token_ids: dict[str, int],
    beta: float = 1.5,
    temperature: float = 0.6,
    top_p: float = 0.98,
    max_new_tokens: int = 320,
    device: torch.device | None = None,
    **forward_kwargs,
) -> tuple[list[int], dict[str, Any]]:
    """Generate with classifier-free guidance using advantage conditioning.

    Runs two parallel forward passes per token:
    - Unconditional: prompt_ids → logits_u
    - Conditioned: prompt_ids + [pos,pos,pos] → logits_c
    - Combined: logits = logits_u + beta * (logits_c - logits_u)

    Args:
        model: VLM model (Qwen3VL or LoRA-wrapped).
        tokenizer: Tokenizer with advantage tokens registered.
        prompt_ids: Tokenized prompt (with fused history trajectory).
        adv_token_ids: Dict mapping token name -> ID.
        special_token_ids: Dict with traj_future_end etc.
        beta: Guidance strength. 0.0 = unconditional, 1.0 = pure conditional,
            >1.0 = amplified conditioning.
        temperature: Sampling temperature (applied after CFG).
        top_p: Nucleus sampling threshold.
        max_new_tokens: Maximum tokens to generate.
        device: CUDA device.
        **forward_kwargs: Extra kwargs for model.generate (pixel_values, etc).

    Returns:
        (generated_ids, metadata) where generated_ids is the completion token
        list and metadata contains diagnostics.
    """
    device = device or next(model.parameters()).device

    # Build conditioning tokens (all positive, obs level only — traj token
    # is inserted mid-generation after CoC, not at the start)
    obs_cond_token = [adv_token_ids["adv_obs_pos"]]

    # Build two input sequences (adv_obs_pos prepended for conditioned pass)
    uncond_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cond_ids = torch.tensor([prompt_ids + obs_cond_token], dtype=torch.long, device=device)

    # Batched input: [unconditional, conditional]
    # Pad shorter sequence to match length
    max_len = max(uncond_ids.shape[1], cond_ids.shape[1])
    pad_id = tokenizer.pad_token_id or 0

    if uncond_ids.shape[1] < max_len:
        pad = torch.full(
            (1, max_len - uncond_ids.shape[1]),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        uncond_ids = torch.cat([pad, uncond_ids], dim=1)  # left-pad
    if cond_ids.shape[1] < max_len:
        pad = torch.full(
            (1, max_len - cond_ids.shape[1]),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        cond_ids = torch.cat([pad, cond_ids], dim=1)

    batched_ids = torch.cat([uncond_ids, cond_ids], dim=0)  # (2, L)

    # Attention mask (1 for real tokens, 0 for padding)
    attn_mask = (batched_ids != pad_id).long()

    # Duplicate forward_kwargs for batch of 2
    batched_kwargs = {}
    for k, v in forward_kwargs.items():
        if isinstance(v, torch.Tensor):
            batched_kwargs[k] = v.repeat(2, *([1] * (v.dim() - 1)))
        else:
            batched_kwargs[k] = v

    # Stop at traj_future_end
    traj_future_end_id = special_token_ids.get("traj_future_end")

    # Token-by-token generation with CFG logits mixing
    generated = []
    past_key_values = None
    current_ids = batched_ids
    current_mask = attn_mask

    model.eval()
    with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
                **batched_kwargs if past_key_values is None else {},
            )

            # Get logits for the last position
            logits_uncond = outputs.logits[0, -1, :]  # (vocab,)
            logits_cond = outputs.logits[1, -1, :]  # (vocab,)

            # CFG interpolation
            logits = logits_uncond + beta * (logits_cond - logits_uncond)

            # Temperature + sampling
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)

                # Top-p filtering
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                sample_idx = torch.multinomial(sorted_probs, 1)
                next_token = sorted_indices[sample_idx]
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            next_token_id = next_token.item()
            generated.append(next_token_id)

            # Check stopping
            if traj_future_end_id and next_token_id == traj_future_end_id:
                break

            # Prepare next step
            past_key_values = outputs.past_key_values
            current_ids = next_token.unsqueeze(0).expand(2, -1)  # (2, 1)
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(2, 1, device=device, dtype=torch.long),
                ],
                dim=1,
            )
            # Clear forward_kwargs after first step (handled by KV cache)
            batched_kwargs = {}

    metadata = {
        "beta": beta,
        "num_tokens_generated": len(generated),
        "stopped_at_eos": (len(generated) > 0 and generated[-1] == traj_future_end_id)
        if traj_future_end_id
        else False,
    }

    return generated, metadata
