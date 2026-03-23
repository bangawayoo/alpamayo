"""Shared building blocks for VLM inference, rollout, and evaluation.

These functions consolidate the duplicated boilerplate for input preparation,
CoC generation, and trajectory decoding that was previously reimplemented
across alpamayo_r1.py, sft_rollout.py, and evaluate_test_set.py.
"""

from __future__ import annotations

import torch


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
