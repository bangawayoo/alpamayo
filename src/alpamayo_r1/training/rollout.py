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

"""Custom GRPOTrainer subclass for Alpamayo-R1 (VLM-only rollouts).

The VLM autoregressively generates both Chain-of-Thought reasoning text AND
discrete trajectory tokens during GRPO rollouts, following the paper
(arXiv:2511.00088). No Expert or Diffusion modules are needed during rollout
-- trajectory tokens are decoded to continuous xyz via the trajectory tokenizer.

Three generation backends are supported:
1. **HuggingFace** (default): ``model.generate()`` via ``_generate_single_turn``.
2. **vLLM colocate**: PagedAttention-accelerated generation via a custom
   ``rollout_func`` passed to TRL's ``GRPOTrainer``.  Enable with
   ``vllm.enabled: true`` in the Hydra config.
3. **vLLM server**: Same as colocate but vLLM runs as a separate process
   (``trl vllm-serve``). Enable with ``vllm.enabled: true, vllm.mode: server``.
   The rollout_func is called on rank 0 only with ALL prompts gathered.

We override ``_generate_single_turn`` to call VLM.generate() directly (instead
of the full VLM -> Expert -> Diffusion pipeline). This simplifies the
architecture and reduces GPU memory during training.
"""

from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from physical_ai_av import PhysicalAIAVDatasetInterface
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, StoppingCriteriaList, TrainerCallback
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation

from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.token_utils import StopAfterEOS, extract_traj_tokens
from alpamayo_r1.training.advantage_conditioning import compute_trajectory_returns
from alpamayo_r1.training.rollout_utils import (
    ClipDataCache,
    collapse_image_pad_tokens,
    compute_gae,
    group_consecutive_prompts,
    parse_clip_metadata,
    prepare_vlm_for_training,  # noqa: F401 (re-exported for train_grpo.py)
    prompt_eq,
)

if TYPE_CHECKING:
    from trl import GRPOConfig

logger = logging.getLogger(__name__)

# Backward-compatible aliases (private names used internally in this file)
_group_consecutive_prompts = group_consecutive_prompts
_prompt_eq = prompt_eq
_parse_clip_metadata = parse_clip_metadata
_collapse_image_pad_tokens = collapse_image_pad_tokens
_compute_gae = compute_gae

_vllm_qwen3vl_patched = False


def _patch_vllm_qwen3vl_embed() -> None:
    """Patch vLLM's transformers-backend ``embed_multimodal`` for Qwen3-VL.

    Qwen3-VL's ``get_image_features`` returns a tuple
    ``(image_embeds, deepstack_image_embeds)`` instead of a single tensor.
    The generic ``embed_multimodal`` in
    ``vllm.model_executor.models.transformers.multimodal.MultiModalMixin``
    only handles the ``torch.Tensor`` case: when the return is a tuple it
    falls through the ``isinstance`` check and is returned raw, causing the
    profiling sanity check (and downstream cache code) to crash.

    We monkey-patch ``embed_multimodal`` to unpack the tuple and apply the
    standard split-by-``num_image_patches`` logic to the first element.
    """
    global _vllm_qwen3vl_patched  # noqa: PLW0603
    if _vllm_qwen3vl_patched:
        return
    try:
        from vllm.model_executor.models.transformers.multimodal import MultiModalMixin

        def _patched_embed_multimodal(self, **kwargs):
            pixel_values = kwargs.pop("pixel_values", None)
            image_embeds = kwargs.pop("image_embeds", None)
            if pixel_values is None:
                pixel_values = kwargs.pop("image_patches", None)

            if image_embeds is not None:
                return image_embeds
            if pixel_values is None:
                return None

            num_image_patches = kwargs.pop("num_image_patches")
            kwargs.pop("token_type_ids", None)

            vision_embeddings = self.model.get_image_features(pixel_values, **kwargs)

            # Qwen3-VL returns (image_embeds, deepstack_image_embeds).
            # Extract just the image embeddings.
            if isinstance(vision_embeddings, (tuple, list)) and not isinstance(
                vision_embeddings, torch.Tensor
            ):
                vision_embeddings = vision_embeddings[0]

            if isinstance(vision_embeddings, torch.Tensor):
                if vision_embeddings.ndim == 2:
                    vision_embeddings = vision_embeddings.unsqueeze(0)
                vision_embeddings = torch.split(
                    vision_embeddings, num_image_patches.flatten().tolist()
                )
                vision_embeddings = [
                    embed.flatten(start_dim=0, end_dim=-2) for embed in vision_embeddings
                ]

            return vision_embeddings

        MultiModalMixin.embed_multimodal = _patched_embed_multimodal
        _vllm_qwen3vl_patched = True
        logger.info(
            "Patched vLLM MultiModalMixin.embed_multimodal for "
            "Qwen3-VL tuple return from get_image_features."
        )
    except (ImportError, AttributeError) as exc:
        logger.debug("Could not patch vLLM embed_multimodal: %s", exc)


def make_vllm_rollout_func(
    full_model: AlpamayoR1,
    data_cache: ClipDataCache,
    rollout_temperature: float = 0.6,
    rollout_top_p: float = 0.98,
    rollout_max_generation_length: int = 256,
    vllm_mode: str = "colocate",
) -> callable:
    """Factory that builds a ``rollout_func`` for TRL's vLLM mode.

    The returned function has signature ``(prompts, trainer) -> dict`` and
    handles ONLY generation.  Trajectory token extraction / decoding happens
    later in ``AlpamayoGRPOTrainer._calculate_rewards()``.

    Supports two vLLM modes:
    - **colocate**: vLLM engine runs in-process (``trainer.vllm_generation.llm``).
      Called on every rank with local prompts.
    - **server**: vLLM runs as a separate process via ``trl vllm-serve``
      (``trainer.vllm_generation.vllm_client``). Called on rank 0 only with
      ALL prompts gathered from all ranks.

    Args:
        full_model: Complete AlpamayoR1 model (used for tokenizer config,
            ``fuse_traj_tokens``, and special token IDs).
        data_cache: Shared ``ClipDataCache`` (with ``cache_pil_images=True``).
        rollout_temperature: Sampling temperature for generation.
        rollout_top_p: Nucleus sampling threshold.
        rollout_max_generation_length: Max CoC text tokens (trajectory
            budget is added automatically).
        vllm_mode: Either ``"colocate"`` or ``"server"``.

    Returns:
        A callable suitable for ``GRPOTrainer(rollout_func=...)``.
    """
    # Pre-resolve model config fields (avoid repeated attribute lookups).
    tokens_per_future_traj = full_model.config.tokens_per_future_traj
    traj_future_end_id = full_model.special_token_ids["traj_future_end"]
    max_new_tokens = rollout_max_generation_length + tokens_per_future_traj + 10

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        """Generate completions for *prompts* using vLLM engine.

        In colocate mode, TRL passes ``B*G`` prompts (each unique prompt
        repeated ``num_generations`` times) on every rank.  In server mode,
        this is called on rank 0 only with ALL prompts gathered.

        We de-duplicate, call vLLM once per unique prompt (with ``n=1``
        per request, one request per generation), and collate the outputs.
        """

        device = trainer.accelerator.device

        # De-duplicate prompts: TRL repeats each prompt, but distributed
        # training may split repetitions across GPUs.  Detect actual local
        # repeat counts instead of assuming num_generations copies are here.
        prompt_groups = _group_consecutive_prompts(prompts)
        gen_counts = [count for _, count in prompt_groups]

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[list[float]]] = []
        all_gt_xyz: list[list[float]] = []
        all_hist_xyz: list[list[float]] = []
        all_hist_rot: list[list[float]] = []
        all_clip_ids: list[str] = []

        # Build vLLM inputs for all prompts (expanded by local gen count)
        vllm_inputs: list[dict] = []
        prompt_ids_per_unique: list[list[int]] = []
        pil_images_per_unique: list[list] = []
        gt_xyz_per_unique: list[list[float]] = []
        hist_xyz_per_unique: list[list[float]] = []
        hist_rot_per_unique: list[list[float]] = []
        clip_ids_per_unique: list[str] = []

        for prompt, local_gen_count in prompt_groups:
            # Resolve prompt to string if conversational
            if isinstance(prompt, list):
                prompt_text = " ".join(
                    m.get("content", "")
                    if isinstance(m.get("content"), str)
                    else " ".join(
                        c.get("text", "") for c in m.get("content", []) if isinstance(c, dict)
                    )
                    for m in prompt
                )
            else:
                prompt_text = prompt

            clip_id, t0_us = _parse_clip_metadata(prompt_text)

            # Load clip data (cached)
            model_inputs, ego_future_xyz = data_cache.get(clip_id, t0_us, device)

            # Fuse history trajectory tokens into input_ids
            tokenized = model_inputs["tokenized_data"]
            input_ids = tokenized.pop("input_ids")
            traj_data = {
                "ego_history_xyz": model_inputs["ego_history_xyz"],
                "ego_history_rot": model_inputs["ego_history_rot"],
            }
            input_ids = full_model.fuse_traj_tokens(input_ids, traj_data)
            prompt_token_ids = input_ids[0].cpu().tolist()

            # Collapse <|image_pad|> runs to a single token per image.
            # vLLM re-inserts the correct count from the PIL images; keeping
            # all pre-expanded pads causes double-insertion and CUDA errors.
            prompt_token_ids = _collapse_image_pad_tokens(prompt_token_ids)

            # Get PIL images for vLLM multimodal
            pil_images = data_cache.get_pil_images(clip_id, t0_us)

            # Cache prompt-level data
            prompt_ids_per_unique.append(prompt_token_ids)
            pil_images_per_unique.append(pil_images)
            gt_xyz_per_unique.append(ego_future_xyz[0, 0].numpy().flatten().tolist())
            hist_xyz_per_unique.append(
                model_inputs["ego_history_xyz"][:, -1].cpu().numpy().flatten().tolist()
            )
            hist_rot_per_unique.append(
                model_inputs["ego_history_rot"][:, -1].cpu().numpy().flatten().tolist()
            )
            clip_ids_per_unique.append(clip_id)

            # Repeat each prompt local_gen_count times (colocate uses n=1)
            for _ in range(local_gen_count):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": prompt_token_ids,
                        "multi_modal_data": {"image": pil_images},
                    }
                )

        # Call vLLM generate — branch on mode
        if vllm_mode == "colocate":
            from vllm import SamplingParams  # lazy import — only needed in colocate mode

            sampling_params = SamplingParams(
                temperature=rollout_temperature,
                top_p=rollout_top_p,
                max_tokens=max_new_tokens,
                stop_token_ids=[traj_future_end_id],
                include_stop_str_in_output=True,
                logprobs=1,
            )
            request_outputs = trainer.vllm_generation.llm.generate(
                vllm_inputs,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            # Parse colocate outputs (list of RequestOutput objects)
            # Build a mapping from request index to unique prompt index
            _req_to_unique = []
            for uid, gc in enumerate(gen_counts):
                _req_to_unique.extend([uid] * gc)
            for req_idx, req_output in enumerate(request_outputs):
                unique_idx = _req_to_unique[req_idx]
                output = req_output.outputs[0]  # n=1, so single output per request

                all_prompt_ids.append(prompt_ids_per_unique[unique_idx])

                completion_ids = list(output.token_ids)

                token_logprobs: list[list[float]] = []
                if output.logprobs:
                    for pos_logprobs in output.logprobs:
                        lp_values = [lp.logprob for lp in pos_logprobs.values()]
                        token_logprobs.append(lp_values[:1] if lp_values else [0.0])
                else:
                    token_logprobs = [[0.0]] * len(completion_ids)

                if completion_ids and completion_ids[-1] != traj_future_end_id:
                    completion_ids.append(traj_future_end_id)
                    token_logprobs.append([0.0])
                if not completion_ids:
                    completion_ids = [traj_future_end_id]
                    token_logprobs = [[0.0]]
                all_completion_ids.append(completion_ids)
                all_logprobs.append(token_logprobs)

                all_gt_xyz.append(gt_xyz_per_unique[unique_idx])
                all_hist_xyz.append(hist_xyz_per_unique[unique_idx])
                all_hist_rot.append(hist_rot_per_unique[unique_idx])
                all_clip_ids.append(clip_ids_per_unique[unique_idx])

        else:  # server mode
            vllm_client = trainer.vllm_generation.vllm_client

            # Build per-generation prompt_token_ids and images.
            # Reuse the already-computed prompt_token_ids (with fused
            # history trajectory tokens) and PIL images from the loop above.
            server_token_ids: list[list[int]] = []
            images_per_prompt: list[list] = []
            for unique_idx, gc in enumerate(gen_counts):
                for _ in range(gc):
                    server_token_ids.append(prompt_ids_per_unique[unique_idx])
                    images_per_prompt.append(pil_images_per_unique[unique_idx])

            result = vllm_client.generate(
                prompt_token_ids=server_token_ids,
                images=images_per_prompt,
                temperature=rollout_temperature,
                top_p=rollout_top_p,
                max_tokens=max_new_tokens,
                logprobs=1,
                mm_processor_kwargs={
                    "min_pixels": helper.MIN_PIXELS,
                    "max_pixels": helper.MAX_PIXELS,
                },
                generation_kwargs={"stop_token_ids": [traj_future_end_id]},
            )

            # Parse server response (dict with lists)
            _srv_to_unique = []
            for uid, gc in enumerate(gen_counts):
                _srv_to_unique.extend([uid] * gc)
            for req_idx in range(len(server_token_ids)):
                unique_idx = _srv_to_unique[req_idx]

                all_prompt_ids.append(result["prompt_ids"][req_idx])

                completion_ids = list(result["completion_ids"][req_idx])
                token_logprobs = list(result["logprobs"][req_idx])

                if completion_ids and completion_ids[-1] != traj_future_end_id:
                    completion_ids.append(traj_future_end_id)
                    token_logprobs.append([0.0])
                if not completion_ids:
                    completion_ids = [traj_future_end_id]
                    token_logprobs = [[0.0]]
                all_completion_ids.append(completion_ids)
                all_logprobs.append(token_logprobs)

                all_gt_xyz.append(gt_xyz_per_unique[unique_idx])
                all_hist_xyz.append(hist_xyz_per_unique[unique_idx])
                all_hist_rot.append(hist_rot_per_unique[unique_idx])
                all_clip_ids.append(clip_ids_per_unique[unique_idx])

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            # Extra fields — forwarded to reward functions via _calculate_rewards
            "gt_xyz": all_gt_xyz,
            "hist_xyz": all_hist_xyz,
            "hist_rot": all_hist_rot,
            "clip_ids": all_clip_ids,
        }

    return rollout_func


class AlpamayoGRPOTrainer(GRPOTrainer):
    """GRPOTrainer subclass with VLM-only rollouts for Alpamayo-R1.

    The VLM autoregressively generates both Chain-of-Thought (CoC) reasoning
    text AND discrete trajectory tokens. Trajectory tokens are decoded to
    continuous xyz via ``traj_tokenizer.decode()`` for reward computation.
    The Expert and Diffusion modules are NOT used during rollouts.

    ``completion_ids`` contains the full generated sequence (CoC text +
    trajectory tokens), so GRPO jointly optimizes reasoning and trajectory
    prediction.

    Args:
        full_model: The complete AlpamayoR1 model. Only ``full_model.vlm``
            and ``full_model.traj_tokenizer`` are used during rollouts.
            The VLM (``full_model.vlm``) should be passed as the ``model``
            arg to the parent GRPOTrainer.
        avdi: PhysicalAI-AV dataset interface for loading driving data.
        rollout_temperature: Sampling temperature for VLM generation.
        rollout_top_p: Nucleus sampling threshold.
        rollout_max_generation_length: Maximum CoC text tokens (trajectory
            tokens budget is added automatically).
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
        logprob_mini_batch_size: int = 4,
        data_cache_max_size: int = 200,
        value_head_cfg: dict | None = None,
        expert_finetune_cfg: dict | None = None,
        **kwargs,
    ):
        # Detect vLLM mode from GRPOConfig *before* calling super().__init__,
        # because the parent passes rollout_func to VLLMGeneration.__init__.
        grpo_config: GRPOConfig | None = kwargs.get("args") or (args[1] if len(args) > 1 else None)
        use_vllm = getattr(grpo_config, "use_vllm", False) if grpo_config is not None else False

        if use_vllm:
            vllm_mode = getattr(grpo_config, "vllm_mode", "colocate")
            # Create ClipDataCache early — shared between rollout_func and trainer.
            # processing_class may be passed as kwarg or positional arg.
            processor = kwargs.get("processing_class")
            self._data_cache = ClipDataCache(
                avdi, processor, cache_pil_images=True, max_size=data_cache_max_size
            )
            rollout_fn = make_vllm_rollout_func(
                full_model=full_model,
                data_cache=self._data_cache,
                rollout_temperature=rollout_temperature,
                rollout_top_p=rollout_top_p,
                rollout_max_generation_length=rollout_max_generation_length,
                vllm_mode=vllm_mode,
            )
            kwargs["rollout_func"] = rollout_fn

            # Workaround: Qwen3-VL's get_image_features returns a tuple
            # (image_embeds, deepstack_embeds) which the generic vLLM
            # transformers backend doesn't handle, crashing during
            # profiling.  Patch embed_multimodal to unpack the tuple.
            _patch_vllm_qwen3vl_embed()

        super().__init__(*args, **kwargs)
        self.full_model = full_model
        self.avdi = avdi
        self.rollout_temperature = rollout_temperature
        self.rollout_top_p = rollout_top_p
        self.rollout_max_generation_length = rollout_max_generation_length
        self.logprob_mini_batch_size = logprob_mini_batch_size
        self.use_vllm = use_vllm
        if not use_vllm:
            self._data_cache = ClipDataCache(
                avdi, self.processing_class, max_size=data_cache_max_size
            )

        # Value head (optional scene-level or segment-level baseline)
        self.value_head = None
        self.value_optimizer = None

        # Pending groups: queue of (h0, group_size) awaiting reward collection.
        # One entry per scene, appended during generation, drained during reward stashing.
        self._value_pending_groups: deque[tuple[torch.Tensor, int]] = deque()
        self._value_pending_rewards: list[float] = []

        # Ready pairs: scene-level (h0, V_mc) after Monte Carlo aggregation
        self._value_h0_stash: list[torch.Tensor] = []
        self._value_targets_stash: list[float] = []
        self._value_group_sizes_stash: list[int] = []
        self._value_reward_weights: list[float] = [0.5, 0.25, 0.25]  # traj/reasoning/consistency
        self._value_pretrain_remaining: int = 0
        self._value_save_path: str | None = None

        # Segment-level value head state
        self._segment_level: bool = False
        self._gamma: float = 1.0
        self._gae_lambda: float = 1.0
        self._traj_mask_ratio: float = 0.5
        self._advantage_enabled: bool = True
        # Segment-level stashes: per-completion hidden states and decomposed rewards
        self._segment_hidden_stash: list[dict] = []  # {h_obs, h_coc, h_traj, ...}
        self._segment_reward_stash: list[dict] = []  # {r_reason, r_consist, r_traj, r_traj_per_step}
        # Per-completion metadata for building (B, T) advantages after scoring
        self._completion_segment_map: list[dict] = []  # {coc_len, traj_len, ...}

        _vh_cfg = value_head_cfg or {}
        if _vh_cfg.get("enabled", False):
            from alpamayo_r1.training.value_head import SegmentValueHead

            _hidden_dim = int(_vh_cfg.get("hidden_dim", 4096))
            _vh_device = self.accelerator.device
            self.value_head = SegmentValueHead(_hidden_dim).to(_vh_device)
            self.value_optimizer = torch.optim.Adam(
                self.value_head.parameters(),
                lr=float(_vh_cfg.get("lr", 1e-4)),
            )
            self._value_pretrain_remaining = int(_vh_cfg.get("pretrain_steps", 0))
            self._value_save_path = _vh_cfg.get("save_path", None) or None
            self._segment_level = bool(_vh_cfg.get("segment_level", False))
            self._gamma = float(_vh_cfg.get("gamma", 1.0))
            self._gae_lambda = float(_vh_cfg.get("gae_lambda", 1.0))
            self._traj_mask_ratio = float(_vh_cfg.get("traj_mask_ratio", 0.5))
            self._advantage_enabled = bool(_vh_cfg.get("advantage_enabled", True))

            # Load pre-trained weights if a checkpoint path is provided
            _load_path = _vh_cfg.get("load_path", None) or None
            if _load_path is not None:
                import os

                if os.path.isfile(_load_path):
                    state = torch.load(_load_path, map_location=_vh_device)
                    self.value_head.load_state_dict(state, strict=False)
                    logger.info("Loaded SegmentValueHead weights from %s", _load_path)
                else:
                    logger.warning(
                        "value_head.load_path=%s not found — starting from random init",
                        _load_path,
                    )

            logger.info(
                "SegmentValueHead enabled: hidden_dim=%d, lr=%.2e, pretrain_steps=%d, "
                "segment_level=%s, gamma=%.2f, gae_lambda=%.2f, traj_mask_ratio=%.2f, device=%s",
                _hidden_dim,
                _vh_cfg.get("lr", 1e-4),
                self._value_pretrain_remaining,
                self._segment_level,
                self._gamma,
                self._gae_lambda,
                self._traj_mask_ratio,
                _vh_device,
            )

            # Segment-level advantages require hidden state extraction from the
            # teacher-forced forward pass, which only happens in the HF generation
            # path.  vLLM generates in a separate engine and never runs this pass.
            if self._segment_level and use_vllm:
                raise ValueError(
                    "value_head.segment_level=true is not compatible with vllm.enabled=true. "
                    "Segment-level advantages require hidden states from the teacher-forced "
                    "VLM forward pass, which is only available in the HuggingFace generation path. "
                    "Disable vLLM or use scene-level (segment_level=false) value head."
                )

        # Expert CFM fine-tuning (optional)
        self._expert_finetune_enabled = False
        self._expert_optimizer = None
        self._expert_rollout_data: list[dict] = []
        self._expert_finetune_cfg: dict = {}

        _ef_cfg = expert_finetune_cfg or {}
        if _ef_cfg.get("enabled", False):
            self._expert_finetune_enabled = True
            self._expert_finetune_cfg = _ef_cfg

            # Unfreeze expert + projection params
            for name, param in full_model.named_parameters():
                if any(
                    name.startswith(prefix)
                    for prefix in ("expert.", "action_in_proj.", "action_out_proj.")
                ):
                    param.requires_grad = True

            expert_params = [
                p
                for n, p in full_model.named_parameters()
                if any(
                    n.startswith(pfx) for pfx in ("expert.", "action_in_proj.", "action_out_proj.")
                )
                and p.requires_grad
            ]
            self._expert_optimizer = torch.optim.AdamW(
                expert_params,
                lr=float(_ef_cfg.get("lr", 1e-4)),
                weight_decay=float(_ef_cfg.get("weight_decay", 0.01)),
            )

            # Load pre-trained expert weights if provided
            _ef_load = _ef_cfg.get("load_path", None) or None
            if _ef_load is not None:
                import os

                if os.path.isfile(_ef_load):
                    state = torch.load(_ef_load, map_location="cpu")
                    if "expert" in state:
                        full_model.expert.load_state_dict(state["expert"])
                    if "action_in_proj" in state:
                        full_model.action_in_proj.load_state_dict(state["action_in_proj"])
                    if "action_out_proj" in state:
                        full_model.action_out_proj.load_state_dict(state["action_out_proj"])
                    logger.info("Loaded expert weights from %s", _ef_load)
                else:
                    logger.warning(
                        "expert_finetune.load_path=%s not found — using defaults", _ef_load
                    )

            n_expert_params = sum(p.numel() for p in expert_params)
            logger.info(
                "Expert CFM fine-tuning enabled: lr=%.2e, %d trainable params, "
                "rollouts_per_step=%d, every_n_steps=%d",
                _ef_cfg.get("lr", 1e-4),
                n_expert_params,
                _ef_cfg.get("rollouts_per_step", 1),
                _ef_cfg.get("every_n_steps", 1),
            )

        # Override EOS token so TRL metrics (clipped_ratio, terminated_length)
        # and the EOS mask in _generate_completions recognise <|traj_future_end|>
        # as a valid termination token instead of only the default <|endoftext|>.
        traj_future_end_id = full_model.special_token_ids["traj_future_end"]
        self.eos_token_id = traj_future_end_id

        # Patch FSDP1 weight sync on the VLLMGeneration instance.
        # TRL's default _sync_fsdp1_params_to_vllm does a post-order
        # traversal calling FSDP.summon_full_params on each sub-module.
        # This triggers _lazy_init on children before the root, corrupting
        # PyTorch FSDP's _is_root hierarchy and raising:
        #   AssertionError: Non-root FSDP instance's '_is_root' should not
        #   have been set yet or should have been set to 'False'
        # Fix: use a single root-level summon_full_params(recurse=True)
        # which initialises the root first, then gathers all params at once.
        if use_vllm and self.is_fsdp_enabled and hasattr(self, "vllm_generation"):
            vllm_gen = self.vllm_generation

            def _patched_sync_fsdp1(module, prefix="", visited=None):
                with FSDP.summon_full_params(module, recurse=True, writeback=False):
                    for name, param in module.named_parameters():
                        name = vllm_gen._fix_param_name_to_vllm(
                            name, extra_prefixes=["_fsdp_wrapped_module."]
                        )
                        if vllm_gen.mode == "server" and vllm_gen.accelerator.is_main_process:
                            vllm_gen.vllm_client.update_named_param(name, param.data)
                        elif vllm_gen.mode == "colocate":
                            llm_model = vllm_gen.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

            vllm_gen._sync_fsdp1_params_to_vllm = _patched_sync_fsdp1
            logger.info(
                "Patched VLLMGeneration._sync_fsdp1_params_to_vllm for FSDP root-level sync."
            )

    def _compute_scene_h0(
        self,
        prompt_input_ids: torch.Tensor,
        model_inputs: dict,
        device: torch.device,
    ) -> torch.Tensor:
        """Get VLM hidden state at last prompt token (scene encoding).

        Runs a single VLM forward pass with output_hidden_states=True on the
        prompt only. The hidden state at the last prompt position encodes the
        model's full scene understanding before any generation begins.

        Args:
            prompt_input_ids: Prompt token IDs, shape (1, L_prompt).
            model_inputs: Dict from ClipDataCache, containing 'tokenized_data'
                with pixel_values, attention_mask, image_grid_thw.
            device: Target compute device.

        Returns:
            h_0: shape (1, hidden_dim), float32, on CPU.
        """
        tokenized = model_inputs["tokenized_data"]
        forward_kwargs = {k: v for k, v in tokenized.items() if k not in ("input_ids",)}
        with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
            outputs = self.full_model.vlm(
                input_ids=prompt_input_ids,
                output_hidden_states=True,
                **forward_kwargs,
            )
        # Last hidden layer, last token position: (1, hidden_dim)
        h0 = outputs.hidden_states[-1][:, -1, :].float().cpu()
        return h0

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Override to mutate the ``logs`` dict in-place during eval.

        TRL's GRPOTrainer.log() merges reward metrics into a **new** dict via
        ``logs = {**logs, **metrics}``, so the original ``output.metrics``
        dict (passed by ``evaluate()``) never receives reward metrics.
        Downstream consumers — ``_determine_best_metric`` and the
        ``EarlyStoppingCallback.on_evaluate`` — then fail to find reward
        metrics like ``eval_rewards/trajectory_quality_reward/mean``.

        We fix this by updating the original ``logs`` dict in-place with the
        pending eval metrics *before* delegating to TRL's ``log()``.
        """
        mode = "train" if self.model.training else "eval"
        if mode == "eval" and self._metrics[mode]:
            extra = {f"eval_{k}": sum(v) / len(v) for k, v in self._metrics[mode].items() if v}
            logs.update(extra)
        super().log(logs, start_time)

    def _save(self, output_dir, state_dict=None):
        """Extend default save to also persist value head and expert weights.

        Saves both a step-numbered checkpoint (e.g. ``value_head_step200.pt``)
        and the fixed ``save_path`` as a "latest" pointer.
        """
        super()._save(output_dir, state_dict=state_dict)

        if self.value_head is not None and self._value_save_path is not None:
            import os

            vh_state = self.value_head.state_dict()
            save_dir = os.path.dirname(os.path.abspath(self._value_save_path))
            os.makedirs(save_dir, exist_ok=True)

            torch.save(vh_state, self._value_save_path)
            logger.info("Saved SceneValueHead weights to %s", self._value_save_path)

            base, ext = os.path.splitext(self._value_save_path)
            step = self.state.global_step
            step_path = f"{base}_step{step}{ext}"
            torch.save(vh_state, step_path)
            logger.info("Saved SceneValueHead checkpoint to %s", step_path)

        if self._expert_finetune_enabled:
            import os

            expert_save_path = self._expert_finetune_cfg.get("save_path", None) or None
            # Always save expert checkpoint alongside the VLM checkpoint
            expert_ckpt = {
                "expert": self.full_model.expert.state_dict(),
                "action_in_proj": self.full_model.action_in_proj.state_dict(),
                "action_out_proj": self.full_model.action_out_proj.state_dict(),
                "global_step": self.state.global_step,
            }
            ckpt_path = os.path.join(output_dir, "expert_checkpoint.pt")
            torch.save(expert_ckpt, ckpt_path)
            logger.info("Saved expert checkpoint to %s", ckpt_path)

            if expert_save_path is not None:
                save_dir = os.path.dirname(os.path.abspath(expert_save_path))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(expert_ckpt, expert_save_path)
                logger.info("Saved expert weights to %s", expert_save_path)

    def _generate_and_score_completions(self, inputs):
        """Override to inject segment-level (B, T) advantages.

        Delegates to the parent for generation, reward computation, and scalar
        advantage calculation. When segment_level is enabled and the value head
        is active, replaces the scalar advantages with per-token segment-level
        advantages computed via the three-level value head and GAE.
        """
        output = super()._generate_and_score_completions(inputs)

        if self.value_head is not None and self._segment_level and self._segment_hidden_stash:
            completion_mask = output.get("completion_mask")
            if completion_mask is None:
                return output

            # Retrieve rewards_per_func from the most recent _calculate_rewards call
            # (stashed as self._last_rewards_per_func by our override)
            rewards_per_func = getattr(self, "_last_rewards_per_func", None)
            if rewards_per_func is None:
                return output

            num_generations = (
                self.num_generations if self.model.training else self.num_generations_eval
            )
            # Always compute segment advantages (which also trains the value head).
            # During pretrain, the advantages are computed but NOT injected into the
            # output — the policy receives the original GRPO advantages.
            segment_advantages = self._compute_segment_advantages(
                completion_mask, rewards_per_func, num_generations
            )
            is_pretrain = self._value_pretrain_remaining > 0
            if segment_advantages is not None and self._advantage_enabled and not is_pretrain:
                output["advantages"] = segment_advantages

        return output

    def _generate_single_turn(self, prompts: list):
        """Generate CoC text + discrete trajectory tokens via VLM only.

        When ``use_vllm=True``, delegates to the parent's vLLM path which
        calls our ``rollout_func``.  Otherwise uses HuggingFace
        ``model.generate()`` directly.

        Args:
            prompts: List of prompt strings or message lists (B*G items,
                with ``num_generations`` duplicates per unique prompt).

        Returns:
            Tuple of (prompt_ids, completion_ids, logprobs, extra_fields)
            matching TRL's internal interface.
        """
        if self.use_vllm:
            return super()._generate_single_turn(prompts)

        # Clear expert rollout data at the start of each new rollout
        if self._expert_finetune_enabled:
            self._expert_rollout_data.clear()

        # Clear value head stashes at the start of each new rollout
        if self.value_head is not None:
            self._value_pending_groups.clear()
            self._value_pending_rewards.clear()
            self._value_h0_stash.clear()
            self._value_targets_stash.clear()
            self._value_group_sizes_stash.clear()
            self._segment_hidden_stash.clear()
            self._segment_reward_stash.clear()
            self._completion_segment_map.clear()

        device = self.accelerator.device
        num_generations = self.num_generations if self.model.training else self.num_generations_eval

        # Model config for trajectory token handling
        traj_token_start_idx = self.full_model.future_token_start_idx
        tokens_per_future_traj = self.full_model.config.tokens_per_future_traj
        traj_vocab_size = self.full_model.config.traj_vocab_size
        special_token_ids = self.full_model.special_token_ids
        traj_tokenizer = self.full_model.traj_tokenizer
        pad_token_id = self.processing_class.tokenizer.pad_token_id

        # De-duplicate prompts: TRL repeats each prompt, but distributed
        # training may split repetitions across GPUs.  Detect actual local
        # repeat counts instead of assuming num_generations copies are here.
        prompt_groups = _group_consecutive_prompts(prompts)

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_pred_xyz: list[list[float]] = []
        all_gt_xyz: list[list[float]] = []
        all_coc_texts: list[str] = []
        all_clip_ids: list[str] = []

        # Stop generation at <|traj_future_end|>
        traj_future_end_id = special_token_ids["traj_future_end"]

        # FSDP-aware generation: unwrap_model_for_generation handles
        # gradient checkpointing disable/enable automatically.
        # summon_full_params gathers sharded VLM weights so the full
        # pipeline (accessed via self.full_model) sees complete params.
        fsdp_ctx = (
            FSDP.summon_full_params(self.model_wrapped, recurse=False)
            if self.is_fsdp_enabled
            else nullcontext()
        )
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator):
            with torch.no_grad(), fsdp_ctx:
                for prompt, local_gen_count in prompt_groups:
                    # Resolve prompt to string if conversational
                    if isinstance(prompt, list):
                        prompt_text = " ".join(
                            m.get("content", "")
                            if isinstance(m.get("content"), str)
                            else " ".join(
                                c.get("text", "")
                                for c in m.get("content", [])
                                if isinstance(c, dict)
                            )
                            for m in prompt
                        )
                    else:
                        prompt_text = prompt

                    clip_id, t0_us = _parse_clip_metadata(prompt_text)

                    # 1. Load driving data and prepare model inputs (cached in CPU RAM)
                    model_inputs, ego_future_xyz = self._data_cache.get(clip_id, t0_us, device)

                    # 2. Fuse history trajectory tokens into input_ids
                    tokenized = model_inputs["tokenized_data"]
                    input_ids = tokenized.pop("input_ids")
                    traj_data = {
                        "ego_history_xyz": model_inputs["ego_history_xyz"],
                        "ego_history_rot": model_inputs["ego_history_rot"],
                    }
                    input_ids = self.full_model.fuse_traj_tokens(input_ids, traj_data)
                    prompt_len = input_ids.shape[1]
                    prompt_input_ids = input_ids.clone()

                    # Compute scene h_0 once per unique scene for value head
                    if self.value_head is not None:
                        scene_h0 = self._compute_scene_h0(prompt_input_ids, model_inputs, device)
                        self._value_pending_groups.append((scene_h0, local_gen_count))

                    # 3. VLM-only generation (no ExpertLogitsProcessor)
                    # Generate only as many sequences as this GPU needs for this
                    # prompt group (may be less than num_generations when
                    # generations are distributed across GPUs).
                    gen_config = GenerationConfig(
                        do_sample=True,
                        temperature=self.rollout_temperature,
                        top_p=self.rollout_top_p,
                        num_return_sequences=local_gen_count,
                        max_new_tokens=self.rollout_max_generation_length
                        + tokens_per_future_traj
                        + 10,
                        pad_token_id=pad_token_id,
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
                            **tokenized,  # pixel_values, attention_mask, image_grid_thw
                        )

                    # vlm_output: (local_gen_count, prompt_len + generated_len)
                    generated_seqs = vlm_output[:, prompt_len:]

                    # 4. Extract trajectory tokens and decode to continuous xyz
                    traj_tokens = extract_traj_tokens(
                        vlm_output,
                        special_token_ids,
                        tokens_per_future_traj,
                        traj_token_start_idx,
                        traj_vocab_size,
                    )
                    hist_xyz = model_inputs["ego_history_xyz"][:, -1]  # (1, T, 3)
                    hist_rot = model_inputs["ego_history_rot"][:, -1]  # (1, T, 3, 3)
                    hist_xyz_rep = hist_xyz.expand(local_gen_count, -1, -1)
                    hist_rot_rep = hist_rot.expand(local_gen_count, -1, -1, -1)
                    pred_xyz_tensor, pred_rot_tensor, _ = traj_tokenizer.decode(
                        hist_xyz_rep,
                        hist_rot_rep,
                        traj_tokens,
                    )

                    # 5. Build per-sample outputs (local_gen_count per unique prompt)
                    prompt_ids_list = prompt_input_ids[0].cpu().tolist()
                    traj_future_start_id = special_token_ids["traj_future_start"]

                    for sample_idx in range(local_gen_count):
                        # Completion = all generated tokens up to <|traj_future_end|>
                        # (trim trailing pad and extra token from StopAfterEOS)
                        raw_completion = generated_seqs[sample_idx].cpu().tolist()
                        completion_ids: list[int] = []
                        for tid in raw_completion:
                            completion_ids.append(tid)
                            if tid == traj_future_end_id:
                                break
                        # Fallback: strip trailing pad tokens if no end marker found
                        if traj_future_end_id not in completion_ids:
                            while completion_ids and completion_ids[-1] == pad_token_id:
                                completion_ids.pop()
                        # TRL requires at least one token
                        if not completion_ids:
                            completion_ids = [self.processing_class.tokenizer.eos_token_id]

                        # Extract CoC text: decode tokens before <|traj_future_start|>.
                        # This is robust regardless of whether <|cot_end|> is generated.
                        try:
                            traj_start_pos = raw_completion.index(traj_future_start_id)
                        except ValueError:
                            traj_start_pos = len(raw_completion)
                        coc_text = self.full_model.tokenizer.decode(
                            raw_completion[:traj_start_pos], skip_special_tokens=True
                        ).strip()

                        all_prompt_ids.append(prompt_ids_list)
                        all_completion_ids.append(completion_ids)
                        all_coc_texts.append(coc_text)
                        all_clip_ids.append(clip_id)

                        # Stash rollout metadata for expert CFM training.
                        # completion_prefix = tokens up to <|traj_future_start|>
                        # (the expert only needs the reasoning trace as conditioning).
                        if self._expert_finetune_enabled:
                            traj_start_id = special_token_ids["traj_future_start"]
                            try:
                                prefix_end = raw_completion.index(traj_start_id) + 1
                            except ValueError:
                                prefix_end = len(raw_completion)
                            self._expert_rollout_data.append(
                                {
                                    "clip_id": clip_id,
                                    "t0_us": t0_us,
                                    "completion_prefix": raw_completion[:prefix_end],
                                }
                            )

                        # Trajectory data for reward computation
                        pred_traj = pred_xyz_tensor[sample_idx].cpu().numpy().flatten().tolist()
                        gt_traj = ego_future_xyz[0, 0].numpy().flatten().tolist()
                        all_pred_xyz.append(pred_traj)
                        all_gt_xyz.append(gt_traj)

                    # 7. Compute log-probs via teacher-forced VLM forward (batched)
                    # When segment-level value head is enabled, piggyback on this
                    # forward pass to extract hidden states at segment boundaries.
                    need_hidden = self.value_head is not None and self._segment_level
                    logprob_result = _compute_batch_logprobs(
                        self.full_model,
                        model_inputs,
                        prompt_input_ids,
                        all_completion_ids[-local_gen_count:],
                        prompt_len,
                        device,
                        mini_batch_size=self.logprob_mini_batch_size,
                        output_hidden_states=need_hidden,
                    )
                    if need_hidden:
                        batch_logprobs, batch_hidden = logprob_result
                        # Extract segment-level hidden states for each completion
                        recent_comp_ids = all_completion_ids[-local_gen_count:]
                        for comp_idx in range(local_gen_count):
                            self._extract_and_stash_segment_hidden(
                                batch_hidden[comp_idx],
                                recent_comp_ids[comp_idx],
                                prompt_len,
                            )
                    else:
                        batch_logprobs = logprob_result
                    all_logprobs.extend(batch_logprobs)

        extra_fields = {
            "pred_xyz": all_pred_xyz,
            "gt_xyz": all_gt_xyz,
        }

        # Stash rollout data for the logging callback (near-zero overhead).
        self._rollout_log_data = {
            "coc_texts": all_coc_texts,
            "clip_ids": all_clip_ids,
            "pred_xyz": all_pred_xyz,
            "gt_xyz": all_gt_xyz,
            "completion_ids": all_completion_ids,
            "num_generations": num_generations,
        }

        return all_prompt_ids, all_completion_ids, all_logprobs, extra_fields

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Override to decode trajectory tokens from vLLM completions.

        In the vLLM path, ``rollout_func`` returns ``hist_xyz``, ``hist_rot``,
        and ``clip_ids`` as extra fields (injected into ``inputs`` by TRL).
        We extract trajectory tokens from ``completion_ids``, decode them to
        continuous xyz via ``traj_tokenizer``, and set ``pred_xyz`` before
        delegating to the parent's reward computation.

        For the HF path, ``pred_xyz`` is already in ``inputs`` (set by
        ``_generate_single_turn``), so we just pass through.
        """
        if not self.use_vllm or not inputs or "hist_xyz" not in inputs[0]:
            result = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
            if self.value_head is not None:
                self._stash_value_rewards(inputs, completions)
            self._last_rewards_per_func = result
            return result

        # vLLM path: decode trajectory tokens from completion_ids
        traj_token_start_idx = self.full_model.future_token_start_idx
        tokens_per_future_traj = self.full_model.config.tokens_per_future_traj
        traj_vocab_size = self.full_model.config.traj_vocab_size
        special_token_ids = self.full_model.special_token_ids
        traj_tokenizer = self.full_model.traj_tokenizer
        num_generations = self.num_generations if self.model.training else self.num_generations_eval

        all_pred_xyz: list[list[float]] = []
        all_coc_texts: list[str] = []
        all_clip_ids: list[str] = []

        device = self.accelerator.device

        for i, inp in enumerate(inputs):
            comp_ids = completion_ids_list[i]
            comp_tensor = torch.tensor(comp_ids, dtype=torch.long, device=device).unsqueeze(0)

            # Extract trajectory tokens
            traj_tokens = extract_traj_tokens(
                comp_tensor,
                special_token_ids,
                tokens_per_future_traj,
                traj_token_start_idx,
                traj_vocab_size,
            )

            # Reconstruct hist_xyz/hist_rot from flattened lists
            hist_xyz_flat = inp["hist_xyz"]
            hist_rot_flat = inp["hist_rot"]
            hist_xyz = torch.tensor(hist_xyz_flat, dtype=torch.float32, device=device)
            hist_rot = torch.tensor(hist_rot_flat, dtype=torch.float32, device=device)

            # Reshape — hist_xyz is (T*3,) flattened, hist_rot is (T*3*3,) flattened
            n_hist = hist_xyz.numel() // 3
            hist_xyz = hist_xyz.reshape(1, n_hist, 3)
            hist_rot = hist_rot.reshape(1, n_hist, 3, 3)

            # Decode trajectory tokens → continuous xyz
            pred_xyz_tensor, _, _ = traj_tokenizer.decode(
                hist_xyz,
                hist_rot,
                traj_tokens,
            )

            pred_traj = pred_xyz_tensor[0].cpu().numpy().flatten().tolist()
            all_pred_xyz.append(pred_traj)

            # Set pred_xyz on the input dict for reward functions
            inp["pred_xyz"] = pred_traj

            # Collect for logging
            all_clip_ids.append(inp.get("clip_ids", ""))

        # Extract CoC text from completions (text strings from TRL)
        for comp in completions:
            text = comp if isinstance(comp, str) else ""
            all_coc_texts.append(text)

        # Clean up vLLM-only fields that reward functions don't expect
        for inp in inputs:
            inp.pop("hist_xyz", None)
            inp.pop("hist_rot", None)
            inp.pop("clip_ids", None)

        # Stash rollout data for the logging callback
        self._rollout_log_data = {
            "coc_texts": all_coc_texts,
            "clip_ids": all_clip_ids,
            "pred_xyz": all_pred_xyz,
            "gt_xyz": [inp.get("gt_xyz", []) for inp in inputs],
            "completion_ids": completion_ids_list,
            "num_generations": num_generations,
        }

        result = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if self.value_head is not None:
            self._stash_value_rewards(inputs, completions)
        self._last_rewards_per_func = result
        return result

    def _extract_and_stash_segment_hidden(
        self,
        hidden_states: torch.Tensor,
        completion_ids: list[int],
        prompt_len: int,
    ) -> None:
        """Extract hidden states at segment boundaries from teacher-forced pass.

        The hidden_states tensor is a slice of the last VLM layer's output,
        shape (1, 1+comp_len, D). Index 0 corresponds to the last prompt
        token (h_obs). Index 1+j corresponds to completion_ids[j] — the
        j-th completion token (0-indexed).

        Finds <cot_end> and trajectory token positions, extracts h_obs, h_coc,
        h_traj, and stashes metadata about segment boundaries.

        Args:
            hidden_states: (1, 1+comp_len, D) CPU float32 tensor. Sliced from
                full-sequence positions [prompt_len-1 .. prompt_len+comp_len-1].
            completion_ids: List of completion token IDs.
            prompt_len: Number of prompt tokens (unused, kept for API clarity).
        """
        special_ids = self.full_model.special_token_ids
        cot_end_id = special_ids["cot_end"]
        traj_future_start_id = special_ids["traj_future_start"]
        traj_token_start_idx = self.full_model.future_token_start_idx
        traj_vocab_size = self.full_model.config.traj_vocab_size

        # hidden_states shape: (1, 1+comp_len, D)
        # Index 0 = last prompt token = h_obs
        h_obs = hidden_states[0, 0:1, :]  # (1, D)

        # Find <cot_end> position in completion_ids
        cot_end_offset = None
        for idx, tid in enumerate(completion_ids):
            if tid == cot_end_id:
                cot_end_offset = idx
                break

        # h_coc: hidden state at <cot_end> position
        # In hidden_states, completion position idx maps to hidden_states[0, idx+1]
        if cot_end_offset is not None:
            h_coc = hidden_states[0, cot_end_offset + 1 : cot_end_offset + 2, :]  # (1, D)
        else:
            h_coc = h_obs  # fallback: use h_obs if no CoC segment

        # Find trajectory token positions
        traj_positions = []
        for idx, tid in enumerate(completion_ids):
            if traj_token_start_idx <= tid < traj_token_start_idx + traj_vocab_size:
                traj_positions.append(idx)

        if traj_positions:
            # h_traj: hidden states at curvature token positions only (stride-2)
            traj_indices = [p + 1 for p in traj_positions]  # +1 for the h_obs offset
            h_traj_all = hidden_states[0, traj_indices, :]  # (T_traj, D)
            curv_sel = list(range(1, len(traj_positions), 2))
            h_traj = h_traj_all[curv_sel]  # (T_curv, D)
        else:
            h_traj = torch.zeros(0, hidden_states.shape[-1])

        # Find traj_future_start position — CoC tokens are between start of completion
        # and traj_future_start (exclusive of special tokens)
        traj_start_offset = None
        for idx, tid in enumerate(completion_ids):
            if tid == traj_future_start_id:
                traj_start_offset = idx
                break

        coc_len = traj_start_offset if traj_start_offset is not None else len(completion_ids)
        traj_len = len(traj_positions)

        self._segment_hidden_stash.append(
            {
                "h_obs": h_obs,  # (1, D)
                "h_coc": h_coc,  # (1, D)
                "h_traj": h_traj,  # (T_traj, D)
            }
        )
        self._completion_segment_map.append(
            {
                "coc_len": coc_len,
                "traj_len": traj_len,
                "traj_positions": traj_positions,
                "total_len": len(completion_ids),
            }
        )

    def _compute_segment_advantages(
        self,
        completion_mask: torch.Tensor,
        rewards_per_func: torch.Tensor,
        num_generations: int,
    ) -> torch.Tensor:
        """Compute (B, T) segment-level advantages using the value head.

        For each completion (return-minus-baseline at each level):
        - CoC tokens get: A_coc = G(s_coc) - V(s_coc)
        - Trajectory tokens get: A_traj_t via GAE(gamma, lambda) on per-timestep rewards
        - Random masking of trajectory positions to balance gradient contribution

        Also trains the value head on the three-level targets and logs metrics.

        Args:
            completion_mask: (B, T) binary mask for valid completion tokens.
            rewards_per_func: (B, num_reward_funcs) per-function rewards.
            num_generations: G generations per prompt.

        Returns:
            (B, T) advantage tensor on the same device as completion_mask.
        """
        from alpamayo_r1.training.value_head import SegmentValueHead

        device = completion_mask.device
        B, T = completion_mask.shape
        advantages = torch.zeros(B, T, device=device)

        if not self._segment_hidden_stash or len(self._segment_hidden_stash) != B:
            logger.warning(
                "Segment hidden stash size %d != batch size %d — falling back to scalar advantage",
                len(self._segment_hidden_stash),
                B,
            )
            return None

        # Reward function order must match train_grpo.py:
        #   index 0 = trajectory_quality_reward
        #   index 1 = reasoning_quality_reward
        #   index 2 = consistency_reward
        # The weights below are unpacked in the same order.
        if rewards_per_func.shape[1] < 2:
            logger.warning(
                "rewards_per_func has %d columns, need at least 2 — falling back",
                rewards_per_func.shape[1],
            )
            return None
        w_traj, w_reason, w_consist = self._value_reward_weights
        mode = "train" if self.model.training else "eval"
        vh_device = next(self.value_head.parameters()).device

        # Collect segment-level value targets and train the value head
        all_v_obs = []
        all_v_coc = []
        all_v_traj = []  # list of (T_traj,) tensors
        all_g_obs = []
        all_g_coc = []
        all_g_traj = []  # list of (T_traj,) tensors

        for i in range(B):
            seg = self._segment_hidden_stash[i]
            seg_map = self._completion_segment_map[i]

            h_obs = seg["h_obs"].to(vh_device)  # (1, D)
            h_coc = seg["h_coc"].to(vh_device)  # (1, D)
            h_traj = seg["h_traj"].to(vh_device)  # (T_traj, D)

            # Value predictions (detached for advantage computation, will backprop in training)
            # Trajectory V is evaluated at curvature positions only (idx % 2 == 1),
            # then expanded to per-token for GAE compatibility. Both tokens in each
            # (acceleration, curvature) pair receive the same V since the curvature
            # token is the natural evaluation point for the timestep.
            with torch.no_grad():
                v_obs = self.value_head(h_obs, level=SegmentValueHead.LEVEL_OBS)  # (1,)
                v_coc = self.value_head(h_coc, level=SegmentValueHead.LEVEL_COC)  # (1,)
                if h_traj.shape[0] > 0:
                    n_curv = h_traj.shape[0]
                    traj_pos = (torch.arange(n_curv, device=vh_device) + SegmentValueHead.POS_TRAJ_START).unsqueeze(0)
                    v_traj_ts = self.value_head(
                        h_traj.unsqueeze(0),
                        level=SegmentValueHead.LEVEL_TRAJ,
                        positions=traj_pos,
                    ).squeeze(0)  # (n_curv,)
                    # Expand to per-token: both tokens in a pair get same V
                    T_tokens = seg_map["traj_len"]
                    v_traj = v_traj_ts.repeat_interleave(2)
                    if v_traj.shape[0] < T_tokens:  # odd T_traj
                        v_traj = torch.cat([v_traj, v_traj_ts[-1:]])
                else:
                    v_traj = torch.zeros(0, device=vh_device)

            # Per-function rewards for this sample
            r_traj_scalar = rewards_per_func[i, 0].item()  # trajectory
            r_reason = rewards_per_func[i, 1].item()  # reasoning
            r_consist = rewards_per_func[i, 2].item() if rewards_per_func.shape[1] > 2 else 0.0

            # Compute per-timestep trajectory rewards if we have stashed data
            T_traj = seg_map["traj_len"]
            if T_traj > 0 and i < len(self._segment_reward_stash):
                seg_rew = self._segment_reward_stash[i]
                r_per_step = seg_rew.get("r_traj_per_step")
                if r_per_step is None or len(r_per_step) != T_traj:
                    # Uniform fallback: spread scalar reward evenly
                    r_per_step = [r_traj_scalar] * T_traj
            elif T_traj > 0:
                r_per_step = [r_traj_scalar] * T_traj
            else:
                r_per_step = []

            # --- Value targets (returns-to-go) ---
            if T_traj > 0:
                g_traj = compute_trajectory_returns(
                    r_per_step, w_traj, w_consist, r_consist,
                )
                traj_total = g_traj[0].item()
                all_g_traj.append(g_traj)
                all_v_traj.append(v_traj.cpu())
            else:
                traj_total = 0.0
                all_g_traj.append(torch.zeros(0))
                all_v_traj.append(torch.zeros(0))

            g_obs = w_reason * r_reason + traj_total  # total return
            g_coc = traj_total  # remaining after CoC

            all_g_obs.append(g_obs)
            all_g_coc.append(g_coc)
            all_v_obs.append(v_obs.item())
            all_v_coc.append(v_coc.item())

            # --- Advantage computation ---
            # CoC advantage: return-minus-baseline at the CoC information level.
            # G(s_coc) = traj_total (R_reasoning already collected at s_coc).
            # A_coc = G(s_coc) - V(s_coc)
            a_coc = g_coc - v_coc.item()

            # Trajectory advantages: GAE(gamma, lambda)
            if T_traj > 0:
                a_traj = _compute_gae(r_traj_weighted, v_traj, self._gamma, self._gae_lambda)

                # Random masking of trajectory positions
                if self._traj_mask_ratio > 0 and self.model.training:
                    mask = torch.rand(T_traj, device=device) > self._traj_mask_ratio
                    # Always keep at least one traj position
                    if not mask.any():
                        mask[0] = True
                    a_traj_masked = a_traj.to(device) * mask.float()
                else:
                    a_traj_masked = a_traj.to(device)
            else:
                a_traj_masked = torch.zeros(0, device=device)

            # Fill the (B, T) advantage tensor
            coc_len = seg_map["coc_len"]
            traj_positions = seg_map["traj_positions"]

            # CoC tokens: positions 0..coc_len-1
            if coc_len > 0 and coc_len <= T:
                advantages[i, :coc_len] = a_coc

            # Trajectory tokens: at their specific positions
            for t_idx, pos in enumerate(traj_positions):
                if pos < T and t_idx < len(a_traj_masked):
                    advantages[i, pos] = a_traj_masked[t_idx]

        # --- Train value head on all three levels ---
        self._train_segment_value_head(
            all_v_obs,
            all_v_coc,
            all_v_traj,
            all_g_obs,
            all_g_coc,
            all_g_traj,
        )

        # --- Log metrics ---
        if B > 0:
            coc_advs = [all_g_coc[i] - all_v_coc[i] for i in range(B)]
            self._metrics[mode]["advantages/coc_mean"].append(float(np.mean(coc_advs)))
        if any(m["traj_len"] > 0 for m in self._completion_segment_map):
            traj_adv_vals = []
            for i in range(B):
                traj_pos = self._completion_segment_map[i]["traj_positions"]
                for pos in traj_pos:
                    if pos < T:
                        traj_adv_vals.append(advantages[i, pos].item())
            if traj_adv_vals:
                self._metrics[mode]["advantages/traj_mean"].append(float(np.mean(traj_adv_vals)))

        self._metrics[mode]["advantages/mean"].append(
            advantages[completion_mask.bool()].mean().item() if completion_mask.any() else 0.0
        )
        self._metrics[mode]["advantages/std"].append(
            advantages[completion_mask.bool()].std().item() if completion_mask.sum() > 1 else 0.0
        )
        self._metrics[mode]["advantages/v_baseline_mean"].append(float(np.mean(all_v_obs)))

        return advantages

    def _train_segment_value_head(
        self,
        all_v_obs: list[float],
        all_v_coc: list[float],
        all_v_traj: list[torch.Tensor],
        all_g_obs: list[float],
        all_g_coc: list[float],
        all_g_traj: list[torch.Tensor],
    ) -> None:
        """Train the value head on all three levels jointly.

        Uses per-completion targets for obs and coc levels, and sub-samples
        trajectory tokens to prevent them from dominating the loss.
        """
        from alpamayo_r1.training.value_head import SegmentValueHead

        B = len(all_g_obs)
        if B == 0:
            return

        vh_device = next(self.value_head.parameters()).device
        mode = "train" if self.model.training else "eval"

        # Obs-level loss: train on per-completion (h_obs, g_obs) pairs
        obs_preds = []
        obs_targets = []
        for i in range(B):
            h_obs = self._segment_hidden_stash[i]["h_obs"].to(vh_device)
            v = self.value_head(h_obs, level=SegmentValueHead.LEVEL_OBS)
            obs_preds.append(v)
            obs_targets.append(all_g_obs[i])

        obs_pred_t = torch.cat(obs_preds)
        obs_target_t = torch.tensor(obs_targets, device=vh_device, dtype=torch.float32)
        loss_obs = F.mse_loss(obs_pred_t, obs_target_t)

        # CoC-level loss
        coc_preds = []
        coc_targets = []
        for i in range(B):
            h_coc = self._segment_hidden_stash[i]["h_coc"].to(vh_device)
            v = self.value_head(h_coc, level=SegmentValueHead.LEVEL_COC)
            coc_preds.append(v)
            coc_targets.append(all_g_coc[i])

        coc_pred_t = torch.cat(coc_preds)
        coc_target_t = torch.tensor(coc_targets, device=vh_device, dtype=torch.float32)
        loss_coc = F.mse_loss(coc_pred_t, coc_target_t)

        # Traj-level loss: sub-sample at curvature positions only.
        # Trajectory tokens come in (acceleration, curvature) pairs per timestep.
        # We evaluate V only at curvature positions (idx % 2 == 1) because the
        # curvature token has seen both components and rewards are per-timestep.
        traj_preds = []
        traj_targets = []
        max_traj_samples = max(B * 2, 16)  # limit traj samples relative to obs+coc
        total_traj = 0

        for i in range(B):
            h_traj = self._segment_hidden_stash[i]["h_traj"]  # (T_curv, D)
            g_traj = all_g_traj[i]  # (T_curv,)
            if h_traj.shape[0] == 0:
                continue
            n_curv = h_traj.shape[0]
            n_keep = min(n_curv, max(1, max_traj_samples - total_traj))
            if n_keep < n_curv:
                sel = torch.randperm(n_curv)[:n_keep]
            else:
                sel = torch.arange(n_curv)
            h_sub = h_traj[sel].to(vh_device)
            g_sub = g_traj[sel].to(vh_device)
            traj_pos = (sel + SegmentValueHead.POS_TRAJ_START).unsqueeze(0).to(vh_device)
            v = self.value_head(
                h_sub.unsqueeze(0), level=SegmentValueHead.LEVEL_TRAJ, positions=traj_pos
            ).squeeze(0)
            traj_preds.append(v)
            traj_targets.append(g_sub)
            total_traj += n_keep

        if traj_preds:
            traj_pred_t = torch.cat(traj_preds)
            traj_target_t = torch.cat(traj_targets)
            loss_traj = F.mse_loss(traj_pred_t, traj_target_t)
        else:
            loss_traj = torch.tensor(0.0, device=vh_device)

        # Combined loss (equal weight for now)
        total_loss = (loss_obs + loss_coc + loss_traj) / 3.0

        self.value_optimizer.zero_grad()
        total_loss.backward()
        self.value_optimizer.step()

        # Log per-level losses
        self._metrics[mode]["value_head/loss"].append(total_loss.item())
        self._metrics[mode]["value_head/loss_obs"].append(loss_obs.item())
        self._metrics[mode]["value_head/loss_coc"].append(loss_coc.item())
        self._metrics[mode]["value_head/loss_traj"].append(loss_traj.item())
        self._metrics[mode]["value_head/pred_mean"].append(obs_pred_t.detach().mean().item())
        self._metrics[mode]["value_head/target_mean"].append(obs_target_t.mean().item())

    def _stash_value_rewards(self, inputs: list[dict], completions: list[str]) -> None:
        """Compute composite rewards and stash them for value head training.

        Per-sample composite rewards are accumulated into the pending buffer.
        When the buffer reaches the expected group size, rewards are aggregated
        into a Monte Carlo target ``V_mc = mean(rewards)`` and flushed to the
        ready stash as a single ``(h0, V_mc)`` pair.

        When segment_level is enabled, also decomposes trajectory rewards into
        per-timestep values and stashes them for segment-level advantage computation.

        Args:
            inputs: Per-sample input dicts (must have pred_xyz and gt_xyz populated).
            completions: CoC text strings, one per sample.
        """
        from alpamayo_r1.training.rewards import (
            consistency_reward,
            reasoning_quality_reward,
            trajectory_per_timestep_rewards,
            trajectory_quality_reward,
        )

        pred_xyz_list = [inp.get("pred_xyz") for inp in inputs]
        gt_xyz_list = [inp.get("gt_xyz") for inp in inputs]
        w_traj, w_reason, w_consist = self._value_reward_weights

        r_traj = trajectory_quality_reward(completions, pred_xyz=pred_xyz_list, gt_xyz=gt_xyz_list)
        r_reason = reasoning_quality_reward(completions)
        r_consist = consistency_reward(completions, pred_xyz=pred_xyz_list)

        for rt, rr, rc in zip(r_traj, r_reason, r_consist):
            composite = w_traj * rt + w_reason * rr + w_consist * rc
            self._value_pending_rewards.append(composite)

        # Stash per-timestep trajectory rewards for segment-level advantages
        if self._segment_level:
            for i in range(len(completions)):
                pred_flat = pred_xyz_list[i]
                gt_flat = gt_xyz_list[i]
                r_per_step = None
                if pred_flat is not None and gt_flat is not None:
                    r_per_step_arr = trajectory_per_timestep_rewards(pred_flat, gt_flat)
                    if r_per_step_arr is not None:
                        r_per_step = r_per_step_arr.tolist()
                self._segment_reward_stash.append(
                    {
                        "r_reason": r_reason[i],
                        "r_consist": r_consist[i],
                        "r_traj": r_traj[i],
                        "r_traj_per_step": r_per_step,
                    }
                )

        self._flush_value_pending()

    def _flush_value_pending(self) -> None:
        """Aggregate pending rewards into Monte Carlo scene-level targets.

        Drains the pending groups queue in FIFO order.  For each group
        ``(h0, G)``, consumes the first ``G`` rewards from the pending
        reward buffer, computes ``V_mc = mean(rewards[:G])``, and appends
        ``(h0, V_mc)`` to the ready stash.  Stops when the next group
        cannot be fully satisfied (i.e. fewer rewards remain than the
        group size).
        """
        while self._value_pending_groups:
            h0, group_size = self._value_pending_groups[0]
            if len(self._value_pending_rewards) < group_size:
                break  # not enough rewards yet for this group

            # Consume exactly group_size rewards
            group_rewards = self._value_pending_rewards[:group_size]
            self._value_pending_rewards = self._value_pending_rewards[group_size:]
            self._value_pending_groups.popleft()

            v_mc = sum(group_rewards) / len(group_rewards)
            self._value_h0_stash.append(h0)
            self._value_targets_stash.append(v_mc)
            self._value_group_sizes_stash.append(group_size)

            logger.debug(
                "value head MC flush | group_size=%d v_mc=%.4f",
                group_size,
                v_mc,
            )

    def _train_value_head_step(self, batch_size: int) -> None:
        """Consume one batch from the stash and update the value head.

        Pops up to ``batch_size`` scene-level ``(h_0, V_mc)`` pairs from the
        ready stashes, runs an MSE update via the separate value optimizer,
        and accumulates metrics into ``self._metrics["train"]`` so they appear
        in TRL's log.  No-ops silently if the stash is empty (e.g. vLLM path
        before h_0 collection is wired in).

        Args:
            batch_size: Number of scene-level samples to consume from the stash.
        """
        n = min(batch_size, len(self._value_h0_stash), len(self._value_targets_stash))
        if n == 0:
            logger.debug(
                "value head stash empty (h0=%d, targets=%d) — skipping update",
                len(self._value_h0_stash),
                len(self._value_targets_stash),
            )
            return

        device = self.accelerator.device

        h0_batch = self._value_h0_stash[:n]
        targets_batch = self._value_targets_stash[:n]
        group_sizes_batch = self._value_group_sizes_stash[:n]
        self._value_h0_stash = self._value_h0_stash[n:]
        self._value_targets_stash = self._value_targets_stash[n:]
        self._value_group_sizes_stash = self._value_group_sizes_stash[n:]

        h0_tensor = torch.cat(h0_batch, dim=0).to(device)  # (n, hidden_dim)
        targets_tensor = torch.tensor(targets_batch, dtype=torch.float32, device=device)

        v_pred = self.value_head(h0_tensor)  # (n,)
        value_loss = F.mse_loss(v_pred, targets_tensor)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Accumulate into TRL's metric dict so values appear in the training log.
        # self._metrics["train"] is a defaultdict(list) maintained by GRPOTrainer.
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["value_head/loss"].append(value_loss.item())
        self._metrics[mode]["value_head/pred_mean"].append(v_pred.detach().mean().item())
        self._metrics[mode]["value_head/target_mean"].append(targets_tensor.mean().item())
        self._metrics[mode]["value_head/scenes_per_step"].append(float(n))
        avg_mc_group = sum(group_sizes_batch) / len(group_sizes_batch)
        self._metrics[mode]["value_head/mc_group_size"].append(avg_mc_group)
        is_pretrain = self._value_pretrain_remaining > 0
        self._metrics[mode]["value_head/pretrain_steps_remaining"].append(
            float(self._value_pretrain_remaining)
        )
        logger.debug(
            "value head%s | loss=%.4f pred=%.3f target=%.3f mc_scenes=%d",
            " [pretrain]" if is_pretrain else "",
            value_loss.item(),
            v_pred.detach().mean().item(),
            targets_tensor.mean().item(),
            n,
        )

    def _get_vlm_kv_cache_teacher_forced(
        self,
        model_inputs: dict,
        completion_prefix: list[int],
        device: torch.device,
    ) -> tuple:
        """Run a teacher-forced VLM forward on prompt + CoC reasoning to get KV cache.

        Unlike ``get_vlm_kv_cache`` in train_expert.py which uses autoregressive
        generation, this runs a single non-autoregressive forward pass on already-known
        tokens — faster, FSDP-compatible, and uses the current LoRA state.

        Args:
            model_inputs: Dict from ClipDataCache with tokenized_data, ego_history_*.
            completion_prefix: Token IDs of the generated CoC text (up to and including
                ``<|traj_future_start|>``).
            device: Compute device.

        Returns:
            (past_key_values, rope_deltas, prefill_seq_len, b_star, offset)
        """
        tokenized = {k: v for k, v in model_inputs["tokenized_data"].items()}
        input_ids = tokenized.pop("input_ids").to(device)

        # Fuse history trajectory tokens
        traj_data = {
            "ego_history_xyz": model_inputs["ego_history_xyz"].to(device),
            "ego_history_rot": model_inputs["ego_history_rot"].to(device),
        }
        input_ids = self.full_model.fuse_traj_tokens(input_ids, traj_data)

        # Concatenate prompt + completion prefix tokens
        prefix_tensor = torch.tensor(completion_prefix, dtype=torch.long, device=device).unsqueeze(
            0
        )
        full_ids = torch.cat([input_ids, prefix_tensor], dim=1)  # (1, L_prompt + L_prefix)

        # Build attention mask for the full sequence
        for k, v in tokenized.items():
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.to(device)

        if "attention_mask" in tokenized:
            orig_mask = tokenized["attention_mask"]
            prefix_mask = torch.ones(
                1, len(completion_prefix), device=device, dtype=orig_mask.dtype
            )
            tokenized["attention_mask"] = torch.cat([orig_mask, prefix_mask], dim=1)

        # Temporarily disable gradient checkpointing so `use_cache=True` is
        # honoured.  HF's forward() checks `self.gradient_checkpointing` on the
        # inner model and forces use_cache=False if it's True.
        # We must disable on BOTH the outer model (Qwen3VLForConditionalGeneration)
        # and all inner sub-modules that carry the flag (Qwen3VLModel, etc.).
        vlm = self.full_model.vlm
        gc_modules = [m for m in vlm.modules() if getattr(m, "gradient_checkpointing", False)]
        for m in gc_modules:
            m.gradient_checkpointing = False

        try:
            with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
                outputs = vlm(
                    input_ids=full_ids,
                    use_cache=True,
                    **tokenized,
                )
        finally:
            for m in gc_modules:
                m.gradient_checkpointing = True

        past_key_values = outputs.past_key_values
        if past_key_values is None:
            raise RuntimeError(
                "VLM forward returned past_key_values=None. "
                "Gradient checkpointing may not have been disabled properly."
            )
        # Navigate through PeftModel wrapper: vlm.model may be
        # Qwen3VLForConditionalGeneration (via LoraModel) instead of
        # the inner Qwen3VLModel that stores rope_deltas.
        _inner = self.full_model.vlm.model
        if not hasattr(_inner, "rope_deltas"):
            _inner = _inner.model
        rope_deltas = _inner.rope_deltas
        prefill_seq_len = past_key_values.get_seq_length()
        b_star = 1

        # Offset: position after the last token (where expert tokens would begin)
        offset = torch.tensor([full_ids.shape[1]], device=device)

        return past_key_values, rope_deltas, prefill_seq_len, b_star, offset

    def _expert_cfm_step(self, inputs: dict) -> None:
        """One expert CFM training step using top-advantage rollouts.

        Selects the highest-advantage rollouts from the current batch, runs a
        teacher-forced VLM forward to get KV cache conditioning, converts GT
        trajectory to action space, computes CFM loss, and updates expert params.
        """
        if not self._expert_rollout_data:
            logger.debug("No expert rollout data — skipping expert CFM step")
            return

        cfg = self._expert_finetune_cfg
        rollouts_per_step = int(cfg.get("rollouts_per_step", 1))
        num_noisy_samples = int(cfg.get("num_noisy_samples", 4))
        max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

        device = self.accelerator.device

        # Get advantages from the current batch and pick top-K
        advantages = inputs.get("advantages", None)
        if advantages is None:
            logger.debug("No advantages in inputs — skipping expert CFM step")
            return

        # advantages is a tensor of shape (batch_size,)
        if isinstance(advantages, torch.Tensor):
            adv_values = advantages.detach().cpu().tolist()
        else:
            adv_values = list(advantages)

        n_rollouts = min(len(adv_values), len(self._expert_rollout_data))
        if n_rollouts == 0:
            return

        # Rank by advantage (descending) and pick top-K
        sorted_indices = sorted(range(n_rollouts), key=lambda i: adv_values[i], reverse=True)
        top_k = sorted_indices[:rollouts_per_step]

        from alpamayo_r1.training.train_expert import compute_cfm_loss

        # Move expert + projections to GPU
        self.full_model.expert.to(device)
        self.full_model.action_in_proj.to(device)
        self.full_model.action_out_proj.to(device)
        self.full_model.action_space.to(device)

        total_loss = 0.0
        n_valid = 0
        self._expert_optimizer.zero_grad()

        for idx in top_k:
            rollout_info = self._expert_rollout_data[idx]
            clip_id = rollout_info["clip_id"]
            t0_us = rollout_info["t0_us"]
            completion_prefix = rollout_info["completion_prefix"]

            try:
                # Load scene data from cache
                model_inputs, ego_future_xyz = self._data_cache.get(clip_id, t0_us, device)
                ego_future_rot = self._data_cache.get_ego_future_rot(clip_id, t0_us).to(device)

                # Teacher-forced VLM forward → KV cache
                kv_result = self._get_vlm_kv_cache_teacher_forced(
                    model_inputs, completion_prefix, device
                )
                prompt_cache, rope_deltas, prefill_seq_len, b_star, offset = kv_result

                # Convert GT trajectory to action space.
                # ego_history_*: (1, N_cameras, T, ...) — take last camera
                # ego_future_*: (1, 1, T, ...) from load_physical_aiavdataset
                hist_xyz = model_inputs["ego_history_xyz"][:, -1].to(device)  # (1, T, 3)
                hist_rot = model_inputs["ego_history_rot"][:, -1].to(device)  # (1, T, 3, 3)
                fut_xyz = ego_future_xyz.to(device)[:, -1]  # (1, T, 3)
                fut_rot = ego_future_rot[:, -1]  # (1, T, 3, 3), already on device

                with torch.no_grad():
                    gt_action = self.full_model.action_space.traj_to_action(
                        hist_xyz, hist_rot, fut_xyz, fut_rot
                    )  # (1, T_future, 2)

                # Compute CFM loss
                with torch.autocast(str(device), dtype=torch.bfloat16):
                    loss = compute_cfm_loss(
                        model=self.full_model,
                        gt_action=gt_action,
                        prompt_cache=prompt_cache,
                        rope_deltas=rope_deltas,
                        prefill_seq_len=prefill_seq_len,
                        b_star=b_star,
                        offset=offset,
                        num_noisy_samples=num_noisy_samples,
                        device=device,
                    )
                    loss = loss / rollouts_per_step  # average over top-K

                loss.backward()
                total_loss += loss.item()
                n_valid += 1

            except Exception as e:
                logger.warning("Expert CFM step failed for %s: %s", clip_id, e)
                continue

        if n_valid > 0:
            # Clip gradients and step
            expert_params = [
                p
                for n, p in self.full_model.named_parameters()
                if any(
                    n.startswith(pfx) for pfx in ("expert.", "action_in_proj.", "action_out_proj.")
                )
                and p.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(expert_params, max_grad_norm)
            self._expert_optimizer.step()

            # Log metrics
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["expert_cfm/loss"].append(total_loss)
            self._metrics[mode]["expert_cfm/valid_rollouts"].append(float(n_valid))
            logger.debug(
                "expert CFM step | loss=%.6f valid_rollouts=%d/%d",
                total_loss,
                n_valid,
                rollouts_per_step,
            )

        # Move expert + projections back to CPU to save GPU memory
        self.full_model.expert.cpu()
        self.full_model.action_in_proj.cpu()
        self.full_model.action_out_proj.cpu()
        self.full_model.action_space.cpu()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to train value head and/or expert CFM alongside the GRPO policy loss.

        **Value head stage 0** (``_value_pretrain_remaining > 0``): only the value head
        trains.  The policy loss is replaced by a zero scalar so the VLM
        receives no gradient.

        **Value head stage 1** (``_value_pretrain_remaining == 0``): normal GRPO plus a
        value head update from the stash.

        **Expert CFM**: When enabled, runs ``_expert_cfm_step()`` every
        ``every_n_steps`` training steps. This is independent of the GRPO loss —
        the expert has its own optimizer and gradients are isolated.
        """
        # Expert CFM step (runs before GRPO to not interfere with policy gradients)
        if self._expert_finetune_enabled:
            every_n = int(self._expert_finetune_cfg.get("every_n_steps", 1))
            if self.state.global_step % every_n == 0:
                try:
                    self._expert_cfm_step(inputs)
                except Exception as e:
                    logger.warning(
                        "Expert CFM step failed at step %d: %s", self.state.global_step, e
                    )

        if self.value_head is None and not self._expert_finetune_enabled:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        if self.value_head is not None:
            # Segment-level training happens in _compute_segment_advantages,
            # so only do scene-level _train_value_head_step when NOT in segment mode.
            if not self._segment_level:
                batch_size = inputs["input_ids"].shape[0] if "input_ids" in inputs else 1
                self._train_value_head_step(batch_size)

            # Stage 0: skip GRPO policy update
            if self._value_pretrain_remaining > 0:
                self._value_pretrain_remaining -= 1
                zero_loss = 0.0 * sum(p.sum() for p in model.parameters() if p.requires_grad)
                if return_outputs:
                    return zero_loss, None
                return zero_loss

        # Normal GRPO policy loss
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


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

    Processes completions in groups of ``mini_batch_size`` to reduce the number
    of VLM forward passes (e.g. 2 passes for 8 completions with batch size 4,
    vs 8 serial passes in the original implementation).

    When ``output_hidden_states=True``, also returns the last-layer hidden
    states for each completion (extracted and moved to CPU immediately).

    Args:
        full_model: The full AlpamayoR1 model.
        model_inputs: Dict with tokenized_data (input_ids may be popped).
        prompt_input_ids: Saved prompt input_ids, shape (1, L_prompt).
        completion_ids_list: List of completion token ID lists.
        prompt_len: Number of prompt tokens.
        device: CUDA device.
        mini_batch_size: Number of completions to process per forward pass.
        output_hidden_states: If True, also return per-completion hidden states.

    Returns:
        If output_hidden_states is False:
            List of per-token log-prob lists, one per completion.
        If output_hidden_states is True:
            Tuple of (logprobs_list, hidden_states_list) where hidden_states_list
            contains one (1, L_completion, D) CPU float32 tensor per completion
            (with positions relative to the full [prompt + completion] sequence,
            sliced to only the completion portion plus the last prompt token).
    """
    results: list[list[float]] = []
    hidden_results: list[torch.Tensor] = []
    tokenized = model_inputs["tokenized_data"]

    for batch_start in range(0, len(completion_ids_list), mini_batch_size):
        batch_comp_ids = completion_ids_list[batch_start : batch_start + mini_batch_size]
        B = len(batch_comp_ids)

        # Build per-completion tensors (use a placeholder token for empties)
        comp_tensors = [
            torch.tensor(ids, dtype=torch.long, device=device)
            if ids
            else torch.tensor([0], dtype=torch.long, device=device)
            for ids in batch_comp_ids
        ]
        comp_lens = [len(ids) for ids in batch_comp_ids]
        max_comp_len = max(t.shape[0] for t in comp_tensors)

        # Pad completions to max_comp_len within the mini-batch
        comp_padded = torch.zeros(B, max_comp_len, dtype=torch.long, device=device)
        for i, t in enumerate(comp_tensors):
            comp_padded[i, : t.shape[0]] = t

        # Build full input_ids: (B, L_prompt + max_comp_len)
        prompt_expanded = prompt_input_ids.expand(B, -1)  # (B, L_prompt)
        full_ids = torch.cat([prompt_expanded, comp_padded], dim=1)

        # Build forward kwargs
        forward_kwargs = {}
        if "attention_mask" in tokenized:
            orig_mask = tokenized["attention_mask"]  # (1, L_prompt)
            comp_mask = torch.zeros(B, max_comp_len, device=device, dtype=orig_mask.dtype)
            for i, comp_len in enumerate(comp_lens):
                if comp_len > 0:
                    comp_mask[i, :comp_len] = 1
            forward_kwargs["attention_mask"] = torch.cat(
                [orig_mask.expand(B, -1), comp_mask], dim=1
            )
        if "pixel_values" in tokenized:
            pv = tokenized["pixel_values"]
            # Repeat along first dim: (N_patches, ...) -> (B*N_patches, ...)
            forward_kwargs["pixel_values"] = pv.repeat(B, *([1] * (pv.dim() - 1)))
        if "image_grid_thw" in tokenized:
            igt = tokenized["image_grid_thw"]  # (N_images, 3)
            forward_kwargs["image_grid_thw"] = igt.repeat(B, 1)

        with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
            outputs = full_model.vlm(
                input_ids=full_ids,
                output_hidden_states=output_hidden_states,
                **forward_kwargs,
            )

        # Extract per-token log-probs for each completion in the mini-batch.
        # Logits at position t predict token t+1, so completion tokens at
        # positions [prompt_len .. prompt_len+comp_len) are predicted by
        # logits at [prompt_len-1 .. prompt_len-1+comp_len).
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
                # Extract last-layer hidden states: include last prompt token + completion
                # positions [prompt_len-1 .. prompt_len-1+comp_len]
                hs = outputs.hidden_states[-1][i, prompt_len - 1 : prompt_len + comp_len]
                hidden_results.append(hs.float().cpu().unsqueeze(0))  # (1, 1+comp_len, D)

    if output_hidden_states:
        return results, hidden_results
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
    logprobs_padded = torch.zeros((len(all_logprobs), max_comp_len), dtype=torch.float32)
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


# ---------------------------------------------------------------------------
# GPU utilization callback
# ---------------------------------------------------------------------------


class GpuUtilizationCallback(TrainerCallback):
    """Prints GPU SM utilization and memory usage to the log on each logging step.

    Uses pynvml when available (nvidia-ml-py3 package); silently skips if not
    installed or if no NVIDIA devices are found.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            import pynvml

            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            parts = []
            for i in range(n):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                parts.append(
                    f"GPU{i} {util.gpu:3d}% SM  {mem.used / 2**30:.1f}/{mem.total / 2**30:.1f} GB"
                )
            logger.info("GPU util | step %d | %s", state.global_step, " | ".join(parts))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Rollout logging callback
# ---------------------------------------------------------------------------


class RolloutLoggingCallback(TrainerCallback):
    """Logs CoC text and BEV trajectory plots to TensorBoard periodically.

    Reads stashed rollout data from ``AlpamayoGRPOTrainer._rollout_log_data``
    and writes it as TensorBoard text + figures.

    Uses ``on_log`` so the TensorBoardCallback's writer is guaranteed to be
    initialised (it calls ``_init_summary_writer`` inside its own ``on_log``).

    Performance: ~35 ms per logging step (matplotlib render + TB write),
    negligible compared to the 10-60 s training step. Only fires every
    ``log_interval`` steps.

    Args:
        log_interval: Steps between text rollout logs (CoC text, generated tokens).
        plot_interval: Steps between BEV trajectory plots. Defaults to log_interval.
        max_samples: Max unique prompts to log per interval.
    """

    def __init__(
        self, log_interval: int = 1, plot_interval: int | None = None, max_samples: int = 2
    ):
        self.log_interval = log_interval
        self.plot_interval = plot_interval if plot_interval is not None else log_interval
        self.max_samples = max_samples
        self.trainer = None  # set after trainer construction
        self._tb_writer = None

    def _get_tb_writer(self):
        """Retrieve the SummaryWriter from the TensorBoardCallback.

        Called from ``on_log``, so the TensorBoardCallback has already run
        its own ``on_log`` (which calls ``_init_summary_writer`` if needed),
        guaranteeing ``tb_writer`` is set.
        """
        if self._tb_writer is not None:
            return self._tb_writer
        if self.trainer is None:
            return None
        for cb in self.trainer.callback_handler.callbacks:
            if hasattr(cb, "tb_writer") and cb.tb_writer is not None:
                self._tb_writer = cb.tb_writer
                return self._tb_writer
        # Fallback: create our own writer using the trainer's logging dir
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = getattr(self.trainer.args, "logging_dir", None)
            if log_dir:
                logger.info("RolloutLoggingCallback: creating own SummaryWriter at %s", log_dir)
                self._tb_writer = SummaryWriter(log_dir=log_dir)
                return self._tb_writer
        except ImportError:
            pass
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        should_log_text = state.global_step % self.log_interval == 0
        should_plot = state.global_step % self.plot_interval == 0
        if not should_log_text and not should_plot:
            return
        if self.trainer is None:
            return
        data = getattr(self.trainer, "_rollout_log_data", None)
        if not data:
            return

        writer = self._get_tb_writer()
        if writer is None:
            return

        step = state.global_step
        # Infer local generation count from clip_ids to handle DDP/FSDP where
        # each rank holds fewer than num_generations samples per prompt.
        clip_ids = data["clip_ids"]
        if clip_ids:
            num_gen = 1
            for i in range(1, len(clip_ids)):
                if clip_ids[i] == clip_ids[0]:
                    num_gen += 1
                else:
                    break
        else:
            num_gen = max(data.get("num_generations", 1), 1)
        num_prompts = len(data["coc_texts"]) // max(num_gen, 1)
        n_log = min(num_prompts, self.max_samples)

        for i in range(n_log):
            base = i * num_gen
            clip_id = data["clip_ids"][base]

            if should_log_text:
                # --- CoC text (as markdown) ---
                text_parts = []
                for j in range(min(num_gen, 4)):
                    text_parts.append(f"**Sample {j}:**\n\n{data['coc_texts'][base + j]}")
                text_md = f"**Clip:** `{clip_id}`\n\n" + "\n\n---\n\n".join(text_parts)
                writer.add_text(f"rollout/coc_text_{i}", text_md, step)

                # --- Generated tokens (decoded) ---
                completion_ids = data.get("completion_ids")
                if completion_ids and self.trainer is not None:
                    tokenizer = self.trainer.processing_class.tokenizer
                    token_parts = []
                    for j in range(min(num_gen, 4)):
                        ids = completion_ids[base + j]
                        decoded = tokenizer.decode(ids, skip_special_tokens=False)
                        token_parts.append(
                            f"**Sample {j}** ({len(ids)} tokens):\n\n"
                            f"`{ids[:20]}{'...' if len(ids) > 20 else ''}`\n\n"
                            f"```\n{decoded}\n```"
                        )
                    tokens_md = f"**Clip:** `{clip_id}`\n\n" + "\n\n---\n\n".join(token_parts)
                    writer.add_text(f"rollout/generated_tokens_{i}", tokens_md, step)
                    logger.info(
                        "Step %d | Clip %s | Sample 0 (%d tokens): %s",
                        step,
                        clip_id,
                        len(completion_ids[base]),
                        tokenizer.decode(completion_ids[base], skip_special_tokens=False)[:200],
                    )

            if should_plot:
                # --- BEV trajectory plot ---
                try:
                    fig = _plot_trajectories_bev(
                        pred_list=[data["pred_xyz"][base + j] for j in range(num_gen)],
                        gt_flat=data["gt_xyz"][base],
                        title=f"Step {step} | {clip_id}",
                    )
                    writer.add_figure(f"rollout/trajectory_bev_{i}", fig, step)
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except OSError as e:
                    logger.warning("Skipping BEV plot at step %d (matplotlib error: %s)", step, e)

        writer.flush()


def _plot_trajectories_bev(
    pred_list: list[list[float]],
    gt_flat: list[float],
    title: str = "",
):
    """Render a bird's-eye-view XY trajectory plot.

    Args:
        pred_list: List of flattened predicted trajectories (one per sample).
        gt_flat: Flattened ground-truth trajectory.
        title: Plot title.

    Returns:
        matplotlib Figure (caller must close it).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))

    gt = np.array(gt_flat, dtype=np.float32).reshape(-1, 3)
    ax.plot(gt[:, 0], gt[:, 1], "k-", linewidth=2.5, label="GT", zorder=10)

    cmap = plt.cm.tab10
    for j, pred_flat in enumerate(pred_list):
        pred = np.array(pred_flat, dtype=np.float32).reshape(-1, 3)
        ax.plot(
            pred[:, 0],
            pred[:, 1],
            "--",
            color=cmap(j % 10),
            alpha=0.6,
            linewidth=1.0,
            label=f"Pred {j}" if j < 6 else None,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="best", fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
