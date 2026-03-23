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

import copy
import logging
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
from transformers import AutoConfig, AutoModel

from alpamayo_r1.action_space import ActionSpace
from alpamayo_r1.models.base_model import ReasoningVLA
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.diffusion.base import BaseDiffusion
from alpamayo_r1.inference import decode_vlm_trajectories, generate_coc, prepare_vlm_inputs
from alpamayo_r1.models.token_utils import (
    ExpertLogitsProcessor,  # noqa: F401 — re-exported for backwards compat
    extract_text_tokens,
    replace_padding_after_eos,
)

logger = logging.getLogger(__name__)
# Ensure this logger emits to stdout even in subprocess/container environments
# where the root logger may not be configured.
if not logger.handlers:
    import sys

    _h = logging.StreamHandler(sys.stdout)
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


class AlpamayoR1(ReasoningVLA):
    """Expert model for reasoning VLA."""

    config_class: type[AlpamayoR1Config] = AlpamayoR1Config
    base_model_prefix = "vlm"

    @classmethod
    def from_pretrained_with_lora(
        cls,
        adapter_path: str,
        base_model_name: str = "nvidia/Alpamayo-R1-10B",
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | None = "auto",
        merge: bool = True,
    ) -> "AlpamayoR1":
        """Load base AlpamayoR1 model and apply a LoRA adapter to the VLM.

        GRPO training saves only a PEFT LoRA adapter for the VLM component.
        This method loads the full AlpamayoR1 model, applies the adapter to
        ``self.vlm``, optionally merges the weights, and returns the model
        ready for inference.

        Args:
            adapter_path: Path to the LoRA adapter checkpoint directory
                (contains ``adapter_config.json`` and ``adapter_model.safetensors``).
            base_model_name: HuggingFace model ID or local path for the base
                AlpamayoR1 model.
            dtype: Torch dtype for model weights.
            device_map: Device placement strategy (passed to ``from_pretrained``).
            merge: If True, merge LoRA weights into the base model and unload
                the adapter for faster inference.

        Returns:
            AlpamayoR1 model with LoRA weights applied.
        """
        from peft import PeftModel

        logger.info("Loading base model from %s", base_model_name)
        model = cls.from_pretrained(base_model_name, dtype=dtype, device_map=device_map)

        logger.info("Applying LoRA adapter from %s", adapter_path)
        model.vlm = PeftModel.from_pretrained(model.vlm, adapter_path)

        if merge:
            logger.info("Merging LoRA weights into base model (in float32 to avoid bf16 underflow)")
            model.vlm = model.vlm.to(torch.float32)
            model.vlm = model.vlm.merge_and_unload()
            model.vlm = model.vlm.to(dtype)

        return model

    def __init__(
        self,
        config: AlpamayoR1Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
    ):
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count=False)

        # we only need the text config for the expert model
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self.expert = AutoModel.from_config(expert_config)
        # we don't need the embed_tokens of the expert model
        del self.expert.embed_tokens

        self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
        self.diffusion: BaseDiffusion = hyu.instantiate(
            config.diffusion_cfg,
            x_dims=self.action_space.get_action_space_dims(),
        )

        self.action_in_proj = hyu.instantiate(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=expert_config.hidden_size,
        )
        self.action_out_proj = hyu.instantiate(
            config.action_out_proj_cfg,
            in_features=expert_config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )

        # Convert action-related modules to the same dtype as expert
        expert_dtype = self.expert.dtype
        if self.config.keep_same_dtype:
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        self.post_init()

    def _postprocess_trajectories(
        self,
        pred_xyz: torch.Tensor,
        pred_rot: torch.Tensor,
        vlm_sequences: torch.Tensor,
        num_traj_sets: int,
        num_traj_samples: int,
        input_ids_B: int,
        return_extra: bool,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        """Reshape predictions to (B, ns, nj, ...) and optionally extract CoC text.

        This is shared post-processing for both the full-pipeline and
        VLM-only trajectory generation methods.
        """
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        if return_extra:
            extra = extract_text_tokens(self.tokenizer, vlm_sequences)
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids_B, num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot

    def _repeat_history(
        self,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        n_samples_total: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeat history tensors to align with num_traj_samples * num_traj_sets."""
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        return hist_xyz_rep, hist_rot_rep

    def sample_trajectories_from_data_with_vlm_rollout(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample trajectories using VLM + Expert + Diffusion (full pipeline).

        The VLM generates CoC reasoning text (trajectory tokens are masked by
        ``ExpertLogitsProcessor``), then the Expert Transformer + Diffusion
        model produces continuous trajectories in action space.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        n_samples_total = num_traj_samples * num_traj_sets
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )

        input_ids, gen_kwargs = prepare_vlm_inputs(self, data)
        device = input_ids.device

        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."

        vlm_outputs = generate_coc(
            self,
            input_ids,
            gen_kwargs,
            mode="expert",
            temperature=temperature,
            top_p=top_p,
            num_samples=num_traj_samples,
            max_new_tokens=max_generation_length,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict=True,
        )
        eos_token_id = self.special_token_ids["traj_future_start"]
        # Navigate through PeftModel wrapper if LoRA is unmerged
        _inner = self.vlm.model
        if not hasattr(_inner, "rope_deltas"):
            _inner = _inner.model
        vlm_outputs.rope_deltas = _inner.rope_deltas

        # Log raw VLM generation for debugging
        prompt_len = input_ids.shape[1]
        for si in range(vlm_outputs.sequences.shape[0]):
            gen_ids = vlm_outputs.sequences[si, prompt_len:].cpu().tolist()
            while gen_ids and gen_ids[-1] == self.tokenizer.pad_token_id:
                gen_ids.pop()
            tail_ids = input_ids[si].cpu().tolist()[-35:]
            input_tail = self.tokenizer.decode(tail_ids, skip_special_tokens=False)
            logger.info(
                "[vlm_rollout] sample %d input tail ids: %s | decoded: ...%s | raw generation (%d tokens): %s",
                si,
                tail_ids,
                input_tail,
                len(gen_ids),
                self.tokenizer.decode(gen_ids, skip_special_tokens=False),
            )

        # manually replace padding after EOS token
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        # find <traj_future_start> token position for each sequence, use last token if not found
        b_star = vlm_outputs.sequences.shape[0]
        traj_future_start_mask = vlm_outputs.sequences == eos_token_id
        # [b_star], True if sequence has <traj_future_start>
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        for i in range(b_star):
            if not has_traj_future_start[i]:
                logger.warning(
                    f"No <traj_future_start> token found in the generated sequences for sequence {i}"
                )
        # [b_star], first occurrence position
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
        last_token_positions = torch.full(
            (b_star,), vlm_outputs.sequences.shape[1] - 1, device=device
        )
        valid_token_pos_id = torch.where(
            has_traj_future_start, traj_future_start_positions, last_token_positions
        )
        # note that vlm_outputs.sequences already include the input_ids,
        # so no need to add the input_ids length
        offset = valid_token_pos_id + 1

        # modify the position ids to remove padding tokens
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        delta = vlm_outputs.rope_deltas + offset[:, None]
        position_ids += delta.to(position_ids.device)

        # modify the attention_masks to remove padding tokens
        attention_mask = torch.zeros(
            (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
            dtype=torch.float32,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                attention_mask.dtype
            ).min

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # 2) Define denoising step that consumes noisy action and timestep
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            # x: (B*, *action_dim)
            # t: broadcastable to x leading dims
            b_star = x.shape[0]
            # Project noisy action to expert token embeddings for the n future tokens
            # Expect shape (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            # Run expert with cached prefill, only on the future tokens
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            # crop the prompt cache to remove the newly added tokens
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # (b*, Tf, hidden_size)
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )  # (b*, Tf, C_action) -> noise/vector field
            return pred

        # 3) Diffusion sampling in action space with multiple samples per input
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        hist_xyz_rep, hist_rot_rep = self._repeat_history(
            ego_history_xyz, ego_history_rot, n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        return self._postprocess_trajectories(
            pred_xyz,
            pred_rot,
            vlm_outputs.sequences,
            num_traj_sets,
            num_traj_samples,
            input_ids.shape[0],
            kwargs.get("return_extra", False),
        )

    def sample_trajectories_from_data_with_vlm_only(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        max_generation_length: int = 256,
        return_extra: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        """Generate trajectories using VLM-only rollout (no Expert/Diffusion).

        The VLM autoregressively generates both Chain-of-Causation reasoning
        text AND discrete trajectory tokens.  Trajectory tokens are decoded
        back to continuous (x, y, z) waypoints via ``traj_tokenizer.decode()``.

        This mirrors the GRPO training rollout and is useful for evaluating
        VLM trajectory prediction quality without the Expert/Diffusion modules
        (which can be kept off-GPU to save memory).

        Args:
            data: Model inputs dict with keys ``tokenized_data``,
                ``ego_history_xyz``, and ``ego_history_rot``.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling (None to disable).
            temperature: Sampling temperature.
            num_traj_samples: Number of trajectory samples per input.
            num_traj_sets: Number of trajectory sets.
            max_generation_length: Max new tokens for CoC text generation
                (trajectory token budget is added automatically).
            return_extra: If True, return a dict with CoC text as third element.

        Returns:
            pred_xyz: Predicted future waypoints, shape
                ``(B, num_traj_sets, num_traj_samples, T, 3)``.
            pred_rot: Predicted future rotations, shape
                ``(B, num_traj_sets, num_traj_samples, T, 3, 3)``.
            extra: (only if ``return_extra=True``) Dict with ``"cot"`` key
                containing CoC text, shape ``(B, num_traj_sets, num_traj_samples)``.
        """
        n_samples_total = num_traj_samples * num_traj_sets
        max_new_tokens = max_generation_length + self.config.tokens_per_future_traj + 10

        input_ids, gen_kwargs = prepare_vlm_inputs(self, data)

        vlm_output = generate_coc(
            self,
            input_ids,
            gen_kwargs,
            mode="vlm",
            temperature=temperature,
            top_p=top_p,
            num_samples=num_traj_samples,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # vlm_output: (B * num_traj_samples, prompt_len + generated_len)

        # Log raw VLM generation for debugging
        prompt_len = input_ids.shape[1]
        for si in range(vlm_output.shape[0]):
            gen_ids = vlm_output[si, prompt_len:].cpu().tolist()
            while gen_ids and gen_ids[-1] == self.tokenizer.pad_token_id:
                gen_ids.pop()
            logger.info(
                "[vlm_only] sample %d raw generation (%d tokens): %s",
                si,
                len(gen_ids),
                self.tokenizer.decode(gen_ids, skip_special_tokens=False),
            )

        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        hist_xyz_rep, hist_rot_rep = self._repeat_history(
            ego_history_xyz, ego_history_rot, n_samples_total
        )

        pred_xyz, pred_rot = decode_vlm_trajectories(self, vlm_output, hist_xyz_rep, hist_rot_rep)

        return self._postprocess_trajectories(
            pred_xyz,
            pred_rot,
            vlm_output,
            num_traj_sets,
            num_traj_samples,
            input_ids.shape[0],
            return_extra,
        )


AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
