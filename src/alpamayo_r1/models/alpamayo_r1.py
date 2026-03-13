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
from transformers import AutoConfig, AutoModel, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from alpamayo_r1.action_space import ActionSpace
from alpamayo_r1.models.base_model import ReasoningVLA
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.diffusion.base import BaseDiffusion
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    extract_traj_tokens,
    replace_padding_after_eos,
    to_special_token,
)

logger = logging.getLogger(__name__)


class ExpertLogitsProcessor(LogitsProcessor):
    """Masks out the logits for discrete trajectory tokens."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        """Initialize the ExpertLogitsProcessor.

        Args:
            traj_token_offset: The offset of the trajectory tokens.
            traj_vocab_size: The vocabulary size of the trajectory tokens.
        """
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call the ExpertLogitsProcessor to mask out the logits for discrete trajectory tokens.

        The discrete trajectory tokens are not used for the expert model thus masking them out for
        better CoC generation.

        Args:
            input_ids: The input IDs.
            scores: The scores.

        Returns:
            torch.FloatTensor: The modified scores tensor with trajectory tokens masked out (set to -inf).
        """
        # Directly assign -inf to the trajectory token positions in the scores tensor
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float('-inf')
        return scores


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
            logger.info("Merging LoRA weights into base model")
            model.vlm = model.vlm.merge_and_unload()

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

    def _prepare_vlm_generation(
        self,
        data: dict[str, Any],
        top_p: float,
        top_k: int | None,
        temperature: float,
        num_traj_samples: int,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor, int]:
        """Prepare inputs and generation config shared by both trajectory methods.

        Unpacks ``data``, fuses history trajectory tokens into ``input_ids``,
        and configures the VLM generation settings.

        Returns:
            input_ids, tokenized_data (remaining), ego_history_xyz,
            ego_history_rot, B (batch size).
        """
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."

        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)

        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_new_tokens
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        return input_ids, tokenized_data, ego_history_xyz, ego_history_rot, B

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

        input_ids, tokenized_data, ego_history_xyz, ego_history_rot, B = (
            self._prepare_vlm_generation(
                data, top_p, top_k, temperature, num_traj_samples, max_generation_length,
            )
        )
        device = input_ids.device

        # Full pipeline needs logits and KV cache for Expert
        generation_config = self.vlm.generation_config
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True

        # use custom stopping criteria to stop after EOS token + one more token,
        # because the KV cache is updated after the next token is generated
        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

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
            pred_xyz, pred_rot, vlm_outputs.sequences,
            num_traj_sets, num_traj_samples,
            input_ids.shape[0], kwargs.get("return_extra", False),
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
        tokens_per_future_traj = self.config.tokens_per_future_traj
        max_new_tokens = max_generation_length + tokens_per_future_traj + 10

        input_ids, tokenized_data, ego_history_xyz, ego_history_rot, B = (
            self._prepare_vlm_generation(
                data, top_p, top_k, temperature, num_traj_samples, max_new_tokens,
            )
        )

        # No ExpertLogitsProcessor — VLM freely generates trajectory tokens.
        # Stop at <|traj_future_end|> (the VLM generates tokens *through* the
        # trajectory section, unlike the full pipeline which stops at _start).
        traj_future_end_id = self.special_token_ids["traj_future_end"]
        stopping_criteria = StoppingCriteriaList(
            [StopAfterEOS(eos_token_id=traj_future_end_id)]
        )

        vlm_output = self.vlm.generate(
            input_ids=input_ids,
            generation_config=self.vlm.generation_config,
            stopping_criteria=stopping_criteria,
            **tokenized_data,
        )
        # vlm_output: (B * num_traj_samples, prompt_len + generated_len)

        # Extract discrete trajectory tokens and decode to continuous xyz
        traj_tokens = extract_traj_tokens(
            vlm_output,
            self.special_token_ids,
            tokens_per_future_traj,
            self.future_token_start_idx,
            self.config.traj_vocab_size,
        )

        hist_xyz_rep, hist_rot_rep = self._repeat_history(
            ego_history_xyz, ego_history_rot, n_samples_total
        )

        pred_xyz, pred_rot, _ = self.traj_tokenizer.decode(
            hist_xyz_rep, hist_rot_rep, traj_tokens
        )

        return self._postprocess_trajectories(
            pred_xyz, pred_rot, vlm_output,
            num_traj_sets, num_traj_samples,
            input_ids.shape[0], return_extra,
        )


AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
