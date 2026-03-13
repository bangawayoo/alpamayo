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
import re
from collections import OrderedDict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from physical_ai_av import PhysicalAIAVDatasetInterface
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoProcessor, GenerationConfig, StoppingCriteriaList, TrainerCallback
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.token_utils import StopAfterEOS, extract_traj_tokens

if TYPE_CHECKING:
    from trl import GRPOConfig

logger = logging.getLogger(__name__)

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


class ClipDataCache:
    """Cache processed clip data (images + ego motion) in CPU RAM.

    On first access for a (clip_id, t0_us) pair, loads raw driving data via
    load_physical_aiavdataset and runs prepare_model_inputs on CPU. Results
    are stored as CPU tensors. Subsequent accesses call helper.to_device which
    always creates new tensors, so callers may modify returned dicts (e.g. pop
    keys) without corrupting the cache.

    When ``cache_pil_images=True``, raw PIL images are also cached for vLLM
    multimodal generation (vLLM expects PIL images, not pre-processed tensors).

    ``max_size`` caps the number of cached clips. When the limit is reached,
    the least-recently-used entry is evicted (LRU policy). Each clip entry
    with PIL images occupies ~130 MB of CPU RAM, so the default of 200 clips
    uses ~26 GB. Set lower if the training node has limited RAM.
    """

    def __init__(
        self,
        avdi: PhysicalAIAVDatasetInterface,
        processor: AutoProcessor,
        cache_pil_images: bool = False,
        max_size: int = 200,
    ) -> None:
        self._avdi = avdi
        self._processor = processor
        self._cache: OrderedDict[tuple[str, int], dict] = OrderedDict()
        self._cache_pil_images = cache_pil_images
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _load_and_cache(self, clip_id: str, t0_us: int) -> None:
        """Load raw data and populate the cache entry, evicting LRU if full."""
        self._misses += 1
        data = load_physical_aiavdataset(
            clip_id=clip_id, t0_us=t0_us, avdi=self._avdi, maybe_stream=True
        )
        model_inputs_cpu = helper.prepare_model_inputs(data, self._processor, device="cpu")
        entry: dict[str, Any] = {
            "model_inputs": model_inputs_cpu,
            "ego_future_xyz": data["ego_future_xyz"],
        }
        if self._cache_pil_images:
            # Convert (N_cameras, num_frames, 3, H, W) → flat list of PIL images
            frames = data["image_frames"].flatten(0, 1)  # (N*F, 3, H, W)
            pil_images = [
                Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8)) for frame in frames
            ]
            entry["pil_images"] = pil_images
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # evict LRU (oldest) entry
            self._evictions += 1
        self._cache[(clip_id, t0_us)] = entry
        logger.debug(
            "Cache miss for (%s, %d). Size: %d/%d. Hits: %d, misses: %d, evictions: %d.",
            clip_id,
            t0_us,
            len(self._cache),
            self._max_size,
            self._hits,
            self._misses,
            self._evictions,
        )

    def get(self, clip_id: str, t0_us: int, device: torch.device) -> tuple[dict, torch.Tensor]:
        """Return (model_inputs_on_device, ego_future_xyz_cpu).

        model_inputs are freshly moved to device each call (safe to modify).
        ego_future_xyz stays on CPU for numpy compatibility.
        """
        key = (clip_id, t0_us)
        if key not in self._cache:
            self._load_and_cache(clip_id, t0_us)
        else:
            self._hits += 1
            self._cache.move_to_end(key)  # mark as recently used
        cached = self._cache[key]
        return helper.to_device(cached["model_inputs"], device=device), cached["ego_future_xyz"]

    def get_pil_images(self, clip_id: str, t0_us: int) -> list[Image.Image]:
        """Return cached PIL images for vLLM multimodal generation.

        Requires ``cache_pil_images=True`` at construction time.
        Must call ``get()`` first to ensure the cache is populated.
        """
        key = (clip_id, t0_us)
        if key not in self._cache:
            self._load_and_cache(clip_id, t0_us)
        else:
            self._cache.move_to_end(key)  # mark as recently used
        return self._cache[key]["pil_images"]


def _group_consecutive_prompts(prompts: list) -> list[tuple[Any, int]]:
    """Group consecutive identical prompts and return (prompt, count) pairs.

    TRL repeats each prompt ``num_generations`` times, but distributed training
    may split those repetitions across GPUs.  This helper detects the *actual*
    local repeat count instead of assuming ``num_generations`` copies are present.
    """
    groups: list[tuple[Any, int]] = []
    for p in prompts:
        if groups and _prompt_eq(groups[-1][0], p):
            groups[-1] = (groups[-1][0], groups[-1][1] + 1)
        else:
            groups.append((p, 1))
    return groups


def _prompt_eq(a: Any, b: Any) -> bool:
    """Check if two prompts are equal (handles str and list-of-dict formats)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, str):
        return a == b
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(x == y for x, y in zip(a, b))
    return a == b


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


def _collapse_image_pad_tokens(
    token_ids: list[int],
    image_pad_id: int = 151655,
    vision_start_id: int = 151652,
    vision_end_id: int = 151653,
) -> list[int]:
    """Collapse runs of ``<|image_pad|>`` to a single token per image.

    vLLM's multimodal preprocessor needs at least one ``<|image_pad|>`` between
    ``<|vision_start|>`` and ``<|vision_end|>`` as a marker to know where to
    expand.  But if ALL pre-expanded pads are kept, vLLM adds a second set
    (double-insertion).  Keeping exactly one pad per image lets vLLM replace
    it with the correct count.
    """
    result = []
    in_vision = False
    pad_emitted = False
    for t in token_ids:
        if t == vision_start_id:
            in_vision = True
            pad_emitted = False
            result.append(t)
        elif t == vision_end_id:
            in_vision = False
            result.append(t)
        elif t == image_pad_id and in_vision:
            if not pad_emitted:
                result.append(t)
                pad_emitted = True
        else:
            result.append(t)
    return result


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
        num_generations = (
            trainer.num_generations if trainer.model.training else trainer.num_generations_eval
        )

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


def prepare_vlm_for_training(full_model: AlpamayoR1) -> None:
    """Patch the VLM and its config for compatibility with TRL, PEFT, and FSDP.

    Must be called before passing ``full_model.vlm`` to ``AlpamayoGRPOTrainer``.

    Patches applied:
    - Sets ``name_or_path`` / ``_name_or_path`` so TRL/vLLM can locate the
      checkpoint.
    - Monkey-patches ``Qwen3VLConfig.__init__`` to expose ``vocab_size`` at
      the top level (PEFT's ``get_peft_model_state_dict`` expects it there
      for the embedding-resize check during checkpoint saving).
    """
    # name_or_path for TRL/vLLM checkpoint lookup
    if not getattr(full_model.vlm, "name_or_path", ""):
        full_model.vlm.name_or_path = full_model.config.vlm_name_or_path
    if not getattr(full_model.vlm.config, "_name_or_path", ""):
        full_model.vlm.config._name_or_path = full_model.config.vlm_name_or_path

    # Qwen3VLConfig keeps vocab_size inside text_config, but PEFT loads a
    # fresh config via from_pretrained() and accesses .vocab_size directly.
    vlm_config_cls = type(full_model.vlm.config)
    if not hasattr(vlm_config_cls, "_patched_vocab_size"):
        _original_init = vlm_config_cls.__init__

        def _patched_init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            if not hasattr(self, "vocab_size") and hasattr(self, "text_config"):
                self.vocab_size = self.text_config.vocab_size

        vlm_config_cls.__init__ = _patched_init
        vlm_config_cls._patched_vocab_size = True


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

        # Value head (optional scene-level baseline)
        self.value_head = None
        self.value_optimizer = None
        self._value_h0_stash: list[torch.Tensor] = []  # h_0 per completion (CPU)
        self._value_rewards_stash: list[float] = []  # composite reward per completion
        self._value_reward_weights: list[float] = [0.5, 0.25, 0.25]  # traj/reasoning/consistency
        self._value_pretrain_remaining: int = 0
        self._value_save_path: str | None = None

        _vh_cfg = value_head_cfg or {}
        if _vh_cfg.get("enabled", False):
            from alpamayo_r1.training.value_head import SceneValueHead

            _hidden_dim = int(_vh_cfg.get("hidden_dim", 4096))
            _vh_device = self.accelerator.device
            self.value_head = SceneValueHead(_hidden_dim).to(_vh_device)
            self.value_optimizer = torch.optim.Adam(
                self.value_head.parameters(),
                lr=float(_vh_cfg.get("lr", 1e-4)),
            )
            self._value_pretrain_remaining = int(_vh_cfg.get("pretrain_steps", 0))
            self._value_save_path = _vh_cfg.get("save_path", None) or None

            # Load pre-trained weights if a checkpoint path is provided
            _load_path = _vh_cfg.get("load_path", None) or None
            if _load_path is not None:
                import os

                if os.path.isfile(_load_path):
                    state = torch.load(_load_path, map_location=_vh_device)
                    self.value_head.load_state_dict(state)
                    logger.info("Loaded SceneValueHead weights from %s", _load_path)
                else:
                    logger.warning(
                        "value_head.load_path=%s not found — starting from random init",
                        _load_path,
                    )

            logger.info(
                "SceneValueHead enabled: hidden_dim=%d, lr=%.2e, pretrain_steps=%d, device=%s",
                _hidden_dim,
                _vh_cfg.get("lr", 1e-4),
                self._value_pretrain_remaining,
                _vh_device,
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
        """Extend default save to also persist value head weights."""
        super()._save(output_dir, state_dict=state_dict)

        if self.value_head is not None and self._value_save_path is not None:
            import os

            os.makedirs(os.path.dirname(os.path.abspath(self._value_save_path)), exist_ok=True)
            torch.save(self.value_head.state_dict(), self._value_save_path)
            logger.info("Saved SceneValueHead weights to %s", self._value_save_path)

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

        # Clear value head stashes at the start of each new rollout
        if self.value_head is not None:
            self._value_h0_stash.clear()
            self._value_rewards_stash.clear()

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
                    with torch.no_grad():
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

                        # Stash h_0 (same scene embedding for all G completions of this scene)
                        if self.value_head is not None:
                            self._value_h0_stash.append(scene_h0)  # (1, hidden_dim) CPU tensor

                        # Trajectory data for reward computation
                        pred_traj = pred_xyz_tensor[sample_idx].cpu().numpy().flatten().tolist()
                        gt_traj = ego_future_xyz[0, 0].numpy().flatten().tolist()
                        all_pred_xyz.append(pred_traj)
                        all_gt_xyz.append(gt_traj)

                    # 7. Compute log-probs via teacher-forced VLM forward (batched)
                    batch_logprobs = _compute_batch_logprobs(
                        self.full_model,
                        model_inputs,
                        prompt_input_ids,
                        all_completion_ids[-local_gen_count:],
                        prompt_len,
                        device,
                        mini_batch_size=self.logprob_mini_batch_size,
                    )
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
            with torch.no_grad():
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
        return result

    def _stash_value_rewards(self, inputs: list[dict], completions: list[str]) -> None:
        """Compute composite rewards and stash them for value head training.

        Args:
            inputs: Per-sample input dicts (must have pred_xyz and gt_xyz populated).
            completions: CoC text strings, one per sample.
        """
        from alpamayo_r1.training.rewards import (
            consistency_reward,
            reasoning_quality_reward,
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
            self._value_rewards_stash.append(composite)

    def _train_value_head_step(self, batch_size: int) -> None:
        """Consume one batch from the stash and update the value head.

        Pops up to ``batch_size`` (h_0, reward) pairs from the stashes,
        runs an MSE update via the separate value optimizer, and accumulates
        metrics into ``self._metrics["train"]`` so they appear in TRL's log.
        No-ops silently if the stash is empty (e.g. vLLM path before h_0
        collection is wired in).

        Args:
            batch_size: Number of samples to consume from the stash.
        """
        n = min(batch_size, len(self._value_h0_stash), len(self._value_rewards_stash))
        if n == 0:
            logger.debug(
                "value head stash empty (h0=%d, rewards=%d) — skipping update",
                len(self._value_h0_stash),
                len(self._value_rewards_stash),
            )
            return

        device = self.accelerator.device

        h0_batch = self._value_h0_stash[:n]
        rewards_batch = self._value_rewards_stash[:n]
        self._value_h0_stash = self._value_h0_stash[n:]
        self._value_rewards_stash = self._value_rewards_stash[n:]

        h0_tensor = torch.cat(h0_batch, dim=0).to(device)  # (n, hidden_dim)
        rewards_tensor = torch.tensor(rewards_batch, dtype=torch.float32, device=device)

        v_pred = self.value_head(h0_tensor)  # (n,)
        value_loss = F.mse_loss(v_pred, rewards_tensor)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Accumulate into TRL's metric dict so values appear in the training log.
        # self._metrics["train"] is a defaultdict(list) maintained by GRPOTrainer.
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["value_head/loss"].append(value_loss.item())
        self._metrics[mode]["value_head/pred_mean"].append(v_pred.detach().mean().item())
        self._metrics[mode]["value_head/target_mean"].append(rewards_tensor.mean().item())
        is_pretrain = self._value_pretrain_remaining > 0
        self._metrics[mode]["value_head/pretrain_steps_remaining"].append(
            float(self._value_pretrain_remaining)
        )
        logger.debug(
            "value head%s | loss=%.4f pred=%.3f target=%.3f",
            " [pretrain]" if is_pretrain else "",
            value_loss.item(),
            v_pred.detach().mean().item(),
            rewards_tensor.mean().item(),
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to train value head alongside (or instead of) the GRPO policy loss.

        **Stage 0** (``_value_pretrain_remaining > 0``): only the value head
        trains.  The policy loss is replaced by a zero scalar so the VLM
        receives no gradient.  The expensive GRPO forward pass is skipped
        entirely for efficiency.  The counter decrements each call.

        **Stage 1** (``_value_pretrain_remaining == 0``): normal GRPO plus a
        value head update from the stash.

        When the value head is disabled, delegates to the parent unchanged.
        """
        if self.value_head is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        batch_size = inputs["input_ids"].shape[0] if "input_ids" in inputs else 1
        self._train_value_head_step(batch_size)

        # Stage 0: skip GRPO policy update
        if self._value_pretrain_remaining > 0:
            self._value_pretrain_remaining -= 1
            zero_loss = torch.tensor(0.0, requires_grad=True, device=self.accelerator.device)
            if return_outputs:
                return zero_loss, None
            return zero_loss

        # Stage 1: normal GRPO
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


def _compute_batch_logprobs(
    full_model: AlpamayoR1,
    model_inputs: dict,
    prompt_input_ids: torch.Tensor,
    completion_ids_list: list[list[int]],
    prompt_len: int,
    device: torch.device,
    mini_batch_size: int = 4,
) -> list[list[float]]:
    """Compute per-token log-probs for a batch of completions.

    Processes completions in groups of ``mini_batch_size`` to reduce the number
    of VLM forward passes (e.g. 2 passes for 8 completions with batch size 4,
    vs 8 serial passes in the original implementation).

    Args:
        full_model: The full AlpamayoR1 model.
        model_inputs: Dict with tokenized_data (input_ids may be popped).
        prompt_input_ids: Saved prompt input_ids, shape (1, L_prompt).
        completion_ids_list: List of completion token ID lists.
        prompt_len: Number of prompt tokens.
        device: CUDA device.
        mini_batch_size: Number of completions to process per forward pass.

    Returns:
        List of per-token log-prob lists, one per completion.
    """
    results: list[list[float]] = []
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
            outputs = full_model.vlm(input_ids=full_ids, **forward_kwargs)

        # Extract per-token log-probs for each completion in the mini-batch.
        # Logits at position t predict token t+1, so completion tokens at
        # positions [prompt_len .. prompt_len+comp_len) are predicted by
        # logits at [prompt_len-1 .. prompt_len-1+comp_len).
        for i, (comp_ids, comp_len) in enumerate(zip(batch_comp_ids, comp_lens)):
            if not comp_ids:
                results.append([])
                continue
            logits = outputs.logits[i, prompt_len - 1 : prompt_len - 1 + comp_len]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            comp_target = comp_tensors[i][:comp_len]
            token_log_probs = log_probs.gather(1, comp_target.unsqueeze(-1)).squeeze(-1)
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
