"""Shared utilities for rollout generation and training pipelines.

This module contains components extracted from rollout.py that are reusable
across both GRPO and SFT training pipelines. Moving them here avoids
duplication and allows the SFT pipeline to import them without pulling in
the entire GRPOTrainer dependency chain.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from physical_ai_av import PhysicalAIAVDatasetInterface
from PIL import Image
from transformers import AutoProcessor

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ClipDataCache
# ---------------------------------------------------------------------------


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
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch_future: Future | None = None
        self._prefetch_lock = threading.Lock()

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
            "ego_future_rot": data["ego_future_rot"],
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

    def prefetch(self, clip_id: str, t0_us: int) -> None:
        """Submit a background load for (clip_id, t0_us) if not already cached.

        The load runs in a single-thread executor so it overlaps with GPU work.
        Call this for the *next* scene before starting generation on the current one.
        """
        key = (clip_id, t0_us)
        with self._prefetch_lock:
            if key in self._cache:
                return
            # Wait for any in-flight prefetch before starting a new one
            if self._prefetch_future is not None and not self._prefetch_future.done():
                self._prefetch_future.result()
            self._prefetch_future = self._prefetch_executor.submit(
                self._load_and_cache, clip_id, t0_us
            )

    def _wait_for_prefetch(self) -> None:
        """Block until any in-flight prefetch completes."""
        with self._prefetch_lock:
            if self._prefetch_future is not None and not self._prefetch_future.done():
                self._prefetch_future.result()

    def get(self, clip_id: str, t0_us: int, device: torch.device) -> tuple[dict, torch.Tensor]:
        """Return (model_inputs_on_device, ego_future_xyz_cpu).

        model_inputs are freshly moved to device each call (safe to modify).
        ego_future_xyz stays on CPU for numpy compatibility.
        If a prefetch is in-flight for this key, waits for it to finish.
        """
        key = (clip_id, t0_us)
        if key not in self._cache:
            self._wait_for_prefetch()
        if key not in self._cache:
            self._load_and_cache(clip_id, t0_us)
        else:
            self._hits += 1
            self._cache.move_to_end(key)  # mark as recently used
        cached = self._cache[key]
        return helper.to_device(cached["model_inputs"], device=device), cached["ego_future_xyz"]

    def get_ego_future_rot(self, clip_id: str, t0_us: int) -> torch.Tensor:
        """Return cached ego_future_rot tensor for a clip.

        Returns:
            ego_future_rot: shape (1, 1, T_future, 3, 3), CPU tensor.
        """
        key = (clip_id, t0_us)
        if key not in self._cache:
            self._load_and_cache(clip_id, t0_us)
        else:
            self._cache.move_to_end(key)
        return self._cache[key]["ego_future_rot"]

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


# ---------------------------------------------------------------------------
# Prompt utilities
# ---------------------------------------------------------------------------


def group_consecutive_prompts(prompts: list) -> list[tuple[Any, int]]:
    """Group consecutive identical prompts and return (prompt, count) pairs.

    TRL repeats each prompt ``num_generations`` times, but distributed training
    may split those repetitions across GPUs.  This helper detects the *actual*
    local repeat count instead of assuming ``num_generations`` copies are present.
    """
    groups: list[tuple[Any, int]] = []
    for p in prompts:
        if groups and prompt_eq(groups[-1][0], p):
            groups[-1] = (groups[-1][0], groups[-1][1] + 1)
        else:
            groups.append((p, 1))
    return groups


def prompt_eq(a: Any, b: Any) -> bool:
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


def parse_clip_metadata(prompt_text: str) -> tuple[str, int]:
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


def collapse_image_pad_tokens(
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


# ---------------------------------------------------------------------------
# VLM preparation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation for a single trajectory.

    Args:
        rewards: Per-timestep rewards, shape (T,).
        values: Value estimates at each timestep, shape (T,).
        gamma: Discount factor.
        lam: GAE trace decay parameter.

    Returns:
        Advantages, shape (T,).
    """
    T = rewards.shape[0]
    if T == 0:
        return torch.zeros(0, device=rewards.device)

    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            # Terminal: V(s_{T+1}) = 0
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages
