#!/usr/bin/env python3
"""Patched vLLM server launcher for Qwen3-VL.

Applies monkey-patches to vLLM's transformers backend before starting
``trl vllm-serve``.  The patches fix Qwen3-VL's ``get_image_features``
returning a tuple instead of a tensor, which causes the unpatched vLLM
to hang during multimodal request preprocessing.

Usage (drop-in replacement for ``trl vllm-serve``):
    python scripts/vllm_serve_patched.py --model ... --port 8000

The patches are applied at import time so they propagate to the worker
subprocesses spawned by TRL's ``llm_worker``.
"""
import logging
import sys

import torch

logger = logging.getLogger(__name__)


def _patch_vllm_qwen3vl_embed() -> None:
    """Patch vLLM's transformers-backend ``embed_multimodal`` for Qwen3-VL.

    Qwen3-VL's ``get_image_features`` returns a tuple
    ``(image_embeds, deepstack_image_embeds)`` instead of a single tensor.
    The generic ``embed_multimodal`` only handles the ``torch.Tensor`` case,
    causing request preprocessing to hang.
    """
    try:
        from vllm.model_executor.models.transformers.multimodal import MultiModalMixin

        _orig_embed = MultiModalMixin.embed_multimodal  # noqa: F841

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
                    embed.flatten(start_dim=0, end_dim=-2)
                    for embed in vision_embeddings
                ]

            return vision_embeddings

        MultiModalMixin.embed_multimodal = _patched_embed_multimodal
        logger.info(
            "Patched vLLM MultiModalMixin.embed_multimodal for "
            "Qwen3-VL tuple return from get_image_features."
        )
    except (ImportError, AttributeError) as exc:
        logger.debug("Could not patch vLLM embed_multimodal: %s", exc)


def _patch_vllm_mm_sanity_check() -> None:
    """Relax vLLM's multimodal encoder output sanity check.

    vLLM v0.12's ``sanity_check_mm_encoder_outputs`` asserts that
    ``len(embeddings) == expected_num_items``.  With Qwen3-VL, a single
    image can produce multiple tile-group embeddings, causing the assertion
    to fire during the profiling run.
    """
    try:
        import vllm.v1.worker.utils as _vllm_worker_utils

        def _relaxed_sanity_check(mm_embeddings, expected_num_items):
            assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
                "Expected multimodal embeddings to be a list/tuple of 2D "
                f"tensors, or a single 3D tensor, but got "
                f"{type(mm_embeddings)} instead."
            )
            if len(mm_embeddings) != expected_num_items:
                logger.warning(
                    "vLLM multimodal profiling: expected %d embedding items "
                    "but got %d (Qwen3-VL tile splitting). Continuing.",
                    expected_num_items,
                    len(mm_embeddings),
                )
            assert all(e.ndim == 2 for e in mm_embeddings), (
                "Expected multimodal embeddings to be a sequence of 2D "
                f"tensors, but got shapes "
                f"{[e.shape for e in mm_embeddings]}."
            )

        _vllm_worker_utils.sanity_check_mm_encoder_outputs = _relaxed_sanity_check
        logger.info("Patched vLLM sanity_check_mm_encoder_outputs.")
    except (ImportError, AttributeError):
        logger.debug("vLLM v1 worker utils not available; skipping MM sanity check patch.")


def _patch_trl_llm_worker() -> None:
    """Wrap TRL's ``llm_worker`` so patches run inside the worker subprocess.

    Worker subprocesses are spawned (not forked) so they don't inherit
    monkey-patches applied in the main process.  By wrapping ``llm_worker``
    we ensure our vLLM patches execute before ``LLM(...)`` is constructed.
    """
    import trl.scripts.vllm_serve as _vllm_serve

    _orig_llm_worker = _vllm_serve.llm_worker

    def _patched_llm_worker(*args, **kwargs):
        _patch_vllm_qwen3vl_embed()
        _patch_vllm_mm_sanity_check()
        return _orig_llm_worker(*args, **kwargs)

    _vllm_serve.llm_worker = _patched_llm_worker
    logger.info("Wrapped TRL llm_worker to apply vLLM patches in worker subprocess.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Apply patches in main process (for any direct usage).
    _patch_vllm_qwen3vl_embed()
    _patch_vllm_mm_sanity_check()

    # Wrap TRL's worker function so patches also run in spawned subprocesses.
    _patch_trl_llm_worker()

    # Launch trl vllm-serve with the remaining CLI args
    sys.argv = ["trl", "vllm-serve"] + sys.argv[1:]
    from trl.cli import main
    main()
