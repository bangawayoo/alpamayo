

"""
Optimized evaluation script with batched processing and parallel data loading.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Must be set before any CUDA import/init.  In MIG environments the default
# CUDA caching allocator queries NVML for per-device free memory, which fails
# with "NVML_SUCCESS == r INTERNAL ASSERT FAILED".  expandable_segments uses
# a different allocation strategy that skips those NVML calls entirely.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from physical_ai_av import PhysicalAIAVDatasetInterface
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.inference import generate_coc, prepare_vlm_inputs
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.token_utils import extract_traj_tokens
from alpamayo_r1.training.advantage_conditioning import compute_advantage_token_ids

# Pin to avoid breaking changes when the upstream HF dataset is updated.
# Must match the revision in training/configs/grpo_default.yaml.
DEFAULT_DATASET_REVISION = "05e158af89ba"


class AlpamayoDataset(Dataset):
    """Dataset for batched loading of Alpamayo samples."""

    def __init__(self, clip_ids, t0_us=5_100_000, revision=None):
        self.clip_ids = clip_ids
        self.t0_us = t0_us
        self.revision = revision
        # Each worker process needs its own AVDI instance
        self.avdi = None

    def __len__(self):
        return len(self.clip_ids)

    def _get_avdi(self):
        """Lazy initialization of AVDI per worker.

        Passes the pre-resolved revision so workers don't each call
        ``list_repo_refs()`` (avoids redundant HF API requests).
        """
        if self.avdi is None:
            self.avdi = PhysicalAIAVDatasetInterface(revision=self.revision)
        return self.avdi

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        try:
            avdi = self._get_avdi()
            data = load_physical_aiavdataset(
                clip_id=clip_id,
                t0_us=self.t0_us,
                avdi=avdi,
                maybe_stream=True,
            )
            return {
                "clip_id": clip_id,
                "image_frames": data["image_frames"],
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
                "ego_future_xyz": data["ego_future_xyz"],
                "ego_future_rot": data["ego_future_rot"],
                "success": True,
                "error": None,
            }
        except Exception as e:
            return {
                "clip_id": clip_id,
                "image_frames": None,
                "ego_history_xyz": None,
                "ego_history_rot": None,
                "ego_future_xyz": None,
                "ego_future_rot": None,
                "success": False,
                "error": str(e),
            }


def collate_fn(batch):
    """Custom collate function that handles failed samples."""
    return batch


def compute_minADE(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> float:
    """Compute minimum Average Displacement Error (minADE)."""
    gt_xy = gt_xyz.cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    return float(min_ade)


def compute_minFDE(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> float:
    """Compute minimum Final Displacement Error (minFDE)."""
    gt_xy_final = gt_xyz.cpu()[0, 0, -1, :2].numpy()
    pred_xy_final = pred_xyz.cpu().numpy()[0, 0, :, -1, :2]
    diff = np.linalg.norm(pred_xy_final - gt_xy_final[None, ...], axis=1)
    min_fde = diff.min()
    return float(min_fde)


logger = logging.getLogger(__name__)


def _plot_trajectories(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    clip_id: str,
    save_path: Path,
    coc_text: str | None = None,
    min_ade: float | None = None,
    min_fde: float | None = None,
) -> None:
    """Plot predicted vs. ground-truth trajectories in BEV and save to disk.

    Args:
        pred_xyz: Predicted trajectories, shape ``(1, 1, N, T, 3)``.
        gt_xyz: Ground-truth trajectory, shape ``(1, 1, T, 3)``.
        clip_id: Clip identifier (used in title / filename).
        save_path: Directory to save the plot.
        coc_text: Optional Chain-of-Causation text to display.
        min_ade: Optional minADE value to display.
        min_fde: Optional minFDE value to display.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_xy = gt_xyz.cpu()[0, 0, :, :2].T.numpy()
    n_samples = pred_xyz.shape[2]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot predicted trajectories
    cmap = plt.cm.tab10
    for i in range(n_samples):
        pred_xy = pred_xyz.cpu()[0, 0, i, :, :2].T.numpy()
        # Rotate 90 CCW for BEV: (-y, x) so "forward" points up
        ax.plot(-pred_xy[1], pred_xy[0], "o-", color=cmap(i), markersize=3,
                label=f"Pred #{i + 1}", alpha=0.7)

    ax.plot(-gt_xy[1], gt_xy[0], "r-", linewidth=2.5, label="Ground Truth")
    ax.plot(0, 0, "k*", markersize=15, label="Ego (t=0)")

    ax.set_xlabel("Lateral (m)")
    ax.set_ylabel("Longitudinal (m)")
    ax.set_aspect("equal")
    # Pad shorter axis to at least 30% of the longer axis to avoid narrow plots
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    min_range = max(x_range, y_range) * 0.3
    if x_range < min_range:
        mid = (xlim[0] + xlim[1]) / 2
        ax.set_xlim(mid - min_range / 2, mid + min_range / 2)
    if y_range < min_range:
        mid = (ylim[0] + ylim[1]) / 2
        ax.set_ylim(mid - min_range / 2, mid + min_range / 2)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    title = f"clip: {clip_id[:12]}..."
    if min_ade is not None:
        title += f"  |  minADE={min_ade:.3f}m"
    if min_fde is not None:
        title += f"  minFDE={min_fde:.3f}m"
    ax.set_title(title, fontsize=10)

    # Add CoC text below the plot
    if coc_text:
        wrapped = coc_text[:300] + ("..." if len(coc_text) > 300 else "")
        fig.text(0.05, 0.01, f"CoC: {wrapped}", fontsize=7, wrap=True,
                 verticalalignment="bottom", family="monospace")
        fig.subplots_adjust(bottom=0.12)

    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path / f"{clip_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _evaluate_expert_with_adv_traj(
    model: AlpamayoR1,
    model_inputs: dict,
    num_traj_samples: int,
    temperature: float,
    top_p: float,
    adv_traj_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Evaluate with expert mode + adv_traj injection via batched pipeline.

    Uses ``run_batched_expert_diffusion`` from sft_rollout for the shared
    TF forward + adv_traj injection + expert diffusion phases.
    """
    from alpamayo_r1.training.sft_rollout import run_batched_expert_diffusion

    device = next(model.vlm.parameters()).device

    # 1. Fuse history trajectory tokens
    input_ids, gen_kwargs = prepare_vlm_inputs(model, model_inputs)
    prompt_len = input_ids.shape[1]

    hist_xyz = model_inputs["ego_history_xyz"][:, -1]  # (1, T, 3)
    hist_rot = model_inputs["ego_history_rot"][:, -1]  # (1, T, 3, 3)

    # 2. Batch AR CoC generation
    traj_future_start_id = model.special_token_ids["traj_future_start"]
    pad_token_id = model.tokenizer.pad_token_id

    vlm_output = generate_coc(
        model,
        input_ids,
        gen_kwargs,
        mode="expert",
        temperature=temperature,
        top_p=top_p,
        num_samples=num_traj_samples,
        max_new_tokens=256 + 10,
        pad_token_id=pad_token_id,
    )
    generated_seqs = vlm_output[:, prompt_len:]

    # Log raw VLM generation for debugging
    for si in range(generated_seqs.shape[0]):
        raw_ids = generated_seqs[si].cpu().tolist()
        # Strip padding for readable log
        while raw_ids and raw_ids[-1] == pad_token_id:
            raw_ids.pop()
        logger.debug(
            "[expert_adv_traj] sample %d raw generation (%d tokens): %s",
            si,
            len(raw_ids),
            model.tokenizer.decode(raw_ids, skip_special_tokens=False),
        )

    # 3. Parse valid completions
    valid_items: list[dict] = []
    for sample_idx in range(num_traj_samples):
        raw = generated_seqs[sample_idx].cpu().tolist()
        while raw and raw[-1] == pad_token_id:
            raw.pop()
        try:
            traj_pos = raw.index(traj_future_start_id)
        except ValueError:
            logger.warning("No <traj_future_start> in sample %d, skipping", sample_idx)
            continue
        coc_tokens = raw[:traj_pos]
        valid_items.append(
            {
                "coc_tokens": coc_tokens,
                "coc_prefix": coc_tokens + [traj_future_start_id],
                "coc_text": model.tokenizer.decode(
                    coc_tokens, skip_special_tokens=True
                ).strip(),
            }
        )
        logger.debug(
            f"[CoC generated] {model.tokenizer.decode(coc_tokens, skip_special_tokens=False)}"
        )

    if not valid_items:
        raise RuntimeError("All trajectory samples failed (no <traj_future_start> found)")

    n_valid = len(valid_items)

    # 4. Build right-padded TF sequences
    full_lens = [prompt_len + len(item["coc_prefix"]) for item in valid_items]
    max_full_len = max(full_lens)

    tf_input_ids = torch.full(
        (n_valid, max_full_len), pad_token_id, dtype=torch.long, device=device
    )
    tf_attention_mask = torch.zeros(n_valid, max_full_len, dtype=torch.long, device=device)

    for i, item in enumerate(valid_items):
        prefix = torch.tensor(item["coc_prefix"], device=device, dtype=torch.long)
        actual_len = prompt_len + prefix.shape[0]
        tf_input_ids[i, :prompt_len] = input_ids[0]
        tf_input_ids[i, prompt_len:actual_len] = prefix
        tf_attention_mask[i, :actual_len] = 1

    # Replicate pixel_values and image_grid_thw for each valid completion
    pv = gen_kwargs.get("pixel_values")
    igt = gen_kwargs.get("image_grid_thw")
    pv_tensor = pv.repeat(n_valid, *([1] * (pv.dim() - 1))) if pv is not None else None
    igt_tensor = igt.repeat(n_valid, 1) if igt is not None else None

    # 5. Run batched TF + adv_traj injection + expert diffusion
    sampled_action = run_batched_expert_diffusion(
        model=model,
        tf_input_ids=tf_input_ids,
        tf_attention_mask=tf_attention_mask,
        full_lens=full_lens,
        pixel_values=pv_tensor,
        image_grid_thw=igt_tensor,
        adv_traj_token_id=adv_traj_token_id,
        expert_non_causal=model.config.expert_non_causal_attention,
    )

    # 6. Batched action_to_traj
    hist_xyz_batch = hist_xyz.expand(n_valid, -1, -1)  # (n_valid, T, 3)
    hist_rot_batch = hist_rot.expand(n_valid, -1, -1, -1)  # (n_valid, T, 3, 3)

    pred_xyz, pred_rot = model.action_space.action_to_traj(
        sampled_action, hist_xyz_batch, hist_rot_batch
    )

    # Reshape: (n_valid, T, 3) -> (1, 1, n_valid, T, 3)
    pred_xyz = pred_xyz.unsqueeze(0).unsqueeze(0)
    pred_rot = pred_rot.unsqueeze(0).unsqueeze(0)

    coc_texts = [item["coc_text"] for item in valid_items]
    extra = {"cot": np.array(coc_texts).reshape(1, 1, -1)}
    return pred_xyz, pred_rot, extra


def _evaluate_vlm_only_with_adv_traj(
    model: AlpamayoR1,
    model_inputs: dict,
    num_traj_samples: int,
    temperature: float,
    top_p: float,
    adv_traj_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Evaluate with VLM-only mode + adv_traj injection via teacher-forced KV cache.

    1. Fuse history tokens, batch AR CoC generation (StopAfterEOS at traj_future_start)
    2. Per-sample: teacher-force [input_ids + CoC + adv_traj] to rebuild KV cache
    3. Per-sample: continue AR from TFS with KV cache (StopAfterEOS at traj_future_end)
    4. Extract trajectory tokens, decode to continuous xyz/rot
    """
    from transformers import GenerationConfig, StoppingCriteriaList

    from alpamayo_r1.models.token_utils import StopAfterEOS

    device = next(model.vlm.parameters()).device

    # 1. Fuse history trajectory tokens
    input_ids, gen_kwargs = prepare_vlm_inputs(model, model_inputs)
    prompt_len = input_ids.shape[1]

    hist_xyz = model_inputs["ego_history_xyz"][:, -1]  # (1, T, 3)
    hist_rot = model_inputs["ego_history_rot"][:, -1]  # (1, T, 3, 3)

    # 2. Batch AR CoC generation (stop at traj_future_start)
    traj_future_start_id = model.special_token_ids["traj_future_start"]
    traj_future_end_id = model.special_token_ids["traj_future_end"]
    tokens_per_future_traj = model.config.tokens_per_future_traj
    pad_token_id = model.tokenizer.pad_token_id

    vlm_output = generate_coc(
        model,
        input_ids,
        gen_kwargs,
        mode="expert",
        temperature=temperature,
        top_p=top_p,
        num_samples=num_traj_samples,
        max_new_tokens=256 + 10,
        pad_token_id=pad_token_id,
    )
    generated_seqs = vlm_output[:, prompt_len:]

    # Log raw VLM generation for debugging
    for si in range(generated_seqs.shape[0]):
        raw_ids = generated_seqs[si].cpu().tolist()
        while raw_ids and raw_ids[-1] == pad_token_id:
            raw_ids.pop()
        logger.debug(
            "[vlm_adv_traj] sample %d raw generation (%d tokens): %s",
            si,
            len(raw_ids),
            model.tokenizer.decode(raw_ids, skip_special_tokens=False),
        )

    # 3. Per-sample: teacher-forced + adv_traj + continued AR
    pred_xyz_list = []
    pred_rot_list = []
    coc_texts = []
    vlm = model.vlm

    # Disable gradient checkpointing for teacher-forced forwards
    gc_modules = [m for m in vlm.modules() if getattr(m, "gradient_checkpointing", False)]
    for m in gc_modules:
        m.gradient_checkpointing = False

    try:
        for sample_idx in range(num_traj_samples):
            raw_completion = generated_seqs[sample_idx].cpu().tolist()

            # Strip padding
            while raw_completion and raw_completion[-1] == pad_token_id:
                raw_completion.pop()

            # Find <traj_future_start>
            try:
                traj_start_pos = raw_completion.index(traj_future_start_id)
            except ValueError:
                logger.warning(
                    "No <traj_future_start> in sample %d, skipping", sample_idx
                )
                continue

            coc_tokens = raw_completion[:traj_start_pos]
            coc_text = model.tokenizer.decode(coc_tokens, skip_special_tokens=True).strip()
            coc_texts.append(coc_text)

            # Teacher-forced: [input_ids + CoC + adv_traj]
            # We include adv_traj before TFS so the KV cache has the conditioning
            traj_ids = (
                [adv_traj_token_id]
                if isinstance(adv_traj_token_id, int)
                else list(adv_traj_token_id)
            )
            completion_prefix_ids = coc_tokens + traj_ids
            prefix_tensor = torch.tensor(
                [completion_prefix_ids], device=device, dtype=torch.long
            )
            full_ids = torch.cat([input_ids, prefix_tensor], dim=1)

            tf_kwargs = {}
            if "attention_mask" in gen_kwargs:
                orig_mask = gen_kwargs["attention_mask"]
                prefix_mask = torch.ones(
                    1, len(completion_prefix_ids), device=device, dtype=orig_mask.dtype
                )
                tf_kwargs["attention_mask"] = torch.cat([orig_mask, prefix_mask], dim=1)
            for k in ("pixel_values", "image_grid_thw"):
                if k in gen_kwargs:
                    tf_kwargs[k] = gen_kwargs[k]

            with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
                tf_out = vlm(
                    input_ids=full_ids,
                    use_cache=True,
                    **tf_kwargs,
                )

            prompt_cache = tf_out.past_key_values

            # Continue AR from TFS token using KV cache
            tfs_tensor = torch.tensor(
                [[traj_future_start_id]], device=device, dtype=torch.long
            )
            cont_gen_config = GenerationConfig(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                max_new_tokens=tokens_per_future_traj + 10,
                pad_token_id=pad_token_id,
            )
            cont_stopping = StoppingCriteriaList(
                [StopAfterEOS(eos_token_id=traj_future_end_id)]
            )

            with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
                cont_output = vlm.generate(
                    input_ids=tfs_tensor,
                    past_key_values=prompt_cache,
                    generation_config=cont_gen_config,
                    stopping_criteria=cont_stopping,
                )
            # cont_output: (1, 1 + generated_len) — starts with TFS

            # Build full sequence for traj token extraction:
            # [... traj_future_start, <traj_tokens>, traj_future_end]
            full_seq = cont_output  # already starts with traj_future_start_id

            traj_tokens = extract_traj_tokens(
                full_seq,
                model.special_token_ids,
                tokens_per_future_traj,
                model.future_token_start_idx,
                model.config.traj_vocab_size,
            )

            pred_xyz, pred_rot, _ = model.traj_tokenizer.decode(
                hist_xyz, hist_rot, traj_tokens
            )
            pred_xyz_list.append(pred_xyz)
            pred_rot_list.append(pred_rot)

            del prompt_cache

    finally:
        for m in gc_modules:
            m.gradient_checkpointing = True

    if not pred_xyz_list:
        raise RuntimeError("All trajectory samples failed (no <traj_future_start> found)")

    # Stack: (1, 1, num_ok_samples, T, 3) / (1, 1, num_ok_samples, T, 3, 3)
    pred_xyz = torch.stack(pred_xyz_list, dim=0).unsqueeze(0).unsqueeze(0)
    pred_rot = torch.stack(pred_rot_list, dim=0).unsqueeze(0).unsqueeze(0)
    pred_xyz = pred_xyz.squeeze(3)
    pred_rot = pred_rot.squeeze(3)

    extra = {"cot": np.array(coc_texts).reshape(1, 1, -1) if coc_texts else None}
    return pred_xyz, pred_rot, extra


def _compute_reward_signals(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    coc_text: str | None,
) -> tuple[float, float, float]:
    """Compute trajectory, reasoning, and consistency reward signals.

    Args:
        pred_xyz: Predicted trajectories, shape (1, 1, S, T, 3).
        gt_xyz: Ground-truth trajectory, shape (1, 1, T, 3).
        coc_text: Chain-of-causation text (may be None).

    Returns:
        (r_traj, r_reason, r_consist): Mean reward across trajectory samples.
    """
    from alpamayo_r1.training.rewards import (
        consistency_reward,
        reasoning_quality_reward,
        trajectory_quality_reward,
    )

    pred_np = pred_xyz.cpu().numpy()[0, 0]  # (S, T, 3)
    gt_np = gt_xyz.cpu().numpy()[0, 0]  # (T, 3)
    gt_flat = gt_np.flatten().tolist()

    n_samples = pred_np.shape[0]
    pred_flats = [pred_np[i].flatten().tolist() for i in range(n_samples)]
    completions = [coc_text or ""] * n_samples

    r_traj = trajectory_quality_reward(completions, pred_flats, [gt_flat] * n_samples)
    r_reason = reasoning_quality_reward(completions)
    r_consist = consistency_reward(completions, pred_flats)

    return float(np.mean(r_traj)), float(np.mean(r_reason)), float(np.mean(r_consist))


def evaluate_batch(
    model: AlpamayoR1,
    processor,
    batch: list,
    num_traj_samples: int,
    temperature: float,
    top_p: float,
    device: str,
    t0_us: int = 5_100_000,
    traj_mode: str = "expert",
    adv_obs_token_id: int | None = None,
    adv_traj_token_id: int | None = None,
    output_dir: str | None = None,
    visualize: bool = False,
) -> list:
    """Evaluate a batch of samples.

    Args:
        traj_mode: Trajectory generation mode.
            ``"expert"`` uses the full VLM + Expert + Diffusion pipeline.
            ``"vlm"`` uses VLM-only discrete trajectory token generation.
        adv_obs_token_id: If set, append this token to input_ids before generation.
        adv_traj_token_id: If set, inject this token between CoC and trajectory
            via teacher-forced KV cache rebuild (per-sample processing).
        visualize: If True, save BEV trajectory plots to output_dir/plots/.
    """
    results = []

    for sample in batch:
        if not sample["success"]:
            results.append(
                {
                    "clip_id": sample["clip_id"],
                    "t0_us": t0_us,
                    "minADE": None,
                    "minFDE": None,
                    "success": False,
                    "error": sample["error"],
                    "coc": None,
                }
            )
            continue

        try:
            # Prepare inputs
            model_inputs = helper.prepare_model_inputs(sample, processor, device)

            # Append adv_obs token(s) to input_ids if requested
            if adv_obs_token_id is not None:
                input_ids = model_inputs["tokenized_data"]["input_ids"]
                # Support both single ID (sentinel) and list of IDs (text mode)
                obs_ids = (
                    [adv_obs_token_id]
                    if isinstance(adv_obs_token_id, int)
                    else list(adv_obs_token_id)
                )
                adv_obs_tensor = torch.tensor(
                    [obs_ids], device=input_ids.device, dtype=input_ids.dtype
                )
                model_inputs["tokenized_data"]["input_ids"] = torch.cat(
                    [input_ids, adv_obs_tensor], dim=1
                )
                if "attention_mask" in model_inputs["tokenized_data"]:
                    attn = model_inputs["tokenized_data"]["attention_mask"]
                    model_inputs["tokenized_data"]["attention_mask"] = torch.cat(
                        [
                            attn,
                            torch.ones(
                                1, len(obs_ids), device=attn.device, dtype=attn.dtype
                            ),
                        ],
                        dim=1,
                    )

            # Log decoded prompt and verify adv token insertion
            _ids = model_inputs["tokenized_data"]["input_ids"][0].tolist()
            logger.debug(
                "[%s] decoded input (%d tokens):\n%s\n%s",
                sample["clip_id"],
                len(_ids),
                model.tokenizer.decode(_ids[-30:], skip_special_tokens=False),
                _ids[-30:],
            )

            if adv_obs_token_id is not None:
                has_obs = any(tid in _ids for tid in obs_ids)
                if not has_obs:
                    logger.warning(
                        "adv_obs token NOT found in input_ids for %s "
                        "(expected one of %s in %d tokens)",
                        sample["clip_id"],
                        obs_ids,
                        len(_ids),
                    )

            if adv_traj_token_id is not None:
                traj_ids = (
                    [adv_traj_token_id]
                    if isinstance(adv_traj_token_id, int)
                    else list(adv_traj_token_id)
                )
                logger.debug(
                    "adv_traj injection enabled for %s (token_ids=%s)",
                    sample["clip_id"],
                    traj_ids,
                )

            # Run inference with the selected trajectory generation mode
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                if adv_traj_token_id is not None:
                    # Per-sample teacher-forced generation with adv_traj injection
                    if traj_mode == "vlm":
                        pred_xyz, pred_rot, extra = _evaluate_vlm_only_with_adv_traj(
                            model=model,
                            model_inputs=model_inputs,
                            num_traj_samples=num_traj_samples,
                            temperature=temperature,
                            top_p=top_p,
                            adv_traj_token_id=adv_traj_token_id,
                        )
                    else:
                        pred_xyz, pred_rot, extra = _evaluate_expert_with_adv_traj(
                            model=model,
                            model_inputs=model_inputs,
                            num_traj_samples=num_traj_samples,
                            temperature=temperature,
                            top_p=top_p,
                            adv_traj_token_id=adv_traj_token_id,
                        )
                elif traj_mode == "vlm":
                    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_only(
                        data=model_inputs,
                        top_p=top_p,
                        temperature=temperature,
                        num_traj_samples=num_traj_samples,
                        max_generation_length=256,
                        return_extra=True,
                    )
                else:
                    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=top_p,
                        temperature=temperature,
                        num_traj_samples=num_traj_samples,
                        max_generation_length=256,
                        return_extra=True,
                    )

            # Log raw CoC from extra (all code paths)
            try:
                cot_arr = extra.get("cot")
                if cot_arr is not None:
                    for si, txt in enumerate(np.asarray(cot_arr).flat):
                        logger.info(
                            "[%s] sample %d CoC: %s",
                            sample["clip_id"],
                            si,
                            txt if txt else "(empty)",
                        )
            except Exception:
                pass

            # Compute metrics
            min_ade = compute_minADE(pred_xyz, sample["ego_future_xyz"])
            min_fde = compute_minFDE(pred_xyz, sample["ego_future_xyz"])

            # Extract CoC — extra["cot"] may be np.array shaped (1, 1, n_samples)
            coc_text = None
            try:
                cot = extra.get("cot")
                if cot is not None:
                    flat = np.asarray(cot).flat
                    if len(flat) > 0:
                        first = str(flat[0])
                        if first:
                            coc_text = first
            except Exception:
                pass

            logger.debug(
                "[%s] generated CoC (%d chars): %s",
                sample["clip_id"],
                len(coc_text) if coc_text is not None else 0,
                coc_text,
            )

            if visualize and output_dir is not None:
                try:
                    _plot_trajectories(
                        pred_xyz=pred_xyz,
                        gt_xyz=sample["ego_future_xyz"],
                        clip_id=sample["clip_id"],
                        save_path=Path(output_dir) / "plots",
                        coc_text=coc_text,
                        min_ade=min_ade,
                        min_fde=min_fde,
                    )
                except Exception as plot_err:
                    logger.warning("Plot failed for %s: %s", sample["clip_id"], plot_err)

            # Compute reward signals
            r_traj_val, r_reason_val, r_consist_val = _compute_reward_signals(
                pred_xyz, sample["ego_future_xyz"], coc_text
            )

            # Stash trajectories for saving (pred: (S, T, 3), gt: (T, 3))
            pred_np = pred_xyz.cpu().numpy()[0, 0]  # (S, T, 3)
            gt_np = sample["ego_future_xyz"].cpu().numpy()[0, 0]  # (T, 3)

            results.append(
                {
                    "clip_id": sample["clip_id"],
                    "t0_us": t0_us,
                    "minADE": min_ade,
                    "minFDE": min_fde,
                    "r_traj": r_traj_val,
                    "r_reason": r_reason_val,
                    "r_consist": r_consist_val,
                    "pred_xyz": pred_np,
                    "gt_xyz": gt_np,
                    "success": True,
                    "error": None,
                    "coc": coc_text,
                }
            )

        except Exception as e:
            import traceback

            results.append(
                {
                    "clip_id": sample["clip_id"],
                    "t0_us": t0_us,
                    "minADE": None,
                    "minFDE": None,
                    "r_traj": None,
                    "r_reason": None,
                    "r_consist": None,
                    "pred_xyz": None,
                    "gt_xyz": None,
                    "success": False,
                    "error": f"{str(e)}\n{traceback.format_exc()}",
                    "coc": None,
                }
            )

    return results


def _stat_dict(arr):
    """Compute summary statistics for an array."""
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


METRIC_COLUMNS = ["minADE", "minFDE"]
REWARD_COLUMNS = ["r_traj", "r_reason", "r_consist"]


def merge_shards(output_dir: str | Path) -> None:
    """Merge per-shard results into unified results.csv and statistics.json.

    Combines results_shard*.csv and trajectories_shard*.npz files,
    computes aggregate statistics including reward signals.
    """
    output_dir = Path(output_dir)

    # Merge CSVs
    shard_csvs = sorted(output_dir.glob("results_shard*.csv"))
    if not shard_csvs:
        print("No shard results found!")
        return
    dfs = [pd.read_csv(p) for p in shard_csvs]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output_dir / "results.csv", index=False)

    # Merge trajectory npz files
    shard_npzs = sorted(output_dir.glob("trajectories_shard*.npz"))
    if shard_npzs:
        merged_traj = {}
        for npz_path in shard_npzs:
            data = np.load(npz_path)
            merged_traj.update(dict(data))
        np.savez_compressed(output_dir / "trajectories.npz", **merged_traj)
        print(f"Merged {len(shard_npzs)} trajectory shards "
              f"({len(merged_traj) // 2} clips)")

    # Compute statistics
    ok = df[df["success"] == True]  # noqa: E712
    stats = {
        "total_samples": len(df),
        "successful_samples": len(ok),
        "failed_samples": len(df) - len(ok),
    }
    if len(ok) > 0:
        for m in METRIC_COLUMNS:
            stats[m] = _stat_dict(ok[m].values)
        stats["rewards"] = {}
        for r in REWARD_COLUMNS:
            if r in ok.columns:
                stats["rewards"][r] = _stat_dict(ok[r].dropna().values)

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Merged {len(shard_csvs)} shards: {len(df)} total, {len(ok)} successful")
    if len(ok) > 0:
        print(f"  minADE mean: {stats['minADE']['mean']:.4f}")
        print(f"  minFDE mean: {stats['minFDE']['mean']:.4f}")
        for r in REWARD_COLUMNS:
            if r in stats.get("rewards", {}):
                print(f"  {r} mean: {stats['rewards'][r]['mean']:.4f}")


def aggregate_trials(output_dir: str | Path, num_trials: int) -> None:
    """Aggregate results across multiple trial directories.

    Each trial is expected at output_dir/trials/trial_{i}/results.csv.
    Produces per-clip averaged results and trial-level variance statistics
    including reward signals.
    """
    output_dir = Path(output_dir)
    trials_dir = output_dir / "trials"

    trial_dfs = []
    for t in range(num_trials):
        p = trials_dir / f"trial_{t}" / "results.csv"
        df = pd.read_csv(p)
        df["trial"] = t
        trial_dfs.append(df)

    all_trials = pd.concat(trial_dfs, ignore_index=True)
    all_trials.to_csv(trials_dir / "all_trials.csv", index=False)

    ok = all_trials[all_trials["success"] == True]  # noqa: E712
    value_cols = METRIC_COLUMNS + [
        r for r in REWARD_COLUMNS if r in ok.columns
    ]
    per_clip_mean = ok.groupby("clip_id")[value_cols].mean().reset_index()
    per_clip_mean["success"] = True
    per_clip_mean.to_csv(output_dir / "results.csv", index=False)

    # Trial-level means for variance estimation
    trial_means = ok.groupby("trial")[value_cols].mean()

    stats = {
        "num_trials": num_trials,
        "total_clips": int(per_clip_mean["clip_id"].nunique()),
    }
    for col in value_cols:
        vals = trial_means[col].values
        stats[col] = {
            "mean_of_trials": float(np.mean(vals)),
            "std_of_trials": float(np.std(vals)),
            "trial_values": [float(v) for v in vals],
        }

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Aggregated {num_trials} trials over {stats['total_clips']} clips")
    for col in value_cols:
        s = stats[col]
        trial_str = ", ".join(f"{v:.4f}" for v in s["trial_values"])
        print(f"  {col}: {s['mean_of_trials']:.4f} +/- "
              f"{s['std_of_trials']:.4f}  trials=[{trial_str}]")

    # Merge trajectory npz files from all trials
    merged_traj = {}
    for t in range(num_trials):
        npz_path = trials_dir / f"trial_{t}" / "trajectories.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            # Prefix with trial index to avoid key collisions
            for key in data:
                merged_traj[f"trial_{t}/{key}"] = data[key]
    if merged_traj:
        np.savez_compressed(output_dir / "trajectories.npz", **merged_traj)
        print(f"Merged trajectories from {num_trials} trials")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Alpamayo-R1 evaluation with batched data loading"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvidia/Alpamayo-R1-10B",
        help="Model name or path (full model or LoRA adapter checkpoint)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="nvidia/Alpamayo-R1-10B",
        help="Base model for LoRA checkpoints (ignored for full model paths)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Number of test samples to evaluate"
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=5,
        help="Number of trajectory samples per prediction",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.98, help="Nucleus sampling top-p")
    parser.add_argument(
        "--traj-mode",
        type=str,
        choices=["expert", "vlm"],
        default="expert",
        help=(
            "Trajectory generation mode: "
            "'expert' uses VLM + Expert + Diffusion (full pipeline), "
            "'vlm' uses VLM-only discrete trajectory tokens (no Expert/Diffusion)"
        ),
    )
    parser.add_argument(
        "--t0-us", type=int, default=5_100_000, help="Default t0 timestamp in microseconds"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-clip-ids-file",
        action="store_true",
        help="Use notebooks/clip_ids.parquet instead of full test set",
    )
    parser.add_argument(
        "--clip-ids",
        type=str,
        default=None,
        help=(
            "Explicit clip IDs to evaluate. "
            "Comma-separated UUIDs, or path to a .txt/.json/.jsonl file. "
            "Overrides --use-clip-ids-file and --num-samples."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker (default: 2)",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Use torch.compile for faster inference (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard index for multi-GPU data parallelism (0-based)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards for multi-GPU data parallelism",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default=DEFAULT_DATASET_REVISION,
        help="HF dataset revision for PhysicalAI-AV (default: pinned revision)",
    )
    parser.add_argument(
        "--iteration-dir",
        type=str,
        default=None,
        help="Path to self-play output directory containing iter_*/final/ subdirs",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Self-play iteration index to load (0-based). Merges VLM LoRA from iter_0 through iter_N.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Keep LoRA adapter unmerged (avoids bf16 underflow for small deltas).",
    )
    parser.add_argument(
        "--adv-obs",
        action="store_true",
        help="Inject positive obs advantage token.",
    )
    parser.add_argument(
        "--adv-traj",
        action="store_true",
        help="Inject positive traj advantage token.",
    )
    parser.add_argument(
        "--adv-mode",
        choices=["embedding", "text"],
        default="embedding",
        help="Advantage conditioning mode: 'embedding' uses learned AdvantageEmbedding "
        "hooks, 'text' uses plain-text tokens the VLM already understands.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save BEV trajectory plots (predicted vs. ground truth) to output_dir/plots/",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge per-shard results in --output-dir and exit (no inference).",
    )
    parser.add_argument(
        "--aggregate-trials",
        type=int,
        default=None,
        metavar="N",
        help="Aggregate N trial directories in --output-dir/trials/ and exit.",
    )

    args = parser.parse_args()

    # Handle aggregation modes (no inference needed)
    if args.merge_shards:
        merge_shards(args.output_dir)
        return
    if args.aggregate_trials is not None:
        aggregate_trials(args.output_dir, args.aggregate_trials)
        return

    if (args.shard_id is None) != (args.num_shards is None):
        parser.error("--shard-id and --num-shards must be used together")
    if (args.iteration_dir is None) != (args.iteration is None):
        parser.error("--iteration-dir and --iteration must be used together")
    if args.shard_id is not None and args.shard_id >= args.num_shards:
        parser.error("--shard-id must be less than --num-shards")

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Alpamayo-R1 Optimized Evaluation (Batched Data Loading)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Trajectory samples per prediction: {args.num_traj_samples}")
    print(f"Trajectory mode: {args.traj_mode}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Data loading workers: {args.num_workers}")
    print(f"Prefetch factor: {args.prefetch_factor}")
    print(f"Model compilation: {'enabled' if args.compile_model else 'disabled'}")
    if args.shard_id is not None:
        print(f"Shard: {args.shard_id}/{args.num_shards}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Force unbuffered stdout (Kubeflow / container environments buffer aggressively)
    import sys

    os.environ["PYTHONUNBUFFERED"] = "1"

    class FlushStreamHandler(logging.StreamHandler):
        """StreamHandler that flushes after every emit (for Kubeflow/containers)."""

        def emit(self, record):
            super().emit(record)
            self.flush()

    class FlushFileHandler(logging.FileHandler):
        """FileHandler that flushes after every emit."""

        def emit(self, record):
            super().emit(record)
            self.flush()

    # Set up our own logger (separate from transformers/root to avoid conflicts)
    log_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't let root logger (transformers) override us
    logger.handlers.clear()
    # Stdout handler (INFO+) — flush after every message
    _sh = FlushStreamHandler(sys.stdout)
    _sh.setLevel(logging.INFO)
    _sh.setFormatter(log_fmt)
    logger.addHandler(_sh)
    # File handler (DEBUG+) — flush after every message
    log_file = output_dir / "eval.log"
    _fh = FlushFileHandler(str(log_file), mode="a")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(log_fmt)
    logger.addHandler(_fh)
    # Also attach handlers to the model logger so [vlm_rollout]/[vlm_only] logs appear
    _model_logger = logging.getLogger("alpamayo_r1.models.alpamayo_r1")
    _model_logger.setLevel(logging.DEBUG)
    _model_logger.propagate = False
    _model_logger.handlers.clear()
    _model_logger.addHandler(_sh)
    _model_logger.addHandler(_fh)
    logger.info("Logging to %s", log_file)

    # Load model
    print("\nLoading model...")
    if args.iteration_dir is not None:
        from alpamayo_r1.training.selfplay_loop import load_vlm_from_iterations

        print(f"Loading self-play checkpoint: {args.iteration_dir} iter {args.iteration}")
        result = load_vlm_from_iterations(
            base_model_name=args.base_model,
            output_dir=args.iteration_dir,
            up_to_iteration=args.iteration,
            dtype=torch.bfloat16,
            device_map="auto",
            merge=not args.no_merge,
        )
        model = result["full_model"]
        print(f"Loaded self-play checkpoint through iteration {args.iteration}")
        if result.get("value_head"):
            print("  Value head: loaded")
        if result.get("adv_embedding"):
            print("  Advantage embedding: loaded")

        adv_emb_from_result = result.get("adv_embedding")
    else:
        model_path = Path(args.model_name)
        is_lora = (model_path / "adapter_config.json").exists()
        if is_lora:
            print(f"Detected LoRA adapter at {model_path}")
            model = AlpamayoR1.from_pretrained_with_lora(
                adapter_path=args.model_name,
                base_model_name=args.base_model,
                dtype=torch.bfloat16,
                device_map="auto",
                merge=not args.no_merge,
            )
        else:
            model = AlpamayoR1.from_pretrained(
                args.model_name, dtype=torch.bfloat16, device_map="auto"
            )
        adv_emb_from_result = None

    # Attach advantage conditioning if requested
    adv_obs_token_id = None
    adv_traj_token_id = None
    if args.adv_obs or args.adv_traj:
        adv_mode = args.adv_mode
        print(f"  adv_mode: {adv_mode}")

        if adv_mode == "text":
            from alpamayo_r1.training.advantage_conditioning import (
                compute_text_advantage_token_ids,
            )

            adv_token_ids = compute_text_advantage_token_ids(model.tokenizer)
        else:
            from alpamayo_r1.training.advantage_conditioning import AdvantageEmbedding

            vocab_size = model.vlm.config.text_config.vocab_size
            adv_token_ids = compute_advantage_token_ids(vocab_size)

            if adv_emb_from_result is not None:
                adv_emb = adv_emb_from_result
                adv_emb.to(model.vlm.device)
                adv_emb.attach(model.vlm)
                print("  Loaded advantage embedding from iteration checkpoint")
            else:
                model_path = Path(args.model_name) if args.model_name else None
                adv_emb_path = None
                if model_path is not None:
                    for candidate in [
                        model_path / "adv_embedding.pt",
                        model_path.parent / "adv_embedding.pt",
                    ]:
                        if candidate.exists():
                            adv_emb_path = candidate
                            break
                if adv_emb_path is not None:
                    hidden_size = model.vlm.config.text_config.hidden_size
                    adv_emb = AdvantageEmbedding(hidden_size, adv_token_ids)
                    adv_emb.load_state_dict(
                        torch.load(adv_emb_path, map_location="cpu")
                    )
                    adv_emb.to(model.vlm.device)
                    adv_emb.attach(model.vlm)
                    print(f"  Loaded advantage embedding from {adv_emb_path}")
                else:
                    print(
                        "  WARNING: No adv_embedding found, "
                        "advantage tokens will use uninitialized embeddings"
                    )

        if args.adv_obs:
            adv_obs_token_id = adv_token_ids["adv_obs_pos"]
            print(f"  adv_obs conditioning: enabled (token_id={adv_obs_token_id})")
        if args.adv_traj:
            adv_traj_token_id = adv_token_ids["adv_traj_pos"]
            print(f"  adv_traj conditioning: enabled (token_id={adv_traj_token_id})")
            
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # Optionally compile model for faster inference
    if args.compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    print("Model loaded successfully!")

    # Get clip IDs
    avdi = PhysicalAIAVDatasetInterface(revision=args.dataset_revision)
    if args.clip_ids is not None:
        # Explicit clip IDs: comma-separated or file path
        clip_ids_arg = args.clip_ids
        if os.path.isfile(clip_ids_arg):
            ext = os.path.splitext(clip_ids_arg)[1].lower()
            if ext == ".json":
                with open(clip_ids_arg) as f:
                    test_clips = json.load(f)
            elif ext == ".jsonl":
                test_clips = []
                with open(clip_ids_arg) as f:
                    for line in f:
                        entry = json.loads(line)
                        if isinstance(entry, str):
                            test_clips.append(entry)
                        elif isinstance(entry, dict) and "clip_id" in entry:
                            test_clips.append(entry["clip_id"])
            else:
                # .txt or other: one clip ID per line
                with open(clip_ids_arg) as f:
                    test_clips = [line.strip() for line in f if line.strip()]
        else:
            test_clips = [c.strip() for c in clip_ids_arg.split(",") if c.strip()]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in test_clips:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        test_clips = unique
        print(f"\nUsing {len(test_clips)} explicit clip IDs")
    elif args.use_clip_ids_file:
        print("\nUsing clip_ids.parquet file (test split only)...")
        clip_ids_df = pd.read_parquet("notebooks/clip_ids.parquet")
        all_eval_ids = set(clip_ids_df["clip_id"].tolist())
        clip_index = avdi.clip_index
        test_ids = set(clip_index[clip_index["split"] == "test"].index)
        test_clips = sorted(all_eval_ids & test_ids)
        print(f"clip_ids.parquet: {len(all_eval_ids)} total, {len(test_clips)} in test split")
    else:
        print("\nLoading test split from dataset...")
        clip_index = avdi.clip_index
        test_df = clip_index[(clip_index["split"] == "test") & clip_index["clip_is_valid"]]
        test_clips = test_df.index.tolist()
        print(f"Found {len(test_clips)} valid test clips")

    # Limit number of samples if specified (skipped when --clip-ids is used)
    if args.num_samples is not None and args.clip_ids is None:
        test_clips = test_clips[: args.num_samples]
        print(f"Limiting evaluation to {len(test_clips)} samples")

    # Shard the clip list for multi-GPU data parallelism
    if args.shard_id is not None:
        total = len(test_clips)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, total)
        test_clips = test_clips[start:end]
        print(f"Shard {args.shard_id}/{args.num_shards}: clips [{start}:{end}] ({len(test_clips)} samples)")

    print(f"\nEvaluating {len(test_clips)} test samples...")
    print("=" * 80)

    # Create dataset and dataloader.
    # Pass the resolved revision so DataLoader workers reuse the same cache
    # (avoids per-worker list_repo_refs API calls and revision mismatch).
    dataset = AlpamayoDataset(test_clips, t0_us=args.t0_us, revision=avdi.revision)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one at a time for inference, but prefetch in background
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Evaluate all samples
    all_results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", total=len(test_clips)):
            results = evaluate_batch(
                model=model,
                processor=processor,
                batch=batch,
                num_traj_samples=args.num_traj_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
                t0_us=args.t0_us,
                traj_mode=args.traj_mode,
                adv_obs_token_id=adv_obs_token_id,
                adv_traj_token_id=adv_traj_token_id,
                output_dir=str(output_dir),
                visualize=args.visualize,
            )
            all_results.extend(results)

    # Extract and save trajectories before converting to DataFrame
    traj_data = {}
    for r in all_results:
        if r.get("success") and r.get("pred_xyz") is not None:
            clip_id = r["clip_id"]
            traj_data[f"{clip_id}/pred"] = r.pop("pred_xyz")
            traj_data[f"{clip_id}/gt"] = r.pop("gt_xyz")
        else:
            r.pop("pred_xyz", None)
            r.pop("gt_xyz", None)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Compute aggregate statistics
    successful_results = results_df[results_df["success"] == True]
    failed_results = results_df[results_df["success"] == False]

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    if len(successful_results) > 0:
        min_ade_values = successful_results["minADE"].values
        min_fde_values = successful_results["minFDE"].values
        r_traj_values = successful_results["r_traj"].values
        r_reason_values = successful_results["r_reason"].values
        r_consist_values = successful_results["r_consist"].values

        stats = {
            "total_samples": len(results_df),
            "successful_samples": len(successful_results),
            "failed_samples": len(failed_results),
            "minADE": _stat_dict(min_ade_values),
            "minFDE": _stat_dict(min_fde_values),
            "rewards": {
                "trajectory": _stat_dict(r_traj_values),
                "reasoning": _stat_dict(r_reason_values),
                "consistency": _stat_dict(r_consist_values),
            },
            "config": {
                "model_name": args.model_name,
                "num_traj_samples": args.num_traj_samples,
                "traj_mode": args.traj_mode,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "t0_us": args.t0_us,
                "seed": args.seed,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "compile_model": args.compile_model,
                "adv_obs": args.adv_obs,
                "adv_traj": args.adv_traj,
            },
        }

        print(f"\nTotal samples: {stats['total_samples']}")
        print(f"Successful: {stats['successful_samples']}")
        print(f"Failed: {stats['failed_samples']}")
        print("\nminADE (meters):")
        for k in ("mean", "median", "std", "min", "max"):
            print(f"  {k.capitalize():7s}: {stats['minADE'][k]:.4f}")
        print("\nminFDE (meters):")
        for k in ("mean", "median", "std", "min", "max"):
            print(f"  {k.capitalize():7s}: {stats['minFDE'][k]:.4f}")
        print("\nReward signals:")
        reward_labels = [
            ("Trajectory", "trajectory"),
            ("Reasoning", "reasoning"),
            ("Consistency", "consistency"),
        ]
        for name, key in reward_labels:
            s = stats["rewards"][key]
            print(f"  {name:12s}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"min={s['min']:.4f}  max={s['max']:.4f}")

    else:
        stats = {
            "total_samples": len(results_df),
            "successful_samples": 0,
            "failed_samples": len(failed_results),
            "error": "All evaluations failed",
        }
        print("\n⚠️  All evaluations failed!")

    # Save results (use shard-specific filenames when sharding)
    shard_suffix = f"_shard{args.shard_id}" if args.shard_id is not None else ""
    results_csv = output_dir / f"results{shard_suffix}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nDetailed results saved to: {results_csv}")

    if traj_data:
        traj_npz = output_dir / f"trajectories{shard_suffix}.npz"
        np.savez_compressed(traj_npz, **traj_data)
        print(f"Trajectories saved to: {traj_npz} ({len(traj_data) // 2} clips, "
              f"{traj_npz.stat().st_size / 1024:.0f} KB)")

    stats_json = output_dir / f"statistics{shard_suffix}.json"
    with open(stats_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Summary statistics saved to: {stats_json}")

    # Print failure summary if any
    if len(failed_results) > 0:
        print(f"\n⚠️  {len(failed_results)} samples failed. Error summary:")
        error_counts = failed_results["error"].value_counts()
        for error, count in error_counts.head(5).items():
            print(f"  - {error[:80]}: {count} samples")

    if args.visualize:
        plots_dir = output_dir / "plots"
        n_plots = len(list(plots_dir.glob("*.png"))) if plots_dir.exists() else 0
        print(f"\nTrajectory plots saved: {n_plots} PNGs in {plots_dir}")
    print("=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
