

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
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1, ExpertLogitsProcessor
from alpamayo_r1.models.token_utils import StopAfterEOS, extract_traj_tokens
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
    from transformers import GenerationConfig, StoppingCriteriaList
    from transformers.generation.logits_process import LogitsProcessorList

    from alpamayo_r1.training.sft_rollout import run_batched_expert_diffusion

    device = next(model.vlm.parameters()).device

    # 1. Fuse history trajectory tokens
    tokenized = {k: v for k, v in model_inputs["tokenized_data"].items()}
    input_ids = tokenized.pop("input_ids")
    traj_data = {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data)
    prompt_len = input_ids.shape[1]

    hist_xyz = model_inputs["ego_history_xyz"][:, -1]  # (1, T, 3)
    hist_rot = model_inputs["ego_history_rot"][:, -1]  # (1, T, 3, 3)

    # 2. Batch AR CoC generation
    traj_future_start_id = model.special_token_ids["traj_future_start"]
    pad_token_id = model.tokenizer.pad_token_id

    gen_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_traj_samples,
        max_new_tokens=256 + 10,
        pad_token_id=pad_token_id,
    )
    logits_processor = LogitsProcessorList(
        [
            ExpertLogitsProcessor(
                traj_token_offset=model.config.traj_token_start_idx,
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )
    stopping = StoppingCriteriaList([StopAfterEOS(eos_token_id=traj_future_start_id)])

    vlm_output = model.vlm.generate(
        input_ids=input_ids,
        generation_config=gen_config,
        stopping_criteria=stopping,
        logits_processor=logits_processor,
        **tokenized,
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
    pv = tokenized.get("pixel_values")
    igt = tokenized.get("image_grid_thw")
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

    device = next(model.vlm.parameters()).device

    # 1. Fuse history trajectory tokens
    tokenized = {k: v for k, v in model_inputs["tokenized_data"].items()}
    input_ids = tokenized.pop("input_ids")
    traj_data = {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data)
    prompt_len = input_ids.shape[1]

    hist_xyz = model_inputs["ego_history_xyz"][:, -1]  # (1, T, 3)
    hist_rot = model_inputs["ego_history_rot"][:, -1]  # (1, T, 3, 3)

    # 2. Batch AR CoC generation (stop at traj_future_start)
    traj_future_start_id = model.special_token_ids["traj_future_start"]
    traj_future_end_id = model.special_token_ids["traj_future_end"]
    tokens_per_future_traj = model.config.tokens_per_future_traj
    pad_token_id = model.tokenizer.pad_token_id

    gen_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_traj_samples,
        max_new_tokens=256 + 10,
        pad_token_id=pad_token_id,
    )
    stopping = StoppingCriteriaList([StopAfterEOS(eos_token_id=traj_future_start_id)])

    vlm_output = model.vlm.generate(
        input_ids=input_ids,
        generation_config=gen_config,
        stopping_criteria=stopping,
        **tokenized,
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
            completion_prefix_ids = coc_tokens + [adv_traj_token_id]
            prefix_tensor = torch.tensor(
                [completion_prefix_ids], device=device, dtype=torch.long
            )
            full_ids = torch.cat([input_ids, prefix_tensor], dim=1)

            tf_kwargs = {}
            if "attention_mask" in tokenized:
                orig_mask = tokenized["attention_mask"]
                prefix_mask = torch.ones(
                    1, len(completion_prefix_ids), device=device, dtype=orig_mask.dtype
                )
                tf_kwargs["attention_mask"] = torch.cat([orig_mask, prefix_mask], dim=1)
            for k in ("pixel_values", "image_grid_thw"):
                if k in tokenized:
                    tf_kwargs[k] = tokenized[k]

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
) -> list:
    """Evaluate a batch of samples.

    Args:
        traj_mode: Trajectory generation mode.
            ``"expert"`` uses the full VLM + Expert + Diffusion pipeline.
            ``"vlm"`` uses VLM-only discrete trajectory token generation.
        adv_obs_token_id: If set, append this token to input_ids before generation.
        adv_traj_token_id: If set, inject this token between CoC and trajectory
            via teacher-forced KV cache rebuild (per-sample processing).
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
                "[%s] decoded input (%d tokens):\n%s",
                sample["clip_id"],
                len(_ids),
                model.tokenizer.decode(_ids, skip_special_tokens=False),
            )
            if adv_obs_token_id is not None:
                obs_ids = (
                    [adv_obs_token_id]
                    if isinstance(adv_obs_token_id, int)
                    else list(adv_obs_token_id)
                )
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
                # adv_traj is injected later (teacher-forced), not in input_ids
                # Just log that it will be injected
                logger.debug(
                    "adv_traj injection enabled for %s (token_ids=%s)",
                    sample["clip_id"],
                    traj_ids,
                )

            # Run inference with the selected trajectory generation mode
            with torch.autocast(device, dtype=torch.bfloat16):
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

            results.append(
                {
                    "clip_id": sample["clip_id"],
                    "t0_us": t0_us,
                    "minADE": min_ade,
                    "minFDE": min_fde,
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
                    "success": False,
                    "error": f"{str(e)}\n{traceback.format_exc()}",
                    "coc": None,
                }
            )

    return results


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
        help="Inject positive obs advantage token. Requires --iteration-dir.",
    )
    parser.add_argument(
        "--adv-traj",
        action="store_true",
        help="Inject positive traj advantage token. Requires --iteration-dir.",
    )
    parser.add_argument(
        "--adv-mode",
        type=str,
        default="embedding",
        choices=["embedding", "text"],
        help="Advantage conditioning mode: 'embedding' (learned hooks) or 'text' (plain text).",
    )

    args = parser.parse_args()

    if (args.shard_id is None) != (args.num_shards is None):
        parser.error("--shard-id and --num-shards must be used together")
    if (args.iteration_dir is None) != (args.iteration is None):
        parser.error("--iteration-dir and --iteration must be used together")
    if args.shard_id is not None and args.shard_id >= args.num_shards:
        parser.error("--shard-id must be less than --num-shards")
    if (args.adv_obs or args.adv_traj) and args.iteration_dir is None:
        parser.error("--adv-obs/--adv-traj require --iteration-dir")

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

        # Attach advantage embedding if conditioning is requested
        adv_obs_token_id = None
        adv_traj_token_id = None
        if args.adv_obs or args.adv_traj:
            adv_mode = getattr(args, "adv_mode", "embedding")
            if adv_mode == "text":
                # from alpamayo_r1.training.advantage_conditioning import (
                #     compute_text_advantage_token_ids,
                # )
                # adv_token_ids = compute_text_advantage_token_ids(model.tokenizer)
                # print("  adv_mode: text (plain-text indicators, no hooks)")
                pass
            else:
                adv_emb = result.get("adv_embedding")
                if adv_emb is None:
                    parser.error("No adv_embedding checkpoint found in iteration dir")
                vocab_size = model.vlm.config.text_config.vocab_size
                adv_token_ids = compute_advantage_token_ids(vocab_size)
                adv_emb.to(model.vlm.device)
                adv_emb.attach(model.vlm)
                print("  adv_mode: embedding (AdvantageEmbedding hooks)")
            if args.adv_obs:
                adv_obs_token_id = adv_token_ids["adv_obs_pos"]
                print(f"  adv_obs conditioning: enabled (token_id={adv_obs_token_id})")
            if args.adv_traj:
                adv_traj_token_id = adv_token_ids["adv_traj_pos"]
                print(f"  adv_traj conditioning: enabled (token_id={adv_traj_token_id})")
    else:
        adv_obs_token_id = None
        adv_traj_token_id = None
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
            )
            all_results.extend(results)

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

        stats = {
            "total_samples": len(results_df),
            "successful_samples": len(successful_results),
            "failed_samples": len(failed_results),
            "minADE": {
                "mean": float(np.mean(min_ade_values)),
                "median": float(np.median(min_ade_values)),
                "std": float(np.std(min_ade_values)),
                "min": float(np.min(min_ade_values)),
                "max": float(np.max(min_ade_values)),
            },
            "minFDE": {
                "mean": float(np.mean(min_fde_values)),
                "median": float(np.median(min_fde_values)),
                "std": float(np.std(min_fde_values)),
                "min": float(np.min(min_fde_values)),
                "max": float(np.max(min_fde_values)),
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
        print(f"\nminADE (meters):")
        print(f"  Mean:   {stats['minADE']['mean']:.4f}")
        print(f"  Median: {stats['minADE']['median']:.4f}")
        print(f"  Std:    {stats['minADE']['std']:.4f}")
        print(f"  Min:    {stats['minADE']['min']:.4f}")
        print(f"  Max:    {stats['minADE']['max']:.4f}")
        print(f"\nminFDE (meters):")
        print(f"  Mean:   {stats['minFDE']['mean']:.4f}")
        print(f"  Median: {stats['minFDE']['median']:.4f}")
        print(f"  Std:    {stats['minFDE']['std']:.4f}")
        print(f"  Min:    {stats['minFDE']['min']:.4f}")
        print(f"  Max:    {stats['minFDE']['max']:.4f}")

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

    print("=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
