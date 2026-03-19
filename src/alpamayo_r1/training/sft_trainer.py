"""Advantage-conditioned SFT trainer with optional action expert CFM finetuning.

This trainer subclasses HuggingFace's Trainer (NOT TRL's GRPOTrainer) to perform
standard cross-entropy training on completions labeled with advantage conditioning
tokens. It also optionally trains the action expert (flow matching model) using
deferred GPU scheduling — the expert CFM step runs *after* VLM backward frees
activations, following the pattern from feat/expert-training.

Usage:
    trainer = AdvCondSFTTrainer(
        model=full_model.vlm,
        args=training_args,
        train_dataset=adv_cond_dataset,
        processing_class=tokenizer,
        full_model=full_model,
        adv_token_ids=adv_token_ids,
        expert_cfg=expert_cfg,
    )
    trainer.train()
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo_r1.training.rollout_utils import ClipDataCache

logger = logging.getLogger(__name__)


def adv_cond_collator(features: list[dict], pad_token_id: int = 0) -> dict:
    """Collate variable-length AdvCondDataset samples by padding to max length.

    Pads input_ids with pad_token_id, labels with IGNORE_INDEX, and
    attention_mask with 0. Also collates expert CFM metadata if present.
    """
    max_len = max(len(f["input_ids"]) for f in features)

    batch = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "is_unconditional": [],
    }
    # Expert CFM metadata (non-tensor, collected as lists)
    clip_ids = []
    t0_us_list = []
    completion_prefixes = []

    for f in features:
        seq_len = len(f["input_ids"])
        pad_len = max_len - seq_len
        batch["input_ids"].append(f["input_ids"] + [pad_token_id] * pad_len)
        batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad_len)
        batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
        batch["is_unconditional"].append(f["is_unconditional"])

        if "clip_id" in f:
            clip_ids.append(f["clip_id"])
        if "t0_us" in f:
            t0_us_list.append(f["t0_us"])
        if "completion_prefix" in f:
            completion_prefixes.append(f["completion_prefix"])

    batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
    batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
    batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
    batch["is_unconditional"] = torch.tensor(batch["is_unconditional"], dtype=torch.bool)

    if clip_ids:
        batch["clip_ids"] = clip_ids
        batch["t0_us_list"] = t0_us_list
        batch["completion_prefixes"] = completion_prefixes

    return batch


class AdvCondSFTTrainer(Trainer):
    """SFT trainer with advantage conditioning and deferred expert CFM training.

    The VLM is trained via standard cross-entropy on completion tokens. The
    dataset (AdvCondDataset) handles conditioning token insertion and dual-loss
    labeling. This trainer adds:

    1. Dual-loss weighting: conditional examples get weight ``alpha``,
       unconditional examples get weight 1.0.
    2. Optional deferred expert CFM step after VLM backward.
    3. Extended checkpoint saving for expert weights.

    Args:
        model: The VLM model (full_model.vlm).
        args: HuggingFace TrainingArguments.
        train_dataset: AdvCondDataset instance.
        processing_class: Tokenizer/processor.
        full_model: Full AlpamayoR1 instance (for expert access).
        adv_token_ids: Dict mapping advantage token name -> ID.
        alpha: Weight for conditional loss (unconditional = 1.0).
        expert_cfg: Expert finetuning config dict, or None to disable.
        data_cache: ClipDataCache for loading scene data during expert step.
        value_head: SegmentValueHead instance to co-save with VLM/expert checkpoints.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args: Any,
        train_dataset: Any,
        processing_class: Any = None,
        full_model: AlpamayoR1 | None = None,
        adv_token_ids: dict[str, int] | None = None,
        alpha: float = 1.0,
        expert_cfg: dict | None = None,
        data_cache: ClipDataCache | None = None,
        value_head: torch.nn.Module | None = None,
        **kwargs,
    ):
        # Determine pad_token_id for the collator
        pad_id = 0
        if processing_class is not None:
            if hasattr(processing_class, "tokenizer"):
                pad_id = processing_class.tokenizer.pad_token_id or 0
            elif hasattr(processing_class, "pad_token_id"):
                pad_id = processing_class.pad_token_id or 0

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            data_collator=lambda features: adv_cond_collator(features, pad_token_id=pad_id),
            **kwargs,
        )
        self.full_model = full_model
        self.adv_token_ids = adv_token_ids or {}
        self.alpha = alpha
        self._data_cache = data_cache
        self._value_head = value_head
        self._expert_metrics: dict[str, list[float]] = {
            "expert_cfm/loss": [],
            "expert_cfm/valid_samples": [],
        }
        self._sft_step_count = 0
        self._uncond_total = 0
        self._uncond_count = 0

        # Expert finetuning setup
        self._expert_enabled = False
        self._expert_optimizer = None
        self._expert_cfg = expert_cfg or {}
        self._expert_every_n_steps = int(self._expert_cfg.get("every_n_steps", 1))

        if self._expert_cfg.get("enabled", False) and full_model is not None:
            self._setup_expert(full_model, self._expert_cfg)

    def _setup_expert(self, full_model: AlpamayoR1, cfg: dict) -> None:
        """Initialize expert optimizer and optionally load pretrained weights.

        Loading order depends on checkpoint type:
        - Full/base checkpoint: load weights first, then apply LoRA on top
        - LoRA-only checkpoint: apply LoRA first, then load LoRA weights
        - No checkpoint: just apply LoRA (if enabled)
        """
        load_path = cfg.get("load_path")
        lora_cfg = cfg.get("expert_lora", {})
        lora_enabled = lora_cfg.get("enabled", False)

        # Load checkpoint and detect type
        checkpoint = None
        checkpoint_is_lora = False
        if load_path:
            checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
            checkpoint_is_lora = checkpoint.get("expert_lora", False)

        # For full/base checkpoints: load weights BEFORE applying LoRA
        if checkpoint and not checkpoint_is_lora:
            full_model.expert.load_state_dict(checkpoint["expert"], strict=False)
            logger.info("Loaded full expert weights from %s", load_path)

        # Apply LoRA to expert transformer if configured
        if lora_enabled:
            from peft import LoraConfig, get_peft_model

            expert_lora_config = LoraConfig(
                r=int(lora_cfg.get("r", 4)),
                lora_alpha=int(lora_cfg.get("alpha", 32)),
                lora_dropout=float(lora_cfg.get("dropout", 0.0)),
                target_modules=list(
                    lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
                ),
                task_type="FEATURE_EXTRACTION",
            )
            full_model.expert = get_peft_model(full_model.expert, expert_lora_config)
            logger.info(
                "Applied LoRA to expert: r=%d, alpha=%d",
                expert_lora_config.r,
                expert_lora_config.lora_alpha,
            )

        # For LoRA-only checkpoints: load LoRA weights AFTER applying LoRA
        if checkpoint and checkpoint_is_lora:
            full_model.expert.load_state_dict(checkpoint["expert"], strict=False)
            logger.info("Loaded expert LoRA weights from %s", load_path)

        # Load projection layers (always full state_dict)
        if checkpoint:
            full_model.action_in_proj.load_state_dict(checkpoint["action_in_proj"])
            full_model.action_out_proj.load_state_dict(checkpoint["action_out_proj"])
            logger.info("Loaded projection layers from %s", load_path)

        # Collect trainable parameters: expert (LoRA or full) + projections
        expert_params = []
        for name, param in full_model.expert.named_parameters():
            if param.requires_grad:
                expert_params.append(param)
        for module in (full_model.action_in_proj, full_model.action_out_proj):
            for param in module.parameters():
                param.requires_grad = True
                expert_params.append(param)

        if not expert_params:
            logger.warning("No expert parameters found — disabling expert finetuning")
            return

        self._expert_optimizer = torch.optim.AdamW(
            expert_params,
            lr=float(cfg.get("lr", 1e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
        )

        self._expert_enabled = True
        logger.info(
            "Expert CFM finetuning enabled: %d trainable params, lr=%s, every_n_steps=%d",
            sum(p.numel() for p in expert_params),
            cfg.get("lr", 1e-4),
            self._expert_every_n_steps,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Standard cross-entropy with dual-loss weighting.

        The dataset has already masked prompt+conditioning positions in labels
        with IGNORE_INDEX. We just apply the alpha weight for conditional examples.
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss

        # Apply alpha weighting for conditional examples
        is_unconditional = inputs.get("is_unconditional")
        if is_unconditional is not None and self.alpha != 1.0:
            # is_unconditional is a bool tensor (batch,)
            conditional_mask = ~is_unconditional
            if conditional_mask.any():
                # Weight the loss: unconditional=1.0, conditional=alpha
                # Since HF Trainer averages loss across the batch, we need to
                # scale per-sample. As a simpler approximation, scale the whole
                # batch loss by the effective weight.
                n_cond = conditional_mask.sum().item()
                n_uncond = is_unconditional.sum().item()
                n_total = n_cond + n_uncond
                if n_total > 0:
                    effective_weight = (self.alpha * n_cond + 1.0 * n_uncond) / n_total
                    loss = loss * effective_weight

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to add deferred expert CFM step after VLM backward."""
        # 1. Normal SFT forward + backward
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Track unconditional fraction
        is_unconditional = inputs.get("is_unconditional")
        if is_unconditional is not None:
            self._uncond_total += is_unconditional.numel()
            self._uncond_count += is_unconditional.sum().item()

        # Log SFT loss periodically
        self._sft_step_count += 1
        if self._sft_step_count % max(1, self.args.logging_steps) == 0:
            uncond_frac = self._uncond_count / max(self._uncond_total, 1)
            logger.info(
                "SFT step %d: loss=%.4f lr=%.2e uncond_frac=%.3f (%d/%d)",
                self._sft_step_count,
                loss.item() if hasattr(loss, "item") else loss,
                self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0,
                uncond_frac,
                self._uncond_count,
                self._uncond_total,
            )

        # 2. Deferred expert CFM step
        if self._expert_enabled and self.state.global_step % self._expert_every_n_steps == 0:
            # Free VLM batch tensors from GPU to make room for expert
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    inputs[k] = v.cpu()
            gc.collect()
            torch.cuda.empty_cache()

            self._expert_cfm_step(inputs)

        return loss

    def _expert_cfm_step(self, inputs: dict) -> None:
        """Run one expert CFM training step using deferred GPU scheduling.

        Two-phase approach to avoid VLM + expert co-residency on GPU:
        Phase A: Extract KV caches with VLM on GPU, then offload VLM to CPU.
        Phase B: Move expert to GPU, compute CFM losses using cached KV,
                 run optimizer step, then move expert back to CPU and
                 restore VLM to GPU.
        """
        if self.full_model is None or self._data_cache is None:
            return

        from alpamayo_r1.training.train_expert import compute_cfm_loss

        cfg = self._expert_cfg
        num_noisy_samples = int(cfg.get("num_noisy_samples", 4))
        max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        device = self.accelerator.device

        # Get clip metadata from inputs (stashed by the dataset)
        clip_ids = inputs.get("clip_ids", [])
        t0_us_list = inputs.get("t0_us_list", [])
        completion_prefixes = inputs.get("completion_prefixes", [])

        if not clip_ids:
            return

        # ---- Phase A: Extract KV caches (VLM on GPU) ----
        cached_samples = []
        for i in range(len(clip_ids)):
            try:
                clip_id = clip_ids[i]
                t0_us = t0_us_list[i]
                completion_prefix = completion_prefixes[i]

                # Load scene data
                model_inputs, ego_future_xyz = self._data_cache.get(clip_id, t0_us, device)
                ego_future_rot = self._data_cache.get_ego_future_rot(clip_id, t0_us).to(device)

                # Teacher-forced VLM forward → KV cache
                kv_result = self._get_vlm_kv_cache_teacher_forced(
                    model_inputs, completion_prefix, device
                )

                # Prepare GT trajectory data
                hist_xyz = model_inputs["ego_history_xyz"][:, -1].to(device)
                hist_rot = model_inputs["ego_history_rot"][:, -1].to(device)
                fut_xyz = ego_future_xyz.to(device)[:, -1]
                fut_rot = ego_future_rot[:, -1]

                cached_samples.append(
                    {
                        "kv_result": kv_result,
                        "hist_xyz": hist_xyz,
                        "hist_rot": hist_rot,
                        "fut_xyz": fut_xyz,
                        "fut_rot": fut_rot,
                    }
                )
            except Exception as e:
                logger.warning("KV cache extraction failed for sample %d: %s", i, e)
                continue

        if not cached_samples:
            return

        # ---- Offload VLM to CPU, free GPU for expert ----
        # self.full_model.vlm.cpu()
        # gc.collect()
        # torch.cuda.empty_cache()

        # ---- Phase B: Expert CFM (expert on GPU, VLM offloaded) ----
        self.full_model.expert.to(device)
        self.full_model.action_in_proj.to(device)
        self.full_model.action_out_proj.to(device)
        self.full_model.action_space.to(device)

        total_loss = 0.0
        n_valid = 0
        self._expert_optimizer.zero_grad()

        for i, sample in enumerate(cached_samples):
            try:
                prompt_cache, rope_deltas, prefill_seq_len, b_star, offset = sample["kv_result"]

                with torch.no_grad():
                    gt_action = self.full_model.action_space.traj_to_action(
                        sample["hist_xyz"],
                        sample["hist_rot"],
                        sample["fut_xyz"],
                        sample["fut_rot"],
                    )

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
                    loss = loss / max(len(cached_samples), 1)

                loss.backward()
                total_loss += loss.item()
                n_valid += 1

            except Exception as e:
                logger.warning("Expert CFM step failed for sample %d: %s", i, e)
                continue

        if n_valid > 0:
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

            self._expert_metrics["expert_cfm/loss"].append(total_loss)
            self._expert_metrics["expert_cfm/valid_samples"].append(float(n_valid))
            logger.debug(
                "expert CFM step | loss=%.6f valid=%d/%d",
                total_loss,
                n_valid,
                len(cached_samples),
            )

        # Move expert back to CPU, restore VLM to GPU for next training step
        # self.full_model.expert.cpu()
        # self.full_model.action_in_proj.cpu()
        # self.full_model.action_out_proj.cpu()
        # self.full_model.action_space.cpu()
        # gc.collect()
        # torch.cuda.empty_cache()
        # self.full_model.vlm.to(device)

    def _get_vlm_kv_cache_teacher_forced(
        self,
        model_inputs: dict,
        completion_prefix: list[int],
        device: torch.device,
    ) -> tuple:
        """Teacher-forced VLM forward to get KV cache for expert conditioning.

        Same pattern as AlpamayoGRPOTrainer._get_vlm_kv_cache_teacher_forced().
        """
        tokenized = {k: v for k, v in model_inputs["tokenized_data"].items()}
        input_ids = tokenized.pop("input_ids").to(device)

        # Fuse history trajectory tokens
        traj_data = {
            "ego_history_xyz": model_inputs["ego_history_xyz"].to(device),
            "ego_history_rot": model_inputs["ego_history_rot"].to(device),
        }
        input_ids = self.full_model.fuse_traj_tokens(input_ids, traj_data)

        # Concatenate prompt + completion prefix
        prefix_tensor = torch.tensor(completion_prefix, dtype=torch.long, device=device).unsqueeze(
            0
        )
        full_ids = torch.cat([input_ids, prefix_tensor], dim=1)

        # Attention mask
        for k, v in tokenized.items():
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.to(device)

        if "attention_mask" in tokenized:
            orig_mask = tokenized["attention_mask"]
            prefix_mask = torch.ones(
                1, len(completion_prefix), device=device, dtype=orig_mask.dtype
            )
            tokenized["attention_mask"] = torch.cat([orig_mask, prefix_mask], dim=1)

        # Disable gradient checkpointing for use_cache=True
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
        offset = torch.tensor([full_ids.shape[1]], device=device)

        return past_key_values, rope_deltas, prefill_seq_len, b_star, offset

    def _save(self, output_dir: str, state_dict=None) -> None:
        """Extend default save to also persist expert and value head weights.

        If the expert is a PeftModel (LoRA enabled), saves only LoRA weights
        and sets ``expert_lora: True`` in the checkpoint. Otherwise saves the
        full expert state_dict.  Projection layers are always saved in full.
        """
        super()._save(output_dir, state_dict)

        if self.full_model is not None:
            expert = self.full_model.expert
            is_lora = hasattr(expert, "peft_config")

            if is_lora:
                # Use PEFT's save_pretrained — saves adapter_config.json +
                # adapter weights so load can use PeftModel.from_pretrained.
                adapter_dir = Path(output_dir) / "expert_adapter"
                expert.save_pretrained(str(adapter_dir))
                logger.info("Saved expert LoRA adapter to %s", adapter_dir)

            # expert_checkpoint.pt: full expert weights (non-LoRA) or just
            # projections (LoRA — adapter handled by save_pretrained above).
            expert_state = {
                "action_in_proj": self.full_model.action_in_proj.state_dict(),
                "action_out_proj": self.full_model.action_out_proj.state_dict(),
            }
            if not is_lora:
                expert_state["expert"] = expert.state_dict()
                expert_state["expert_lora"] = False
            if self._expert_optimizer is not None:
                expert_state["optimizer"] = self._expert_optimizer.state_dict()

            expert_path = Path(output_dir) / "expert_checkpoint.pt"
            torch.save(expert_state, expert_path)
            logger.info("Saved expert checkpoint to %s (lora=%s)", expert_path, is_lora)

        if self._value_head is not None:
            vh_path = Path(output_dir) / "value_head.pt"
            torch.save(self._value_head.state_dict(), vh_path)
            logger.info("Saved value head to %s", vh_path)

        adv_embed = getattr(self.model, "adv_embedding", None)
        if adv_embed is not None:
            adv_embed_path = Path(output_dir) / "adv_embedding.pt"
            torch.save(adv_embed.state_dict(), adv_embed_path)
            logger.info("Saved advantage embedding to %s", adv_embed_path)
