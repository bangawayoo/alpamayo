"""Flow Matching Action Expert training entry point.

Usage:
    python -m alpamayo_r1.training.train_expert --config-name expert_default

This script trains the Expert Transformer, ActionInProj, and ActionOutProj
using a Conditional Flow Matching (CFM) objective. The VLM is frozen and
used only to produce KV-cache conditioning for the expert.

Training pipeline:
1. Load AlpamayoR1 model (freeze VLM, unfreeze expert + projections)
2. For each scene:
   a. Run frozen VLM forward pass to get KV cache (scene conditioning)
   b. Convert ground-truth trajectory to action space
   c. Sample noise x_0 ~ N(0, I) and timestep t ~ U(0, 1)
   d. Interpolate x_t = (1-t)*x_0 + t*x_1
   e. Expert predicts vector field v_pred = step_fn(x_t, t)
   f. Loss = MSE(v_pred, x_1 - x_0)
3. Backpropagate through expert + projections only
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from pathlib import Path

import einops
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from physical_ai_av import PhysicalAIAVDatasetInterface
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import StoppingCriteriaList

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.token_utils import StopAfterEOS
from alpamayo_r1.training.dataset import build_alpamayo_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ExpertTrainingDataset(Dataset):
    """Lazy-loading dataset that yields (model_inputs, gt_future_xyz/rot) pairs.

    Clips are loaded and processed on first access, then cached in CPU RAM
    (LRU eviction when cache exceeds max_size).
    """

    def __init__(
        self,
        clip_ids: list[str],
        t0_us: int,
        avdi: PhysicalAIAVDatasetInterface,
        processor,
        max_cache_size: int = 200,
    ):
        self.clip_ids = clip_ids
        self.t0_us = t0_us
        self._avdi = avdi
        self._processor = processor
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_cache_size = max_cache_size

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> dict:
        clip_id = self.clip_ids[idx]
        if clip_id not in self._cache:
            data = load_physical_aiavdataset(
                clip_id=clip_id, t0_us=self.t0_us, avdi=self._avdi, maybe_stream=True
            )
            model_inputs = helper.prepare_model_inputs(data, self._processor, device="cpu")
            entry = {
                "model_inputs": model_inputs,
                "ego_future_xyz": data["ego_future_xyz"],
                "ego_future_rot": data["ego_future_rot"],
            }
            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)
            self._cache[clip_id] = entry
        else:
            self._cache.move_to_end(clip_id)

        return self._cache[clip_id]


def expert_collate_fn(batch: list[dict]) -> dict:
    """Identity collate — returns list of dicts (variable-size tensors)."""
    return batch


# ---------------------------------------------------------------------------
# VLM KV-cache extraction (frozen forward pass)
# ---------------------------------------------------------------------------


@torch.no_grad()
def get_vlm_kv_cache(
    model: AlpamayoR1,
    model_inputs: dict,
    device: torch.device,
) -> tuple:
    """Run frozen VLM forward pass and return KV cache + metadata.

    The VLM processes the full prompt (images + history trajectory + CoC
    generation up to <|traj_future_start|>), and we capture the KV cache
    that the expert transformer will attend to.

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
    input_ids = model.fuse_traj_tokens(input_ids, traj_data)

    # Move remaining tokenized data to device
    for k, v in tokenized.items():
        if isinstance(v, torch.Tensor):
            tokenized[k] = v.to(device)

    # Generate CoC text up to <|traj_future_start|>
    eos_token_id = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])

    from transformers.generation.logits_process import LogitsProcessorList
    from alpamayo_r1.models.alpamayo_r1 import ExpertLogitsProcessor

    logits_processor = LogitsProcessorList(
        [
            ExpertLogitsProcessor(
                traj_token_offset=model.config.traj_token_start_idx,
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )

    gen_config = model.vlm.generation_config
    gen_config.do_sample = True
    gen_config.temperature = 0.6
    gen_config.top_p = 0.98
    gen_config.num_return_sequences = 1
    gen_config.max_new_tokens = 300
    gen_config.pad_token_id = model.tokenizer.pad_token_id
    gen_config.output_logits = True
    gen_config.return_dict_in_generate = True

    with torch.autocast(str(device), dtype=torch.bfloat16):
        vlm_outputs = model.vlm.generate(
            input_ids=input_ids,
            generation_config=gen_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized,
        )

    vlm_outputs.rope_deltas = model.vlm.model.rope_deltas
    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()

    # Find <traj_future_start> position
    b_star = vlm_outputs.sequences.shape[0]
    traj_future_start_mask = vlm_outputs.sequences == eos_token_id
    has_traj_future_start = traj_future_start_mask.any(dim=1)
    traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
    last_token_positions = torch.full((b_star,), vlm_outputs.sequences.shape[1] - 1, device=device)
    valid_token_pos_id = torch.where(
        has_traj_future_start, traj_future_start_positions, last_token_positions
    )
    offset = valid_token_pos_id + 1

    return prompt_cache, vlm_outputs.rope_deltas, prefill_seq_len, b_star, offset


# ---------------------------------------------------------------------------
# CFM loss computation
# ---------------------------------------------------------------------------


def compute_cfm_loss(
    model: AlpamayoR1,
    gt_action: torch.Tensor,
    prompt_cache,
    rope_deltas: torch.Tensor,
    prefill_seq_len: int,
    b_star: int,
    offset: torch.Tensor,
    num_noisy_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute Conditional Flow Matching loss for a single scene.

    Args:
        model: AlpamayoR1 model (expert + projections are trainable).
        gt_action: Ground-truth action, shape (1, T_future, C_action).
        prompt_cache: VLM KV cache from frozen forward pass.
        rope_deltas: Rotary position embedding deltas.
        prefill_seq_len: Length of the prefill sequence in the cache.
        b_star: Batch size (1 for single scene).
        offset: Token position offset for the expert.
        num_noisy_samples: Number of noise samples per GT action.
        device: Compute device.

    Returns:
        Scalar CFM loss (MSE between predicted and target vector fields).
    """
    action_dims = model.action_space.get_action_space_dims()
    n_diffusion_tokens = action_dims[0]

    # Expand GT action for multiple noise samples: (num_noisy_samples, T, C)
    x_1 = gt_action.expand(num_noisy_samples, -1, -1)

    # Sample noise and timesteps
    x_0 = torch.randn_like(x_1)
    t = torch.rand(num_noisy_samples, 1, 1, device=device)

    # OT-CFM interpolation: x_t = (1-t)*x_0 + t*x_1
    x_t = (1.0 - t) * x_0 + t * x_1

    # Target vector field: v = x_1 - x_0
    v_target = x_1 - x_0

    # Build position IDs for the expert (same logic as inference)
    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=num_noisy_samples).clone()
    # Expand offset and rope_deltas for num_noisy_samples
    offset_expanded = offset.expand(num_noisy_samples)
    rope_deltas_expanded = rope_deltas.expand(num_noisy_samples, -1)
    delta = rope_deltas_expanded + offset_expanded[:, None]
    position_ids += delta.to(position_ids.device)

    # Build attention mask
    attention_mask = torch.zeros(
        (num_noisy_samples, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
        dtype=torch.float32,
        device=device,
    )
    for i in range(num_noisy_samples):
        attention_mask[i, :, :, offset_expanded[i] : -n_diffusion_tokens] = torch.finfo(
            attention_mask.dtype
        ).min

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    # Expand KV cache from batch=1 to batch=num_noisy_samples so the expert
    # attention can concatenate cached and new key/value states along the
    # sequence dimension without a batch-size mismatch.
    cache_batch = prompt_cache.get_seq_length()  # just to verify it's initialized
    if cache_batch >= 0 and num_noisy_samples > 1:
        for layer in prompt_cache.layers:
            if hasattr(layer, "keys") and layer.keys.numel() > 0 and layer.keys.shape[0] < num_noisy_samples:
                layer.keys = layer.keys.expand(num_noisy_samples, -1, -1, -1)
                layer.values = layer.values.expand(num_noisy_samples, -1, -1, -1)

    # Forward pass through expert
    # Pass t as-is (num_noisy_samples, 1, 1) — action_in_proj expects ≥3D timesteps
    # so that timesteps[..., -1] remains 2D and Fourier output is 3D for repeat/cat.
    future_token_embeds = model.action_in_proj(x_t, t)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(num_noisy_samples, n_diffusion_tokens, -1)

    expert_out = model.expert(
        inputs_embeds=future_token_embeds,
        position_ids=position_ids,
        past_key_values=prompt_cache,
        attention_mask=attention_mask,
        use_cache=True,
        **forward_kwargs,
    )
    # Crop the KV cache back to prefill length (expert added new tokens)
    prompt_cache.crop(prefill_seq_len)

    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
    v_pred = model.action_out_proj(last_hidden).view(num_noisy_samples, *action_dims)

    loss = F.mse_loss(v_pred, v_target)
    return loss


# ---------------------------------------------------------------------------
# Evaluation (sampling + ADE)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_expert(
    model: AlpamayoR1,
    eval_dataset: ExpertTrainingDataset,
    device: torch.device,
    num_diffusion_steps: int = 10,
    max_eval_samples: int = 50,
) -> dict[str, float]:
    """Evaluate expert by sampling trajectories and computing ADE.

    Runs the full inference pipeline (VLM → Expert → Diffusion → action_to_traj)
    and compares against ground truth.
    """
    model.eval()
    total_ade = 0.0
    total_samples = 0
    n_eval = min(len(eval_dataset), max_eval_samples)

    for idx in range(n_eval):
        sample = eval_dataset[idx]
        mi = helper.to_device(sample["model_inputs"], device=device)
        gt_future_xyz = sample["ego_future_xyz"].to(device)  # (1, 1, T, 3)

        try:
            # Get VLM KV cache
            prompt_cache, rope_deltas, prefill_seq_len, b_star, offset = get_vlm_kv_cache(
                model, mi, device
            )

            action_dims = model.action_space.get_action_space_dims()
            n_diffusion_tokens = action_dims[0]

            # Build position IDs and attention mask (same as training)
            position_ids = torch.arange(n_diffusion_tokens, device=device)
            position_ids = einops.repeat(position_ids, "l -> 3 b l", b=1).clone()
            delta = rope_deltas + offset[:, None]
            position_ids += delta.to(position_ids.device)

            attention_mask = torch.zeros(
                (1, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
                dtype=torch.float32,
                device=device,
            )
            for i in range(1):
                attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                    attention_mask.dtype
                ).min

            forward_kwargs = {}
            if model.config.expert_non_causal_attention:
                forward_kwargs["is_causal"] = False

            # Define step_fn for diffusion sampling
            def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                future_embeds = model.action_in_proj(x, t)
                if future_embeds.dim() == 2:
                    future_embeds = future_embeds.view(1, n_diffusion_tokens, -1)
                out = model.expert(
                    inputs_embeds=future_embeds,
                    position_ids=position_ids,
                    past_key_values=prompt_cache,
                    attention_mask=attention_mask,
                    use_cache=True,
                    **forward_kwargs,
                )
                prompt_cache.crop(prefill_seq_len)
                hidden = out.last_hidden_state[:, -n_diffusion_tokens:]
                return model.action_out_proj(hidden).view(-1, *action_dims)

            # Sample action via diffusion
            sampled_action = model.diffusion.sample(
                batch_size=1,
                step_fn=step_fn,
                device=device,
                return_all_steps=False,
                inference_step=num_diffusion_steps,
            )

            # Convert action to trajectory
            hist_xyz = mi["ego_history_xyz"][:, -1]
            hist_rot = mi["ego_history_rot"][:, -1]
            pred_xyz, _ = model.action_space.action_to_traj(sampled_action, hist_xyz, hist_rot)

            # ADE (xy plane)
            gt_xy = gt_future_xyz[0, 0, :, :2]
            pred_xy = pred_xyz[0, :, :2]
            min_len = min(gt_xy.shape[0], pred_xy.shape[0])
            ade = (gt_xy[:min_len] - pred_xy[:min_len]).norm(dim=-1).mean().item()

            total_ade += ade
            total_samples += 1
        except Exception as e:
            logger.warning("Eval failed for sample %d: %s", idx, e)
            continue

    model.train()

    if total_samples == 0:
        return {"eval/ade": float("inf"), "eval/num_samples": 0}

    return {
        "eval/ade": total_ade / total_samples,
        "eval/num_samples": total_samples,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


@hydra.main(config_path="configs", config_name="expert_default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main expert training function."""
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = output_dir / "resolved_config.yaml"
    config_save_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # Seeds
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # ---------------------------------------------------------------
    # 1. Load model
    # ---------------------------------------------------------------
    model_name = cfg.get("model_name", "nvidia/Alpamayo-R1-10B")
    base_model_name = cfg.get("base_model_name", None)

    adapter_config_path = Path(model_name) / "adapter_config.json"
    if adapter_config_path.exists() and base_model_name:
        logger.info("Loading base model %s with LoRA adapter %s", base_model_name, model_name)
        model = AlpamayoR1.from_pretrained_with_lora(
            adapter_path=model_name,
            base_model_name=base_model_name,
            dtype=torch.bfloat16,
            device_map=None,
            merge=True,
        )
    else:
        logger.info("Loading model: %s", model_name)
        model = AlpamayoR1.from_pretrained(model_name, dtype=torch.bfloat16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------
    # 2. Freeze everything, then selectively unfreeze expert + projections
    # ---------------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to expert if configured
    expert_lora_cfg = cfg.get("expert_lora", {})
    use_expert_lora = bool(expert_lora_cfg.get("enabled", False))

    if use_expert_lora:
        from peft import LoraConfig as PeftLoraConfig
        from peft import inject_adapter_in_model

        expert_lora_config = PeftLoraConfig(
            r=int(expert_lora_cfg.get("r", 16)),
            lora_alpha=int(expert_lora_cfg.get("alpha", 32)),
            lora_dropout=float(expert_lora_cfg.get("dropout", 0.05)),
            target_modules=list(
                expert_lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
            ),
        )
        inject_adapter_in_model(model.expert, expert_lora_config, adapter_name="expert_lora")
        # Only LoRA params are trainable in the expert
        for name, param in model.expert.named_parameters():
            param.requires_grad = "lora_" in name
        logger.info(
            "Expert LoRA enabled: r=%d, alpha=%d, targets=%s",
            expert_lora_config.r,
            expert_lora_config.lora_alpha,
            expert_lora_config.target_modules,
        )
    else:
        # Full-parameter expert training
        for param in model.expert.parameters():
            param.requires_grad = True

    # action_in_proj and action_out_proj are always fully trainable (small layers)
    for param in model.action_in_proj.parameters():
        param.requires_grad = True
    for param in model.action_out_proj.parameters():
        param.requires_grad = True

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Parameters: %d total, %d trainable (%.1f%%)",
        total_params,
        total_trainable_params,
        100.0 * total_trainable_params / total_params,
    )

    model.to(device)
    model.train()
    # Keep VLM in eval mode (frozen)
    model.vlm.eval()

    # ---------------------------------------------------------------
    # 3. Dataset and dataloader
    # ---------------------------------------------------------------
    data_cfg = cfg.get("data", {})
    dataset_revision = data_cfg.get("dataset_revision", None)
    avdi = PhysicalAIAVDatasetInterface(revision=dataset_revision)
    processor = helper.get_processor(model.tokenizer)

    hf_dataset = build_alpamayo_dataset(
        split=data_cfg.get("split", "train"),
        t0_us=data_cfg.get("t0_us", 5_100_000),
        max_samples=data_cfg.get("max_samples", None),
        clip_ids_file=data_cfg.get("clip_ids_file", None),
        exclude_clip_ids_file=data_cfg.get("exclude_clip_ids_file", None),
        avdi=avdi,
    )
    clip_ids = [row["clip_id"] for row in hf_dataset]

    train_dataset = ExpertTrainingDataset(
        clip_ids=clip_ids,
        t0_us=data_cfg.get("t0_us", 5_100_000),
        avdi=avdi,
        processor=processor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # We process one scene at a time (variable-size inputs)
        shuffle=True,
        num_workers=0,  # Must be 0 for LRU cache to work
        collate_fn=expert_collate_fn,
    )

    # Eval dataset
    eval_cfg = cfg.get("eval", {})
    eval_dataset = None
    if eval_cfg.get("enabled", False):
        eval_clip_ids_file = eval_cfg.get(
            "eval_clip_ids_file", data_cfg.get("exclude_clip_ids_file")
        )
        eval_hf = build_alpamayo_dataset(
            split=eval_cfg.get("eval_split", "val"),
            t0_us=data_cfg.get("t0_us", 5_100_000),
            max_samples=eval_cfg.get("eval_max_samples", 50),
            clip_ids_file=eval_clip_ids_file,
            avdi=avdi,
        )
        eval_dataset = ExpertTrainingDataset(
            clip_ids=[row["clip_id"] for row in eval_hf],
            t0_us=data_cfg.get("t0_us", 5_100_000),
            avdi=avdi,
            processor=processor,
        )
        logger.info("Built eval dataset: %d samples", len(eval_dataset))

    # ---------------------------------------------------------------
    # 4. Optimizer and scheduler
    # ---------------------------------------------------------------
    train_cfg = cfg.get("training", {})
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    num_epochs = int(train_cfg.get("num_epochs", 10))
    grad_acc_steps = int(train_cfg.get("gradient_accumulation_steps", 4))
    effective_batch_size = grad_acc_steps  # 1 scene per step * grad_acc
    total_steps = num_epochs * math.ceil(len(train_dataset) / effective_batch_size)
    warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.05)))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(train_cfg.get("learning_rate", 1e-4)),
        total_steps=total_steps,
        pct_start=warmup_steps / max(total_steps, 1),
        anneal_strategy="cos",
    )

    # ---------------------------------------------------------------
    # 5. TensorBoard writer
    # ---------------------------------------------------------------
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    # ---------------------------------------------------------------
    # 6. Training loop
    # ---------------------------------------------------------------
    num_noisy_samples = int(train_cfg.get("num_noisy_samples", 8))
    logging_steps = int(train_cfg.get("logging_steps", 10))
    save_steps = int(train_cfg.get("save_steps", 500))
    save_total_limit = int(train_cfg.get("save_total_limit", 3))
    eval_steps = int(eval_cfg.get("eval_steps", 200))

    global_step = 0
    running_loss = 0.0
    saved_checkpoints: list[Path] = []

    logger.info(
        "Starting training: %d epochs, %d samples, ~%d steps",
        num_epochs,
        len(train_dataset),
        total_steps,
    )

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            sample = batch[0]
            mi = helper.to_device(sample["model_inputs"], device=device)
            ego_future_xyz = sample["ego_future_xyz"].to(device)
            ego_future_rot = sample["ego_future_rot"].to(device)

            try:
                # Get VLM KV cache (frozen)
                prompt_cache, rope_deltas, prefill_seq_len, b_star, offset_tensor = (
                    get_vlm_kv_cache(model, mi, device)
                )

                # Convert GT trajectory to action space
                hist_xyz = mi["ego_history_xyz"][:, -1]  # (1, T_hist, 3)
                hist_rot = mi["ego_history_rot"][:, -1]  # (1, T_hist, 3, 3)
                fut_xyz = ego_future_xyz[:, -1]  # (1, T_fut, 3)
                fut_rot = ego_future_rot[:, -1]  # (1, T_fut, 3, 3)

                with torch.no_grad():
                    gt_action = model.action_space.traj_to_action(
                        hist_xyz, hist_rot, fut_xyz, fut_rot
                    )  # (1, T_future, 2)

                # Compute CFM loss
                with torch.autocast(str(device), dtype=torch.bfloat16):
                    loss = compute_cfm_loss(
                        model=model,
                        gt_action=gt_action,
                        prompt_cache=prompt_cache,
                        rope_deltas=rope_deltas,
                        prefill_seq_len=prefill_seq_len,
                        b_star=b_star,
                        offset=offset_tensor,
                        num_noisy_samples=num_noisy_samples,
                        device=device,
                    )
                    loss = loss / grad_acc_steps

                loss.backward()
                running_loss += loss.item()

            except Exception as e:
                logger.warning("Training step failed for batch %d: %s", batch_idx, e)
                continue

            # Gradient accumulation step
            if (batch_idx + 1) % grad_acc_steps == 0:
                max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "Step %d | Epoch %d | Loss: %.6f | LR: %.2e",
                        global_step,
                        epoch,
                        avg_loss,
                        lr,
                    )
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    running_loss = 0.0

                # Save checkpoint
                if global_step % save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    # Save only trainable weights
                    if use_expert_lora:
                        expert_sd = {
                            k: v
                            for k, v in model.expert.state_dict().items()
                            if "lora_" in k
                        }
                    else:
                        expert_sd = model.expert.state_dict()
                    expert_state = {
                        "expert": expert_sd,
                        "expert_lora": use_expert_lora,
                        "action_in_proj": model.action_in_proj.state_dict(),
                        "action_out_proj": model.action_out_proj.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }
                    torch.save(expert_state, ckpt_dir / "expert_checkpoint.pt")
                    logger.info("Saved checkpoint to %s", ckpt_dir)

                    saved_checkpoints.append(ckpt_dir)
                    if len(saved_checkpoints) > save_total_limit:
                        old_ckpt = saved_checkpoints.pop(0)
                        import shutil

                        shutil.rmtree(old_ckpt, ignore_errors=True)
                        logger.info("Removed old checkpoint %s", old_ckpt)

                # Evaluation
                if (
                    eval_dataset is not None
                    and eval_cfg.get("enabled", False)
                    and global_step % eval_steps == 0
                ):
                    logger.info("Running evaluation at step %d...", global_step)
                    eval_metrics = evaluate_expert(
                        model,
                        eval_dataset,
                        device,
                        num_diffusion_steps=int(eval_cfg.get("num_diffusion_eval_steps", 10)),
                        max_eval_samples=int(eval_cfg.get("eval_max_samples", 50)),
                    )
                    for k, v in eval_metrics.items():
                        writer.add_scalar(k, v, global_step)
                    logger.info("Eval step %d: %s", global_step, eval_metrics)

    # ---------------------------------------------------------------
    # 7. Save final model
    # ---------------------------------------------------------------
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if use_expert_lora:
        expert_sd = {
            k: v for k, v in model.expert.state_dict().items() if "lora_" in k
        }
    else:
        expert_sd = model.expert.state_dict()
    expert_state = {
        "expert": expert_sd,
        "expert_lora": use_expert_lora,
        "action_in_proj": model.action_in_proj.state_dict(),
        "action_out_proj": model.action_out_proj.state_dict(),
        "global_step": global_step,
    }
    torch.save(expert_state, final_dir / "expert_checkpoint.pt")
    logger.info("Saved final expert checkpoint to %s", final_dir)

    writer.close()
    logger.info("Training complete! %d steps, %d epochs.", global_step, num_epochs)


if __name__ == "__main__":
    main()
