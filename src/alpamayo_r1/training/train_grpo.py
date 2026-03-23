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

"""GRPO post-training entry point for Alpamayo-R1.

Usage:
    python -m alpamayo_r1.training.train_grpo --config-name grpo_default

This script:
1. Loads the full AlpamayoR1 model (only VLM + trajectory tokenizers are
   moved to GPU; expert/diffusion/projections stay on CPU)
2. Freezes all non-VLM parameters
3. Optionally applies LoRA to the VLM's attention layers (lora.enabled=true,
   the default) or trains all VLM parameters (lora.enabled=false).
   Full-parameter training should use FSDP for multi-GPU sharding.
4. Runs GRPO training with VLM-only rollouts via AlpamayoGRPOTrainer
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from physical_ai_av import PhysicalAIAVDatasetInterface
from trl import GRPOConfig

from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.training.dataset import build_alpamayo_dataset
from alpamayo_r1.training.rewards import (
    consistency_reward,
    reasoning_quality_reward,
    trajectory_quality_reward,
)
from alpamayo_r1.training.rollout import (
    AlpamayoGRPOTrainer,
    GpuUtilizationCallback,
    RolloutLoggingCallback,
    prepare_vlm_for_training,
)

logger = logging.getLogger(__name__)


def _freeze_non_vlm_params(model: AlpamayoR1) -> None:
    """Freeze all parameters that are not part of the VLM backbone.

    Only VLM text-generation parameters will be trained via LoRA.
    Expert, diffusion, action space, and projections are frozen.

    Args:
        model: The full AlpamayoR1 model.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if not name.startswith("vlm."):
            param.requires_grad = False
            frozen_count += 1
    logger.info("Froze %d non-VLM parameter groups", frozen_count)


@hydra.main(config_path="configs", config_name="grpo_default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main GRPO training function.

    Args:
        cfg: Hydra config with model, training, data, and reward settings.
    """
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Save resolved Hydra config to training output_dir for reproducibility
    output_dir = Path(cfg.get("training", {}).get("output_dir", "outputs/grpo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = output_dir / "resolved_config.yaml"
    config_save_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info("Saved resolved config to %s", config_save_path)

    # Set seeds
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model_name = cfg.get("model_name", "nvidia/Alpamayo-R1-10B")
    base_model_name = cfg.get("base_model_name", None)

    # ---------------------------------------------------------------
    # 1. Load the full AlpamayoR1 model (on CPU; FSDP/accelerator
    #    will handle VLM device placement)
    # ---------------------------------------------------------------
    adapter_config_path = Path(model_name) / "adapter_config.json"
    if adapter_config_path.exists():
        if base_model_name is None:
            raise ValueError(
                f"model_name={model_name!r} appears to be a LoRA adapter checkpoint "
                "(contains adapter_config.json). Set base_model_name= to the full "
                "AlpamayoR1 model (e.g., nvidia/Alpamayo-R1-10B)."
            )
        logger.info(
            "Detected LoRA adapter checkpoint; loading base model %s then applying adapter",
            base_model_name,
        )
        full_model = AlpamayoR1.from_pretrained_with_lora(
            adapter_path=model_name,
            base_model_name=base_model_name,
            dtype=torch.bfloat16,
            device_map=None,  # keep on CPU; FSDP/accelerator handles placement
            merge=True,
        )
    else:
        logger.info("Loading model: %s", model_name)
        full_model = AlpamayoR1.from_pretrained(model_name, dtype=torch.bfloat16)

    # ---------------------------------------------------------------
    # 2. Freeze non-VLM parameters
    # ---------------------------------------------------------------
    _freeze_non_vlm_params(full_model)

    # ---------------------------------------------------------------
    # 3. Processor and dataset interface
    # ---------------------------------------------------------------
    processor = helper.get_processor(full_model.tokenizer)
    data_cfg = cfg.get("data", {})
    dataset_revision = data_cfg.get("dataset_revision", None)
    avdi = PhysicalAIAVDatasetInterface(revision=dataset_revision)

    # ---------------------------------------------------------------
    # 4. LoRA configuration (optional — disabled for full-parameter training)
    # ---------------------------------------------------------------
    lora_cfg = cfg.get("lora", {})
    use_lora = bool(lora_cfg.get("enabled", True))

    if use_lora:
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=list(
                lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
            ),
            task_type="CAUSAL_LM",
        )
        logger.info("LoRA enabled: r=%d, alpha=%d", lora_config.r, lora_config.lora_alpha)
    else:
        lora_config = None
        logger.info("LoRA disabled — full-parameter VLM training")

    # ---------------------------------------------------------------
    # 5. GRPO training config
    # ---------------------------------------------------------------
    early_stopping_cfg = cfg.get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
    train_cfg = cfg.get("training", {})
    reward_cfg = cfg.get("rewards", {})
    reward_weights = [
        float(reward_cfg.get("trajectory_weight", 0.5)),
        float(reward_cfg.get("reasoning_weight", 0.25)),
        float(reward_cfg.get("consistency_weight", 0.25)),
    ]
    num_generations = train_cfg.get("num_generations", 8)
    per_device_bs = train_cfg.get("per_device_train_batch_size", 4)
    grad_acc = train_cfg.get("gradient_accumulation_steps", 16)

    # vLLM configuration (colocate or server mode)
    vllm_cfg = cfg.get("vllm", {})
    vllm_enabled = bool(vllm_cfg.get("enabled", False))

    vllm_kwargs = {}
    if vllm_enabled:
        vllm_mode = str(vllm_cfg.get("mode", "colocate"))
        vllm_kwargs = dict(
            use_vllm=True,
            vllm_mode=vllm_mode,
            vllm_model_impl=str(vllm_cfg.get("model_impl", "transformers")),
        )
        if vllm_mode == "colocate":
            vllm_kwargs.update(
                vllm_gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.3)),
                vllm_tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
                vllm_max_model_length=vllm_cfg.get("max_model_length", None),
                vllm_enable_sleep_mode=bool(vllm_cfg.get("enable_sleep_mode", False)),
            )
        else:  # server mode
            vllm_kwargs.update(
                vllm_server_host=str(vllm_cfg.get("server_host", "0.0.0.0")),
                vllm_server_port=int(vllm_cfg.get("server_port", 8000)),
                vllm_server_timeout=float(vllm_cfg.get("server_timeout", 240.0)),
                vllm_group_port=int(vllm_cfg.get("group_port", 51216)),
            )
            server_base_url = vllm_cfg.get("server_base_url", None)
            if server_base_url is not None:
                vllm_kwargs["vllm_server_base_url"] = str(server_base_url)
        logger.info("vLLM %s mode enabled: %s", vllm_mode, vllm_kwargs)

    # Early stopping eval args (conditionally added to GRPOConfig)
    eval_kwargs = {}
    if early_stopping_enabled:
        es_metric = early_stopping_cfg.get("metric", "rewards/trajectory_quality_reward/mean")
        eval_kwargs = dict(
            eval_strategy="steps",
            eval_steps=int(early_stopping_cfg.get("eval_steps", 100)),
            per_device_eval_batch_size=per_device_bs,
            metric_for_best_model=es_metric,
            greater_is_better=True,
            load_best_model_at_end=True,
        )

    training_args = GRPOConfig(
        output_dir=train_cfg.get("output_dir", "outputs/grpo"),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_acc,
        learning_rate=float(train_cfg.get("learning_rate", 1e-5)),
        num_generations=num_generations,
        max_completion_length=train_cfg.get("max_completion_length", 256),
        beta=float(train_cfg.get("beta", 0.0)),
        loss_type=train_cfg.get("loss_type", "grpo"),
        bf16=True,
        logging_steps=train_cfg.get("logging_steps", 1),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        seed=seed,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        report_to=train_cfg.get("report_to", "tensorboard"),
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint", None),
        reward_weights=reward_weights,
        **eval_kwargs,
        **vllm_kwargs,
    )

    # ---------------------------------------------------------------
    # 6. Build dataset
    # ---------------------------------------------------------------
    data_cfg = cfg.get("data", {})
    dataset = build_alpamayo_dataset(
        split=data_cfg.get("split", "train"),
        t0_us=data_cfg.get("t0_us", 5_100_000),
        max_samples=data_cfg.get("max_samples", None),
        clip_ids_file=data_cfg.get("clip_ids_file", None),
        exclude_clip_ids_file=data_cfg.get("exclude_clip_ids_file", None),
        avdi=avdi,
    )

    # ---------------------------------------------------------------
    # 6b. Early stopping — build eval dataset from val split of curated clips
    # ---------------------------------------------------------------
    eval_dataset = None

    if early_stopping_enabled:
        eval_clip_ids_file = early_stopping_cfg.get(
            "eval_clip_ids_file", data_cfg.get("exclude_clip_ids_file")
        )
        eval_dataset = build_alpamayo_dataset(
            split=early_stopping_cfg.get("eval_split", "val"),
            t0_us=data_cfg.get("t0_us", 5_100_000),
            max_samples=early_stopping_cfg.get("eval_max_samples", 50),
            clip_ids_file=eval_clip_ids_file,
            avdi=avdi,
        )
        logger.info("Built eval dataset: %d samples", len(eval_dataset))

    # ---------------------------------------------------------------
    # 7. Create trainer and train
    # ---------------------------------------------------------------
    rollout_cfg = cfg.get("rollout", {})
    prepare_vlm_for_training(full_model)

    logger.info("Initializing AlpamayoGRPOTrainer...")
    trainer = AlpamayoGRPOTrainer(
        model=full_model.vlm,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[
            trajectory_quality_reward,
            reasoning_quality_reward,
            consistency_reward,
        ],
        processing_class=processor,
        peft_config=lora_config,
        # Alpamayo-specific args
        full_model=full_model,
        avdi=avdi,
        rollout_temperature=rollout_cfg.get("temperature", 0.6),
        rollout_top_p=rollout_cfg.get("top_p", 0.98),
        rollout_max_generation_length=rollout_cfg.get("max_generation_length", 256),
        logprob_mini_batch_size=int(rollout_cfg.get("logprob_mini_batch_size", 4)),
        data_cache_max_size=int(rollout_cfg.get("data_cache_max_size", 200)),
        value_head_cfg=dict(cfg.get("value_head", {})),
    )

    # Rollout logging callback (CoC text + BEV trajectory plots to TensorBoard)
    rollout_log_interval = rollout_cfg.get("log_interval", train_cfg.get("logging_steps", 10))
    rollout_plot_interval = rollout_cfg.get("plot_interval", None)
    if rollout_plot_interval is not None:
        rollout_plot_interval = int(rollout_plot_interval)
    rollout_callback = RolloutLoggingCallback(
        log_interval=int(rollout_log_interval),
        plot_interval=rollout_plot_interval,
        max_samples=int(rollout_cfg.get("log_max_samples", 2)),
    )
    trainer.add_callback(rollout_callback)
    rollout_callback.trainer = trainer

    trainer.add_callback(GpuUtilizationCallback())

    # Early stopping callback
    if early_stopping_enabled:
        from transformers import EarlyStoppingCallback

        es_patience = int(early_stopping_cfg.get("patience", 5))
        es_threshold = float(early_stopping_cfg.get("threshold", 0.01))
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=es_patience,
                early_stopping_threshold=es_threshold,
            )
        )
        logger.info(
            "Early stopping enabled: patience=%d, threshold=%.3f, metric=%s",
            es_patience,
            es_threshold,
            training_args.metric_for_best_model,
        )

    logger.info("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # ---------------------------------------------------------------
    # 8. Save final model
    # ---------------------------------------------------------------
    output_dir = Path(training_args.output_dir) / "final"
    logger.info("Saving final model to %s", output_dir)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
