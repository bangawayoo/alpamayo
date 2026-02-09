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
1. Loads the full AlpamayoR1 model (VLM + expert + diffusion)
2. Freezes all non-VLM parameters (expert, diffusion, action space, projections)
3. Applies LoRA to the VLM's attention layers
4. Runs GRPO training with custom generation via AlpamayoGRPOTrainer
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
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
from alpamayo_r1.training.rollout import AlpamayoGRPOTrainer, RolloutLoggingCallback

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

    # Set seeds
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = cfg.get("device", "cuda")
    model_name = cfg.get("model_name", "nvidia/Alpamayo-R1-10B")

    # ---------------------------------------------------------------
    # 1. Load the full AlpamayoR1 model
    # ---------------------------------------------------------------
    logger.info("Loading model: %s", model_name)
    full_model = AlpamayoR1.from_pretrained(model_name, dtype=torch.bfloat16)
    full_model.to(device)

    # ---------------------------------------------------------------
    # 2. Freeze non-VLM parameters
    # ---------------------------------------------------------------
    _freeze_non_vlm_params(full_model)

    # ---------------------------------------------------------------
    # 3. Processor and dataset interface
    # ---------------------------------------------------------------
    processor = helper.get_processor(full_model.tokenizer)
    avdi = PhysicalAIAVDatasetInterface()

    # ---------------------------------------------------------------
    # 4. LoRA configuration for VLM text layers
    # ---------------------------------------------------------------
    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        task_type="CAUSAL_LM",
    )

    # ---------------------------------------------------------------
    # 5. GRPO training config
    # ---------------------------------------------------------------
    train_cfg = cfg.get("training", {})
    reward_cfg = cfg.get("rewards", {})
    reward_weights = [
        float(reward_cfg.get("trajectory_weight", 0.5)),
        float(reward_cfg.get("reasoning_weight", 0.25)),
        float(reward_cfg.get("consistency_weight", 0.25)),
    ]
    training_args = GRPOConfig(
        output_dir=train_cfg.get("output_dir", "outputs/grpo"),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(train_cfg.get("learning_rate", 1e-5)),
        num_generations=train_cfg.get("num_generations", 8),
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
        report_to=train_cfg.get("report_to", "tensorboard"),
        reward_weights=reward_weights,
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
        avdi=avdi,
    )

    # ---------------------------------------------------------------
    # 7. Create trainer and train
    # ---------------------------------------------------------------
    rollout_cfg = cfg.get("rollout", {})
    logger.info("Initializing AlpamayoGRPOTrainer...")
    trainer = AlpamayoGRPOTrainer(
        model=full_model.vlm,
        args=training_args,
        train_dataset=dataset,
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
    )

    # Rollout logging callback (CoC text + BEV trajectory plots to TensorBoard)
    rollout_log_interval = rollout_cfg.get(
        "log_interval", train_cfg.get("logging_steps", 10)
    )
    rollout_callback = RolloutLoggingCallback(
        log_interval=int(rollout_log_interval),
        max_samples=int(rollout_cfg.get("log_max_samples", 2)),
    )
    trainer.add_callback(rollout_callback)
    rollout_callback.trainer = trainer

    logger.info("Starting GRPO training...")
    trainer.train()

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
