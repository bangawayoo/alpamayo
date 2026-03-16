"""Advantage-conditioned iterative SFT entry point for Alpamayo-R1.

Usage:
    python -m alpamayo_r1.training.train_sft --config-name sft_default

This script:
1. Loads the full AlpamayoR1 model
2. Registers advantage conditioning tokens
3. Builds the dataset and partitions scenes across iterations
4. Runs the self-play loop: rollout -> evaluate -> train (SFT + expert)

Each iteration resets to pi_0 (base model) and trains a fresh LoRA adapter
on advantage-conditioned completions from the current + historical rollouts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from physical_ai_av import PhysicalAIAVDatasetInterface

from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.training.advantage_conditioning import register_advantage_tokens
from alpamayo_r1.training.dataset import build_alpamayo_dataset
from alpamayo_r1.training.rollout_utils import prepare_vlm_for_training
from alpamayo_r1.training.selfplay_loop import SelfPlayLoop

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="sft_default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main advantage-conditioned iterative SFT function.

    Args:
        cfg: Hydra config with model, training, data, advantage_conditioning,
             expert_finetune, and reward settings.
    """
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Save resolved Hydra config
    output_dir = Path(cfg.get("training", {}).get("output_dir", "outputs/sft_advcond"))
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
    # 1. Load the full AlpamayoR1 model
    # ---------------------------------------------------------------
    adapter_config_path = Path(model_name) / "adapter_config.json"
    if adapter_config_path.exists():
        if base_model_name is None:
            raise ValueError(
                f"model_name={model_name!r} appears to be a LoRA adapter checkpoint "
                "(contains adapter_config.json). Set base_model_name= to the full "
                "AlpamayoR1 model (e.g., nvidia/Alpamayo-R1-10B)."
            )
        logger.info("Detected LoRA adapter; loading base model %s then applying", base_model_name)
        full_model = AlpamayoR1.from_pretrained_with_lora(
            adapter_path=model_name,
            base_model_name=base_model_name,
            dtype=torch.bfloat16,
            device_map=None,
            merge=True,
        )
    else:
        logger.info("Loading model: %s", model_name)
        full_model = AlpamayoR1.from_pretrained(model_name, dtype=torch.bfloat16)

    # ---------------------------------------------------------------
    # 2. Processor and dataset interface
    # ---------------------------------------------------------------
    processor = helper.get_processor(full_model.tokenizer)
    data_cfg = cfg.get("data", {})
    dataset_revision = data_cfg.get("dataset_revision", None)
    avdi = PhysicalAIAVDatasetInterface(revision=dataset_revision)

    # ---------------------------------------------------------------
    # 3. Register advantage tokens and resize embeddings
    # ---------------------------------------------------------------
    register_advantage_tokens(processor.tokenizer)
    full_model.vlm.resize_token_embeddings(len(processor.tokenizer))
    logger.info(
        "Registered advantage tokens and resized embeddings to %d",
        len(processor.tokenizer),
    )

    # ---------------------------------------------------------------
    # 4. Prepare VLM for training compatibility
    # ---------------------------------------------------------------
    prepare_vlm_for_training(full_model)

    # ---------------------------------------------------------------
    # 5. Build dataset and extract clip IDs for partitioning
    # ---------------------------------------------------------------
    dataset = build_alpamayo_dataset(
        split=data_cfg.get("split", "train"),
        t0_us=data_cfg.get("t0_us", 5_100_000),
        max_samples=data_cfg.get("max_samples", None),
        clip_ids_file=data_cfg.get("clip_ids_file", None),
        exclude_clip_ids_file=data_cfg.get("exclude_clip_ids_file", None),
        avdi=avdi,
    )
    all_clip_ids = dataset["clip_id"]
    logger.info("Dataset: %d scenes total", len(all_clip_ids))

    # ---------------------------------------------------------------
    # 6. Create and run self-play loop
    # ---------------------------------------------------------------
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    loop = SelfPlayLoop(
        cfg=cfg_dict,
        full_model=full_model,
        avdi=avdi,
        processor=processor,
        all_clip_ids=all_clip_ids,
    )

    logger.info("Starting advantage-conditioned iterative SFT...")
    loop.run()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
