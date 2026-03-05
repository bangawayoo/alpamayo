"""
TRL GRPO training script for a standard Qwen model using vLLM server mode.

This script is meant to be launched on GPUs separate from the vLLM server.
See scripts/run_trl_grpo_qwen.sh for the full pipeline.

Usage:
    # Server mode (vLLM server must already be running):
    CUDA_VISIBLE_DEVICES=1 accelerate launch scripts/trl_grpo_qwen.py

    # Server mode with custom args:
    CUDA_VISIBLE_DEVICES=1 accelerate launch scripts/trl_grpo_qwen.py \
        --model Qwen/Qwen2.5-3B --dataset trl-lib/DeepMath-103K
"""

import argparse
import re

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# Reward function: check math answer accuracy
# ---------------------------------------------------------------------------
def accuracy_reward(completions, **kwargs) -> list[float]:
    """Simple math accuracy reward — checks if the answer inside \\boxed{} matches.

    TRL passes completions as list[list[dict]] (chat messages) or list[str].
    We handle both formats.
    """
    solutions = kwargs.get("answer", kwargs.get("solution", []))
    rewards = []
    for completion, solution in zip(completions, solutions):
        # Extract text from chat-formatted completion
        if isinstance(completion, list):
            # Chat format: [{"role": "assistant", "content": "..."}]
            text = completion[-1]["content"] if completion else ""
        else:
            text = completion
        # Extract content inside \boxed{...}
        match = re.search(r"\\boxed\{([^}]*)\}", text)
        predicted = match.group(1).strip() if match else ""
        gold = str(solution).strip()
        rewards.append(1.0 if predicted == gold else 0.0)
    return rewards


def format_prompt(example: dict) -> dict:
    """Format dataset examples into chat messages for the model."""
    prompt = example.get("problem", example.get("question", ""))
    example["prompt"] = [
        {"role": "system", "content": "Solve the following math problem step by step. Put your final answer within \\boxed{}."},
        {"role": "user", "content": prompt},
    ]
    return example


def main():
    parser = argparse.ArgumentParser(description="TRL GRPO with vLLM server for Qwen")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model ID")
    parser.add_argument("--dataset", type=str, default="trl-lib/DeepMath-103K",
                        help="HuggingFace dataset ID")
    parser.add_argument("--dataset-split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size (for testing)")
    parser.add_argument("--output-dir", type=str, default="outputs/trl_grpo_qwen",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of completions per prompt for GRPO")
    parser.add_argument("--max-completion-length", type=int, default=512,
                        help="Max tokens to generate per completion")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="Learning rate")
    parser.add_argument("--vllm-server-url", type=str, default="http://localhost:8000",
                        help="URL of the vLLM server")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Disable vLLM (use standard generation)")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--report-to", type=str, default="tensorboard")
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------
    print(f"Loading dataset: {args.dataset} (split={args.dataset_split})")
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = dataset.map(format_prompt)
    print(f"Dataset size: {len(dataset)} examples")

    # -------------------------------------------------------------------
    # GRPOConfig
    # -------------------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        # vLLM settings
        use_vllm=not args.no_vllm,
        vllm_mode="server",
        # Generation
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # Training
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        # Logging / saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        # Misc
        bf16=True,
        max_grad_norm=0.1,
        warmup_ratio=0.1,
        log_completions=True,
    )

    # -------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------
    print(f"Initializing GRPOTrainer with model: {args.model}")
    print(f"  vLLM enabled: {not args.no_vllm}, mode: server")
    print(f"  num_generations: {args.num_generations}")

    trainer = GRPOTrainer(
        model=args.model,
        args=grpo_config,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Save final model
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
