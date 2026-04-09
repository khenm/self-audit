"""
Finetuning entrypoint — uses FinetuneTrainer for cardiac evaluation metrics.

Usage:
    uv run python scripts/finetune.py --config finetune
    uv run python scripts/finetune.py --config finetune max_epochs=100

    # Resume from pretrain checkpoint
    uv run python scripts/finetune.py --config finetune \
        checkpoint.resume_checkpoint_path=/workspace/logs/pretrain_pseudomask_001/ckpts/checkpoint.pt

    # Multi-GPU
    uv run torchrun --nproc_per_node=8 scripts/finetune.py --config finetune
"""

import argparse

from hydra import compose, initialize

from src.trainers.finetune_trainer import FinetuneTrainer


def main():
    parser = argparse.ArgumentParser(description="Finetune with cardiac evaluation metrics")
    parser.add_argument("--config", type=str, default="finetune")
    args, overrides = parser.parse_known_args()

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config, overrides=overrides)

    trainer = FinetuneTrainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
