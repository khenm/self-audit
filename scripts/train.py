"""
Training entrypoint.

Usage:
    uv run python scripts/train.py --config train
    uv run torchrun --nproc_per_node=8 scripts/train.py --config train optim=sgd
"""

import argparse
import sys
import os
from hydra import compose, initialize
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train with composable Hydra configs")
    parser.add_argument(
        "--config",
        type=str,
        default="train",
        help="Config name (without .yaml), default: train",
    )
    args, overrides = parser.parse_known_args()

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config, overrides=overrides)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
