"""
Phase 1b entrypoint — calibrates c and γ in V = c·A^γ with everything else frozen.

Usage:
    # Resume from Phase 1 pretrain checkpoint
    uv run python scripts/calibrate.py --config calibrate \\
        checkpoint.resume_checkpoint_path=/workspace/logs/pretrain_001/ckpts/checkpoint.pt

    # Multi-GPU
    uv run torchrun --nproc_per_node=8 scripts/calibrate.py --config calibrate \\
        checkpoint.resume_checkpoint_path=/workspace/logs/pretrain_001/ckpts/checkpoint.pt
"""

import argparse

from hydra import compose, initialize

from src.trainers.finetune_trainer import FinetuneTrainer


def main():
    parser = argparse.ArgumentParser(description="Phase 1b: c/γ calibration")
    parser.add_argument("--config", type=str, default="calibrate")
    args, overrides = parser.parse_known_args()

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config, overrides=overrides)

    trainer = FinetuneTrainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
