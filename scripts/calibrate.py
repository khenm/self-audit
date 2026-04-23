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
import logging
import torch.nn as nn
import torch


def main():
    parser = argparse.ArgumentParser(description="Phase 1b: c/γ calibration")
    parser.add_argument("--config", type=str, default="calibrate")
    args, overrides = parser.parse_known_args()

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config, overrides=overrides)

    trainer = FinetuneTrainer(**cfg)

    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    if hasattr(model, "volume_derivation"):
        logging.info("Randomly reinitializing the volume head weights for calibration stage.")
        with torch.no_grad():
            nn.init.normal_(model.volume_derivation.log_c)
            nn.init.normal_(model.volume_derivation.gamma_raw)
            nn.init.normal_(model.volume_derivation.bias)

    trainer.run()


if __name__ == "__main__":
    main()
