"""
Pretraining entrypoint for EchoNet cardiac models.

Pretrain mode enforces that every clip contains both the ED and ES landmark
frames, so the model is always exposed to a complete cardiac cycle. This is
distinct from the standard `train.py` which uses all sliding-window clips.

Usage:
    uv run python scripts/pretrain.py
    uv run python scripts/pretrain.py --config train_cardiac_mamba
    uv run torchrun --nproc_per_node=4 scripts/pretrain.py --config train_cardiac_mamba
    # With pre-computed flow:
    uv run torchrun --nproc_per_node=4 scripts/pretrain.py \\
        data.train.flow_dir=data/flow data.val.flow_dir=data/flow
"""

import argparse

from hydra import compose, initialize

from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Pretrain cardiac echo model on EchoNet-Dynamic")
    parser.add_argument(
        "--config",
        type=str,
        default="train_cardiac_mamba",
        help="Config name (without .yaml), default: train_cardiac_mamba",
    )
    args, overrides = parser.parse_known_args()
    
    override_keys = {o.split("=")[0] for o in overrides}
    pretrain_overrides = []
    if "data.train.pretrain" not in override_keys:
        pretrain_overrides.append("data.train.pretrain=true")
    if "data.val.pretrain" not in override_keys:
        pretrain_overrides.append("data.val.pretrain=true")

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config, overrides=pretrain_overrides + overrides)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
