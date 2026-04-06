import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def robust_torch_save(state: Dict[str, Any], path: str):
    """Save a checkpoint with a backup-swap strategy to survive preemptions.

    1. If *path* already exists, rename it to *path.bak*.
    2. Write the new checkpoint to *path*.
    3. Delete the backup.
    """
    backup = path + ".bak"
    has_backup = False

    if os.path.exists(path):
        if os.path.exists(backup):
            os.remove(backup)
        os.rename(path, backup)
        has_backup = True

    with open(path, "wb") as f:
        torch.save(state, f)

    if has_backup and os.path.exists(backup):
        os.remove(backup)


class CheckpointSaver:
    """Saves training checkpoints (rank-0 only)."""

    def __init__(
        self,
        checkpoint_folder: str,
        checkpoint_names: List[str],
        rank: int,
        epoch: int,
    ):
        self.folder = checkpoint_folder
        self.names = checkpoint_names
        self.rank = rank
        self.epoch = epoch

    def save(self, model: nn.Module, **extra):
        """Save *model* state dict plus any extra tensors (optimizer, scaler, …)."""
        if self.rank != 0:
            return

        state = dict(**extra)
        state["model"] = model.state_dict()

        os.makedirs(self.folder, exist_ok=True)
        for name in self.names:
            path = os.path.join(self.folder, f"{name}.pt")
            logging.info(f"Saving checkpoint epoch={self.epoch} → {path}")
            robust_torch_save(state, path)
