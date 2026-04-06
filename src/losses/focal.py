import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for classification (Lin et al., 2017).

    Instantiated via Hydra ``_target_: src.losses.focal.FocalLoss``.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values down-weight easy examples more.
    alpha : float or None
        Balancing factor. If None, no class balancing is applied.
    reduction : str
        ``"mean"`` | ``"sum"`` | ``"none"``.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) raw logits.
            targets: (N,) integer class labels.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1.0 - pt) ** self.gamma

        loss = focal_weight * ce_loss

        if self.alpha is not None:
            loss = self.alpha * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
