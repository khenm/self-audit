import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Model(nn.Module):
    """ResNet-18 wrapper for classification.

    Instantiated via Hydra ``_target_: src.models.model.ResNet18Model``.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
