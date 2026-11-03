"""ResNet model builders for steel defect classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18_classifier(num_classes: int) -> nn.Module:
    """Create an ImageNet-pretrained ResNet18 with a new classification head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count all model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())
