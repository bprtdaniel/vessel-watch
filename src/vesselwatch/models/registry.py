from __future__ import annotations

import torch.nn as nn
from torchvision import models as tv_models

from .small_cnn import Net
from .vgg_style import Net_Max

CUSTOM = {"net": Net, "net_max": Net_Max}
RESNETS = ("resnet18", "resnet50", "resnet101")
MODEL_NAMES = (*CUSTOM, *RESNETS)


def build_resnet(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """torchvision ResNet with its final layer replaced for `num_classes`."""
    # IMAGENET1K_V1 is what the original `pretrained=True` loaded.
    weights = tv_models.get_model_weights(name).IMAGENET1K_V1 if pretrained else None
    model = tv_models.get_model(name, weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if name in CUSTOM:
        if pretrained:
            raise ValueError(f"No pretrained weights for custom model '{name}'")
        return CUSTOM[name](num_classes=num_classes)
    if name in RESNETS:
        return build_resnet(name, num_classes, pretrained)
    raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODEL_NAMES)}")
