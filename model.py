from torch import nn
from torch.nn import Module
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

from args import *


class MyModel(Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()

        # TODO: could be done cleaner
        if backbone == "resnet18":
            self.model = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif backbone == "resnet34":
            self.model = resnet34(
                weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif backbone == "resnet50":
            self.model = resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
        else:
            self.model = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )

        self.model.fc = nn.Linear(self.model.fc.in_features, 5)

    def forward(self, x):
        return self.model(x)
