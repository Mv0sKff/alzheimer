from torch import nn
from torchvision import models

from schemas import CLASSES

BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
}
NUM_CLASSES = len(CLASSES)


class MyModel(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, num_classes=NUM_CLASSES):
        super().__init__()
        build, weights = BACKBONES[backbone]
        self.model = build(weights=weights if pretrained else None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
