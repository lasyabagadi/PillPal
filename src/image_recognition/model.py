import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, backbone: str = "resnet18", pretrained: bool = True):
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        return m
    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_feats = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feats, num_classes)
        return m
    # else:
    #     raise ValueError("Unsupported backbone. Choose 'resnet18' or 'efficientnet_b0'.")