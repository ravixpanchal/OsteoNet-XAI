import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    densenet121,
    DenseNet121_Weights
)

from src.config import NUM_CLASSES


def get_resnet18_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    print("\nPretrained ResNet18 loaded successfully!")

    for param in model.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, NUM_CLASSES)
    )
    return model


def get_resnet50_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    print("\nPretrained ResNet50 loaded successfully!")

    for param in model.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, NUM_CLASSES)
    )
    return model


def get_efficientnet_b0_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    print("\nPretrained EfficientNetB0 loaded successfully!")

    for param in model.parameters():
        param.requires_grad = True

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, NUM_CLASSES)
    )
    return model


def get_densenet121_model():
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    print("\nPretrained DenseNet121 loaded successfully!")

    for param in model.parameters():
        param.requires_grad = True

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, NUM_CLASSES)
    )
    return model


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nUsing GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("\nUsing CPU")
    return device