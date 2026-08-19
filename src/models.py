import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from src.config import NUM_CLASSES


def get_resnet18_model():

    # Load pretrained ResNet18
    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    print("\nPretrained ResNet18 loaded successfully!")

    # Enable gradients for all layers (Fine-tuning)
    for param in model.parameters():
        param.requires_grad = True

    # Replace final fully connected layer
    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    # Train only the last layer
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def get_resnet50_model():

    # Load pretrained ResNet50
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    print("\nPretrained ResNet50 loaded successfully!")

    # Enable gradients for all layers (Fine-tuning)
    for param in model.parameters():
        param.requires_grad = True

    # Replace final fully connected layer
    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    # Train only the last layer
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def get_efficientnet_b0_model():

    # Load pretrained EfficientNetB0
    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    print("\nPretrained EfficientNetB0 loaded successfully!")

    # Enable gradients for all layers (Fine-tuning)
    for param in model.parameters():
        param.requires_grad = True

    # Replace final classifier layer
    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    # Train only the last layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("\nUsing GPU:", torch.cuda.get_device_name(0))

    else:

        device = torch.device("cpu")

        print("\nUsing CPU")

    return device