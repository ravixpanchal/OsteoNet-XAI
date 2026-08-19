import os
import json

import matplotlib.pyplot as plt

from src.config import CURVES_PATH


# =====================================
# LOAD HISTORY
# =====================================

with open(
    "outputs/resnet18_history.json",
    "r"
) as f:

    history = json.load(f)


train_losses = history["train_losses"]
val_losses = history["val_losses"]

train_accuracies = history["train_accuracies"]
val_accuracies = history["val_accuracies"]


epochs = range(
    1,
    len(train_losses) + 1
)


os.makedirs(
    CURVES_PATH,
    exist_ok=True
)


# =====================================
# LOSS CURVE
# =====================================

plt.figure(figsize=(8, 6))

plt.plot(
    epochs,
    train_losses,
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("ResNet18 Loss Curve")

plt.legend()

plt.grid(True)

plt.savefig(
    f"{CURVES_PATH}/resnet18_loss_curve.png"
)

plt.show()


# =====================================
# ACCURACY CURVE
# =====================================

plt.figure(figsize=(8, 6))

plt.plot(
    epochs,
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title("ResNet18 Accuracy Curve")

plt.legend()

plt.grid(True)

plt.savefig(
    f"{CURVES_PATH}/resnet18_accuracy_curve.png"
)

plt.show()


print("\nCurves saved successfully!")