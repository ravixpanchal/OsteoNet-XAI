import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from src.config import (
    CLASS_NAMES,
    EFFICIENTNET_PATH,
    CONFUSION_MATRIX_PATH
)

from src.dataset import test_loader

from src.models import (
    get_efficientnet_b0_model,
    get_device
)


# =====================================
# DEVICE
# =====================================

device = get_device()


# =====================================
# LOAD MODEL
# =====================================

model = get_efficientnet_b0_model()

model.load_state_dict(
    torch.load(
        EFFICIENTNET_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# =====================================
# PREDICTIONS
# =====================================

y_true = []
y_pred = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())


# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\nClassification Report:\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
)

print(report)


# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("EfficientNetB0 Confusion Matrix")


os.makedirs(
    CONFUSION_MATRIX_PATH,
    exist_ok=True
)


save_path = f"{CONFUSION_MATRIX_PATH}/efficientnetb0_cm.png"

plt.savefig(
    save_path,
    bbox_inches="tight"
)

plt.show()


print(f"\nConfusion Matrix saved at:\n{save_path}")
