import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from src.config import (
    CLASS_NAMES,
    RESNET18_PATH,
    CURVES_PATH
)

from src.dataset import test_loader

from src.models import (
    get_resnet18_model,
    get_device
)


# =====================================
# DEVICE
# =====================================

device = get_device()


# =====================================
# LOAD TRAINED MODEL
# =====================================

model = get_resnet18_model()

model.load_state_dict(
    torch.load(
        RESNET18_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =====================================
# STORE TRUE LABELS & PROBABILITIES
# =====================================

y_true = []
y_scores = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        y_true.extend(labels.numpy())

        y_scores.extend(
            probabilities.cpu().numpy()
        )


# =====================================
# CONVERT TO NUMPY
# =====================================

y_true = np.array(y_true)
y_scores = np.array(y_scores)


# =====================================
# ONE-HOT ENCODING
# =====================================

y_true_bin = label_binarize(
    y_true,
    classes=[0, 1, 2]
)


# =====================================
# CREATE ROC CURVES
# =====================================

plt.figure(figsize=(8, 6))

for i in range(len(CLASS_NAMES)):

    fpr, tpr, _ = roc_curve(
        y_true_bin[:, i],
        y_scores[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})"
    )

    print(f"{CLASS_NAMES[i]} AUC: {roc_auc:.3f}")


# =====================================
# RANDOM CLASSIFIER LINE
# =====================================

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


# =====================================
# PLOT SETTINGS
# =====================================

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ResNet18 Multi-Class ROC Curve")

plt.legend()

plt.grid(True)


# =====================================
# SAVE FIGURE
# =====================================

os.makedirs(
    CURVES_PATH,
    exist_ok=True
)

save_path = f"{CURVES_PATH}/resnet18_roc_curve.png"

plt.savefig(
    save_path,
    bbox_inches="tight"
)

plt.show()


print("\nROC Curve Saved Successfully!")
print(save_path)