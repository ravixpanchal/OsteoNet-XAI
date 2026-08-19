import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from src.config import (
    EPOCHS,
    LEARNING_RATE,
    RESNET18_PATH
)

from src.dataset import (
    train_loader,
    val_loader
)

from src.models import (
    get_resnet18_model,
    get_device
)

from src.utils import calculate_class_weights


# =====================================
# DEVICE
# =====================================

device = get_device()


# =====================================
# MODEL
# =====================================

model = get_resnet18_model()
model = model.to(device)


# =====================================
# CLASS WEIGHTS
# =====================================

weights = calculate_class_weights(
    train_loader.dataset
)

weights = weights.to(device)


criterion = nn.CrossEntropyLoss(
    weight=weights
)


# =====================================
# OPTIMIZER
# =====================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================
# TRAINING HISTORY
# =====================================

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []


# =====================================
# TRAINING LOOP
# =====================================

best_val_accuracy = 0


for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    # -------------------------
    # TRAINING
    # -------------------------

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader)

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix(
            loss=loss.item()
        )

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    running_val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            _, predicted = outputs.max(1)

            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()


    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)


    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        os.makedirs("models", exist_ok=True)

        torch.save(
            model.state_dict(),
            RESNET18_PATH
        )

        print("✅ Best model saved!")


print("\nTraining Finished Successfully!")
print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

# =====================================
# SAVE TRAINING HISTORY
# =====================================

history = {

    "train_losses": train_losses,

    "val_losses": val_losses,

    "train_accuracies": train_accuracies,

    "val_accuracies": val_accuracies
}


with open(
    "outputs/resnet18_history.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


print("\nTraining history saved!")