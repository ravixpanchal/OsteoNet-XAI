import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from src.config import (
    EPOCHS,
    LEARNING_RATE,
    DENSENET121_PATH
)

from src.dataset import (
    train_loader,
    val_loader
)

from src.models import (
    get_densenet121_model,
    get_device
)


device = get_device()

model = get_densenet121_model()
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-2
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    model.train()
    running_loss = 0.0
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

        progress_bar.set_postfix(loss=loss.item())

    scheduler.step()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
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
    val_accuracy = 100.0 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), DENSENET121_PATH)
        print("✅ Best DenseNet121 model saved!")

print(f"\nDenseNet121 Training Finished! Best Validation Accuracy: {best_val_accuracy:.2f}%")

history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accuracies": train_accuracies,
    "val_accuracies": val_accuracies
}

os.makedirs("outputs", exist_ok=True)
with open("outputs/densenet121_history.json", "w") as f:
    json.dump(history, f, indent=4)

print("Training history saved to outputs/densenet121_history.json!")
