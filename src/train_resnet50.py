import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from src.config import (
    EPOCHS,
    RESNET50_PATH
)

from src.dataset import (
    train_loader,
    val_loader,
    class_weights_tensor,
    FocalLoss
)

from src.models import (
    get_resnet50_model,
    get_device
)


device = get_device()

model = get_resnet50_model()
model = model.to(device)

criterion = FocalLoss(alpha=class_weights_tensor.to(device), gamma=1.5, label_smoothing=0.05)

# Freeze backbone initially (Phase 1)
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-4,
    weight_decay=1e-2
)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

best_val_f1 = 0.0
best_val_accuracy = 0.0
WARMUP_EPOCHS = 5

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    if epoch == WARMUP_EPOCHS:
        print("\n🔥 Unfreezing backbone for Phase 2 fine-tuning with differential learning rates...")
        for param in model.parameters():
            param.requires_grad = True

        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
        head_params = [p for n, p in model.named_parameters() if "fc" in n]

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 2e-5},
            {'params': head_params, 'lr': 2e-4}
        ], weight_decay=1e-2)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS - WARMUP_EPOCHS,
            eta_min=1e-6
        )

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

    if epoch >= WARMUP_EPOCHS:
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
    val_preds = []
    val_targets = []

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

            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100.0 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    _, _, val_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='weighted', zero_division=0)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_accuracy:.2f}% | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_accuracy > best_val_accuracy):
        best_val_f1 = val_f1
        best_val_accuracy = val_accuracy
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), RESNET50_PATH)
        print(f"✅ Best ResNet50 model saved! (Val F1: {val_f1:.4f}, Val Acc: {val_accuracy:.2f}%)")

print(f"\nResNet50 Training Finished! Best Validation F1: {best_val_f1:.4f}")

history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accuracies": train_accuracies,
    "val_accuracies": val_accuracies
}

os.makedirs("outputs", exist_ok=True)
with open("outputs/resnet50_history.json", "w") as f:
    json.dump(history, f, indent=4)

print("Training history saved to outputs/resnet50_history.json!")
