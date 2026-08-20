import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

from src.config import (
    CLASS_NAMES,
    RESNET18_PATH,
    RESNET50_PATH,
    EFFICIENTNET_PATH,
    DENSENET121_PATH,
    CONFUSION_MATRIX_PATH,
    CURVES_PATH
)

from src.dataset import test_loader

from src.models import (
    get_resnet18_model,
    get_resnet50_model,
    get_efficientnet_b0_model,
    get_densenet121_model,
    get_device
)

# Colors matching dark healthcare website theme
THEME_BG = '#131b2e'
CANVAS_BG = '#0f172a'
TEXT_MAIN = '#f8fafc'
TEXT_MUTED = '#94a3b8'
COLOR_C1 = '#10b981'   # Emerald Green
COLOR_C2 = '#f59e0b'   # Amber Yellow
COLOR_C3 = '#ef4444'   # Rose Red
COLOR_MICRO = '#38bdf8'# Light Cyan/Blue
COLOR_TRAIN = '#14b8a6'# Teal
COLOR_VAL = '#f59e0b'  # Amber

def apply_dark_theme(fig, ax):
    fig.patch.set_facecolor(THEME_BG)
    ax.set_facecolor(CANVAS_BG)
    for spine in ax.spines.values():
        spine.set_color('#334155')
        spine.set_linewidth(1.0)
    ax.tick_params(colors=TEXT_MUTED, which='both', labelsize=10)
    ax.yaxis.label.set_color(TEXT_MAIN)
    ax.xaxis.label.set_color(TEXT_MAIN)
    ax.title.set_color('#38bdf8')
    ax.title.set_weight('bold')
    ax.title.set_fontsize(12)
    ax.grid(True, linestyle='--', alpha=0.15, color='#ffffff')


def generate_plots_and_metrics(y_true, y_pred, y_scores, model_name):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = (y_pred == y_true).mean()
    
    model_key = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    os.makedirs(CONFUSION_MATRIX_PATH, exist_ok=True)
    os.makedirs(CURVES_PATH, exist_ok=True)
    
    # --------------------------------------------------
    # 1. CONFUSION MATRIX (Counts + Percentages)
    # --------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    cm_totals = cm.sum(axis=1, keepdims=True)
    cm_perc = np.divide(cm, cm_totals, out=np.zeros_like(cm, dtype=float), where=cm_totals!=0) * 100
    
    annot_labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"
            
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    apply_dark_theme(fig, ax)
    
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap=sns.dark_palette("#14b8a6", as_cmap=True),
        cbar=False,
        xticklabels=["C1 (Normal)", "C2 (Osteopenia)", "C3 (Osteoporosis)"],
        yticklabels=["C1 (Normal)", "C2 (Osteopenia)", "C3 (Osteoporosis)"],
        ax=ax,
        annot_kws={"size": 11, "weight": "bold", "color": "#f8fafc"}
    )
    
    ax.set_xlabel("Predicted Diagnostic Class", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("True Ground-Truth Class", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(f"{model_name} — Confusion Matrix", pad=12)
    
    cm_save_path = os.path.join(CONFUSION_MATRIX_PATH, f"{model_key}_cm.png")
    plt.tight_layout()
    plt.savefig(cm_save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    # Binarize labels for One-vs-Rest evaluation
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    # --------------------------------------------------
    # 2. MULTI-CLASS ROC CURVE (One-vs-Rest + Micro)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    apply_dark_theme(fig, ax)
    
    class_colors = [COLOR_C1, COLOR_C2, COLOR_C3]
    class_labels = ["C1 Normal", "C2 Osteopenia", "C3 Osteoporosis"]
    
    for i in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            color=class_colors[i],
            linewidth=2.2,
            label=f"{class_labels[i]} (AUC = {roc_auc:.3f})"
        )
        
    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(
        fpr_micro, tpr_micro,
        color=COLOR_MICRO,
        linestyle=':',
        linewidth=2.0,
        label=f"Micro-average (AUC = {auc_micro:.3f})"
    )
    
    # Random baseline
    ax.plot([0, 1], [0, 1], color='#64748b', linestyle='--', linewidth=1.5, label='Chance Baseline (AUC = 0.500)')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(f"{model_name} — Multi-Class ROC Curves", pad=12)
    ax.legend(loc="lower right", facecolor=CANVAS_BG, edgecolor='#334155', labelcolor=TEXT_MAIN, fontsize=9.5)
    
    roc_save_path = os.path.join(CURVES_PATH, f"{model_key}_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    # --------------------------------------------------
    # 3. PRECISION-RECALL CURVE
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    apply_dark_theme(fig, ax)
    
    for i in range(len(CLASS_NAMES)):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], y_scores[:, i])
        ax.plot(
            rec, prec,
            color=class_colors[i],
            linewidth=2.2,
            label=f"{class_labels[i]} (AP = {ap_score:.3f})"
        )
        
    prec_micro, rec_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_scores.ravel())
    ap_micro = average_precision_score(y_true_bin, y_scores, average="micro")
    ax.plot(
        rec_micro, prec_micro,
        color=COLOR_MICRO,
        linestyle=':',
        linewidth=2.0,
        label=f"Micro-average (AP = {ap_micro:.3f})"
    )
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(f"{model_name} — Precision–Recall Curves", pad=12)
    ax.legend(loc="lower left", facecolor=CANVAS_BG, edgecolor='#334155', labelcolor=TEXT_MAIN, fontsize=9.5)
    
    pr_save_path = os.path.join(CURVES_PATH, f"{model_key}_pr_curve.png")
    plt.tight_layout()
    plt.savefig(pr_save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "sensitivity": float(recall)
    }


def evaluate_model(model, device, test_loader, model_name):
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    return generate_plots_and_metrics(y_true, y_pred, y_scores, model_name)


def evaluate_ensemble(models, device, test_loader, model_name="Ensemble"):
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            batch_probs = []
            for model in models:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())
            
            avg_probs = np.mean(batch_probs, axis=0)
            y_scores.extend(avg_probs)
            y_true.extend(labels.numpy())
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.argmax(y_scores, axis=1)
    
    return generate_plots_and_metrics(y_true, y_pred, y_scores, model_name)


def plot_learning_curves(history_path, model_name):
    if not os.path.exists(history_path):
        print(f"History file {history_path} not found. Skipping learning curves plotting.")
        return
        
    with open(history_path, "r") as f:
        history = json.load(f)
        
    train_losses = history["train_losses"]
    val_losses = history["val_losses"]
    train_accuracies = history["train_accuracies"]
    val_accuracies = history["val_accuracies"]
    
    epochs = list(range(1, len(train_losses) + 1))
    model_key = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    apply_dark_theme(fig, ax)
    
    ax.plot(epochs, train_losses, color=COLOR_TRAIN, linewidth=2.2, label='Training Loss')
    ax.plot(epochs, val_losses, color=COLOR_VAL, linewidth=2.2, linestyle='--', label='Validation Loss')
    
    ax.scatter([epochs[-1]], [train_losses[-1]], color=COLOR_TRAIN, s=40, zorder=5)
    ax.scatter([epochs[-1]], [val_losses[-1]], color=COLOR_VAL, s=40, zorder=5)
    
    ax.set_xlabel("Epoch", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(f"{model_name} — Loss Convergence Curve", pad=12)
    ax.legend(loc="upper right", facecolor=CANVAS_BG, edgecolor='#334155', labelcolor=TEXT_MAIN, fontsize=10)
    
    loss_save_path = os.path.join(CURVES_PATH, f"{model_key}_loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    apply_dark_theme(fig, ax)
    
    ax.plot(epochs, train_accuracies, color=COLOR_TRAIN, linewidth=2.2, label='Training Accuracy (%)')
    ax.plot(epochs, val_accuracies, color=COLOR_C1, linewidth=2.2, linestyle='--', label='Validation Accuracy (%)')
    
    ax.scatter([epochs[-1]], [train_accuracies[-1]], color=COLOR_TRAIN, s=40, zorder=5)
    ax.scatter([epochs[-1]], [val_accuracies[-1]], color=COLOR_C1, s=40, zorder=5)
    
    ax.set_ylim([0, 105])
    ax.set_xlabel("Epoch", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(f"{model_name} — Accuracy Progression Curve", pad=12)
    ax.legend(loc="lower right", facecolor=CANVAS_BG, edgecolor='#334155', labelcolor=TEXT_MAIN, fontsize=10)
    
    acc_save_path = os.path.join(CURVES_PATH, f"{model_key}_accuracy_curve.png")
    plt.tight_layout()
    plt.savefig(acc_save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print(f"Saved diagnostic curves for {model_name}!")


def main():
    device = get_device()
    comparison_data = {}
    loaded_models = []
    
    # 1. ResNet18
    print("\n--- Evaluating ResNet18 ---")
    try:
        r18 = get_resnet18_model()
        r18.load_state_dict(torch.load(RESNET18_PATH, map_location=device))
        r18 = r18.to(device)
        r18.eval()
        metrics_r18 = evaluate_model(r18, device, test_loader, "ResNet18")
        comparison_data["ResNet18"] = metrics_r18
        plot_learning_curves("outputs/resnet18_history.json", "ResNet18")
        loaded_models.append(r18)
    except Exception as e:
        print(f"Error evaluating ResNet18: {e}")
        
    # 2. ResNet50
    print("\n--- Evaluating ResNet50 ---")
    try:
        r50 = get_resnet50_model()
        r50.load_state_dict(torch.load(RESNET50_PATH, map_location=device))
        r50 = r50.to(device)
        r50.eval()
        metrics_r50 = evaluate_model(r50, device, test_loader, "ResNet50")
        comparison_data["ResNet50"] = metrics_r50
        plot_learning_curves("outputs/resnet50_history.json", "ResNet50")
        loaded_models.append(r50)
    except Exception as e:
        print(f"Error evaluating ResNet50: {e}")
        
    # 3. EfficientNetB0
    print("\n--- Evaluating EfficientNetB0 ---")
    try:
        eff = get_efficientnet_b0_model()
        eff.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))
        eff = eff.to(device)
        eff.eval()
        metrics_eff = evaluate_model(eff, device, test_loader, "EfficientNetB0")
        comparison_data["EfficientNetB0"] = metrics_eff
        plot_learning_curves("outputs/efficientnet_history.json", "EfficientNetB0")
        loaded_models.append(eff)
    except Exception as e:
        print(f"Error evaluating EfficientNetB0: {e}")

    # 4. DenseNet121
    print("\n--- Evaluating DenseNet121 ---")
    try:
        dense = get_densenet121_model()
        dense.load_state_dict(torch.load(DENSENET121_PATH, map_location=device))
        dense = dense.to(device)
        dense.eval()
        metrics_dense = evaluate_model(dense, device, test_loader, "DenseNet121")
        comparison_data["DenseNet121"] = metrics_dense
        plot_learning_curves("outputs/densenet121_history.json", "DenseNet121")
        loaded_models.append(dense)
    except Exception as e:
        print(f"Error evaluating DenseNet121: {e}")

    # 5. Soft-Voting Ensemble
    if len(loaded_models) > 1:
        print("\n--- Evaluating Soft-Voting Ensemble ---")
        try:
            metrics_ensemble = evaluate_ensemble(loaded_models, device, test_loader, "Ensemble Model")
            comparison_data["Ensemble Model"] = metrics_ensemble
        except Exception as e:
            print(f"Error evaluating Ensemble: {e}")
        
    # Save comparison metrics JSON
    with open("outputs/model_comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=4)
        
    print("\nEvaluation metrics and diagnostic plots saved successfully!")


if __name__ == "__main__":
    main()
