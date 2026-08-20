# 🦴 Osteoporosis Detection Using Deep Transfer Learning & Explainable AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![TorchCAM](https://img.shields.io/badge/TorchCAM-Explainable%20AI-009688.svg)](https://github.com/frgousseau/torchcam)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced AI-driven medical image classification system for osteoporosis detection using female panoramic jaw radiograph (X-ray) images and multi-model deep transfer learning architectures (**ResNet18**, **ResNet50**, and **EfficientNet-B0**).

The system classifies jaw radiograph images into three progressive clinical stages, runs real-time consensus predictions across multiple deep learning backbones, provides explainable visual heatmaps using **SmoothGrad-CAM++**, generates comparative metrics & ROC curves, and delivers a responsive, modern Streamlit diagnostic dashboard.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Core Features](#-core-features)
- [Dataset Information](#-dataset-information)
- [Model Performance & Benchmarks](#-model-performance--benchmarks)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Pipeline Execution Guide](#-pipeline-execution-guide)
  - [1. Model Training](#1-model-training)
  - [2. Model Evaluation & Benchmarking](#2-model-evaluation--benchmarking)
  - [3. CLI Single Prediction](#3-cli-single-prediction)
  - [4. Explainable AI & Curve Plotting](#4-explainable-ai--curve-plotting)
- [Streamlit Diagnostic Web App](#-streamlit-diagnostic-web-app)
- [Explainable AI (XAI) Details](#-explainable-ai-xai-details)
- [Medical Disclaimer](#-medical-disclaimer)
- [Author & Credits](#-author--credits)

---

## 📌 Project Overview

Osteoporosis is a systemic skeletal disorder characterized by low bone mass and micro-architectural deterioration of bone tissue, leading to enhanced bone fragility and fracture risk. Early detection through routine dental panoramic radiographs provides an opportunity for early intervention.

This project assists healthcare professionals and researchers by automatically classifying female jawline X-ray images into three distinct clinical stages:

* **C1 (Normal):** Healthy bone density with intact trabecular structure.
* **C2 (Osteopenia):** Mildly reduced bone density (precursor stage requiring monitoring).
* **C3 (Osteoporosis):** Significantly reduced bone density with elevated fracture risk.

The system integrates multi-model deep learning and Explainable AI (XAI) to ensure high diagnostic transparency, reliability, and clinical auditability.

---

## 🚀 Core Features

* **Multi-Model Architecture Support:** Fully implemented training, evaluation, and inference pipelines for 3 backbones:
  * **ResNet18:** Lightweight and fast baseline architecture.
  * **ResNet50:** Deeper residual network with increased capacity for complex feature maps.
  * **EfficientNet-B0:** State-of-the-art parameter-efficient architecture utilizing depthwise separable convolutions.
* **Imbalanced Dataset Handling:** Incorporates medical-safe image augmentations and weighted cross-entropy loss to mitigate class imbalance biases (C1 vs C2 vs C3).
* **Professional Medical Report Generator:** Automatically generates structured, publication-quality medical reports featuring patient metadata, primary diagnosis, morphometric cortical observations, clinical action protocols, XAI Grad-CAM saliency heatmaps, multi-model consensus scores, and downloadable PDF report generation (`fpdf2`).
* **Explainable AI (SmoothGrad-CAM++):** Visualizes feature importance over the mandibular cortical and trabecular regions using class activation mapping.
* **Comprehensive Evaluation Toolkit:** Automated generation of confusion matrices, multi-class ROC/AUC curves, per-class metrics (Accuracy, F1-Score, Sensitivity), and training loss/accuracy history plots.
* **Interactive Streamlit Dashboard:** Built with custom CSS styling using Google Outfit typography, teal-cyan gradients, glassmorphic containers, mobile responsiveness, dark mode design, and a dedicated Medical Report tab.

---

## 📂 Dataset Information

The system is trained and evaluated on female panoramic jaw X-ray images classified under dental supervision:

| Class Code | Clinical Diagnostic Label | Description | Sample Count |
| :--- | :--- | :--- | :--- |
| **C1** | Normal | Healthy bone mass and density | 286 |
| **C2** | Osteopenia | Mild bone mass loss (pre-osteoporosis) | 118 |
| **C3** | Osteoporosis | Severe bone mass loss & high fracture risk | 55 |
| **Total** | | | **459 images** |

*Dataset Path:* `dataset/` (configured in [config.py](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/src/config.py))

---

## 📊 Model Performance & Benchmarks

Evaluation metrics computed across the validation set (stored in [model_comparison.json](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/outputs/model_comparison.json)):

| Model Architecture | Accuracy | F1-Score (Weighted) | Sensitivity (Recall) | Model File |
| :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-B0 ⭐ (Best Model)** | **69.57%** | **0.669** | **69.57%** | [efficientnet_b0.pth](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/models/efficientnet_b0.pth) |
| **ResNet18** | **65.22%** | **0.651** | **65.22%** | [resnet18.pth](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/models/resnet18.pth) |
| **ResNet50** | **62.32%** | **0.576** | **62.32%** | [resnet50.pth](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/models/resnet50.pth) |
| **DenseNet121** | 59.42% | 0.564 | 59.42% | [densenet121.pth](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/models/densenet121.pth) |

---

## 📁 Project Structure

```text
Medical_Project/
├── dataset/                        # Dataset partitioned into class subdirectories
│   ├── C1/                         # Normal panoramic jaw X-rays
│   ├── C2/                         # Osteopenia panoramic jaw X-rays
│   └── C3/                         # Osteoporosis panoramic jaw X-rays
├── models/                         # Model checkpoint directory (.pth)
│   ├── resnet18.pth
│   ├── resnet50.pth
│   └── efficientnet_b0.pth
├── outputs/                        # Generated evaluation & XAI artifacts
│   ├── confusion_matrix/           # Confusion matrix heatmaps per model
│   ├── curves/                     # ROC curves and loss/accuracy plots
│   ├── gradcam/                    # Offline Grad-CAM saliency heatmaps
│   └── model_comparison.json       # Benchmarking JSON report
├── src/                            # Core Python source code directory
│   ├── __init__.py
│   ├── config.py                   # Global hyperparameters & path configurations
│   ├── dataset.py                  # PyTorch Dataset & DataLoader utilities
│   ├── models.py                   # Model builders (ResNet18, ResNet50, EfficientNetB0)
│   ├── utils.py                    # Helper utilities & logging handlers
│   ├── train_resnet18.py           # Training script for ResNet18
│   ├── train_resnet50.py           # Training script for ResNet50
│   ├── train_efficientnet.py       # Training script for EfficientNetB0
│   ├── evaluate_resnet18.py        # Evaluation script for ResNet18
│   ├── evaluate_resnet50.py        # Evaluation script for ResNet50
│   ├── evaluate_efficientnet.py    # Evaluation script for EfficientNetB0
│   ├── evaluate_all.py             # Consolidated multi-model benchmark evaluation
│   ├── predict_resnet18.py         # CLI prediction script for ResNet18
│   ├── predict_resnet50.py         # CLI prediction script for ResNet50
│   ├── predict_efficientnet.py     # CLI prediction script for EfficientNetB0
│   ├── gradcam_resnet18.py         # Grad-CAM heatmap visualization script
│   ├── roc_resnet18.py             # ROC curve generator script for ResNet18
│   ├── plot_curves.py              # Loss and accuracy curve plotting tool
│   └── report_generator.py         # PDF medical report generator engine (`fpdf2`)
├── app.py                          # Streamlit web application dashboard
├── requirements.txt                # Python environment dependencies
└── README.md                       # Project documentation
```

Key Source File Links:
- Configuration: [config.py](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/src/config.py)
- Model Architectures: [models.py](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/src/models.py)
- Dataset Pipeline: [dataset.py](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/src/dataset.py)
- Report Generator: [report_generator.py](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/src/report_generator.py)
- Web Application: [app.py](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/app.py)
- Benchmark Results: [model_comparison.json](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/outputs/model_comparison.json)

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- `pip` package manager
- (Optional) CUDA-enabled GPU for faster training/inference

### Steps

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd Medical_Project
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Linux/macOS
   # .venv\Scripts\activate    # On Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Pipeline Execution Guide

All execution hyperparameters (batch size, learning rate, epochs, image resolution) are centralized in [config.py](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/src/config.py).

### 1. Model Training

Train models individually:

* **Train ResNet18:**
  ```bash
  python src/train_resnet18.py
  ```
* **Train ResNet50:**
  ```bash
  python src/train_resnet50.py
  ```
* **Train EfficientNet-B0:**
  ```bash
  python src/train_efficientnet.py
  ```

Trained weights will be saved to `models/`.

### 2. Model Evaluation & Benchmarking

* **Evaluate Individual Models:**
  ```bash
  python src/evaluate_resnet18.py
  python src/evaluate_resnet50.py
  python src/evaluate_efficientnet.py
  ```

* **Run Comprehensive Multi-Model Benchmark:**
  ```bash
  python src/evaluate_all.py
  ```
  This script evaluates all models, computes confusion matrices, plots ROC curves, and outputs [model_comparison.json](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/outputs/model_comparison.json).

### 3. CLI Single Prediction

Perform single-image inference via the command line:

* **ResNet18 Prediction:**
  ```bash
  python src/predict_resnet18.py
  ```
* **ResNet50 Prediction:**
  ```bash
  python src/predict_resnet50.py
  ```
* **EfficientNet-B0 Prediction:**
  ```bash
  python src/predict_efficientnet.py
  ```

*Input Prompt:* Enter the relative or absolute path of an image when prompted.

### 4. Explainable AI & Curve Plotting

* **Generate Grad-CAM Saliency Map:**
  ```bash
  python src/gradcam_resnet18.py
  ```
* **Generate ResNet18 ROC Curves:**
  ```bash
  python src/roc_resnet18.py
  ```
* **Plot Accuracy/Loss Curves:**
  ```bash
  python src/plot_curves.py
  ```

---

## 🌐 Streamlit Diagnostic Web App

Launch the interactive web application built with Streamlit in [app.py](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/app.py):

```bash
streamlit run app.py
```

### Dashboard Features
- **Radiograph Image Uploader:** Drag-and-drop support for PNG/JPG/JPEG jaw radiographs.
- **Consensus Clinical Diagnosis:** Displays predictions, confidence percentages, and voting consensus across ResNet18, ResNet50, and EfficientNet-B0.
- **Interactive Explainable AI (Grad-CAM):** Dynamic activation overlay selection allowing visual inspection of cortical/trabecular focus regions.
- **Publication-Quality PDF Medical Report Export:** Dedicated interactive tab allowing patient metadata input, custom morphometric findings, clinical recommendation protocol selection, and downloadable PDF report generation powered by [report_generator.py](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/Medical_Project/src/report_generator.py).
- **Performance Analytics Center:** Embedded visualization tabs for confusion matrices, ROC curves, accuracy/loss trends, and summary benchmark tables.
- **Custom UI System:** Dark glassmorphic design, custom Google Fonts (`Outfit`), responsive grid layouts, and animated status indicators.

---

## 🔬 Explainable AI (XAI) Details

To address the "black box" challenge of deep neural networks in clinical environments, this project uses **SmoothGrad-CAM++** (via `torchcam`).

SmoothGrad-CAM++ calculates visual saliency maps by taking gradients of the target class score relative to feature maps of the final convolutional layer. In jaw radiographs, key diagnostic indicators—such as the thickness and density of the inferior border of the mandible (cortical bone index) and bone trabeculation patterns—are highlighted with warm heatmap hues (red/yellow), allowing clinicians to verify AI feature attribution.

---

## ⚠️ Medical Disclaimer

This project is developed strictly for **research, educational, and experimental decision-support purposes**. It is **not** a certified medical device and should **not** be used as a standalone diagnostic system or to replace the clinical judgment of qualified dentists, oral radiologists, or medical professionals.

---

## 👨‍💻 Author & Credits

**Ravi Panchal**  
*B.Tech Final Year Student | Deep Learning & Medical Computer Vision Specialist*  

*Made with ♥ by Ravi Panchal. All Rights Reserved © 2026*
