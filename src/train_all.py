import os
import subprocess
import sys

def main():
    print("==========================================")
    print("  STARTING ALL MODEL TRAINING PIPELINE")
    print("==========================================")

    python_bin = sys.executable

    models_to_train = [
        ("ResNet18", "src.train_resnet18"),
        ("ResNet50", "src.train_resnet50"),
        ("EfficientNetB0", "src.train_efficientnet"),
        ("DenseNet121", "src.train_densenet121")
    ]

    for name, module in models_to_train:
        print(f"\n---> [TRAINING {name}] <---")
        cmd = [python_bin, "-m", module]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"❌ Error training {name}. Continuing to next model...")

    print("\n---> [RUNNING COMPREHENSIVE EVALUATION & ENSEMBLE BENCHMARK] <---")
    res = subprocess.run([python_bin, "-m", "src.evaluate_all"])
    if res.returncode == 0:
        print("\n🎉 ALL MODELS TRAINED AND EVALUATED SUCCESSFULLY!")
    else:
        print("\n⚠️ Evaluation encountered errors.")

if __name__ == "__main__":
    main()
