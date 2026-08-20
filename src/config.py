# =====================================
# DATASET CONFIGURATION
# =====================================

DATASET_PATH = "dataset"

CLASS_NAMES = [
    "C1",
    "C2",
    "C3"
]

NUM_CLASSES = len(CLASS_NAMES)


# =====================================
# IMAGE CONFIGURATION
# =====================================

IMAGE_SIZE = 224


# =====================================
# TRAINING CONFIGURATION
# =====================================

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.0001


# =====================================
# MODEL SAVE PATHS
# =====================================

RESNET18_PATH = "models/resnet18.pth"

RESNET50_PATH = "models/resnet50.pth"

EFFICIENTNET_PATH = "models/efficientnet_b0.pth"

DENSENET121_PATH = "models/densenet121.pth"


# =====================================
# OUTPUT PATHS
# =====================================

CONFUSION_MATRIX_PATH = "outputs/confusion_matrix"

CURVES_PATH = "outputs/curves"

GRADCAM_PATH = "outputs/gradcam"


# =====================================
# RANDOM SEED
# =====================================

RANDOM_SEED = 42