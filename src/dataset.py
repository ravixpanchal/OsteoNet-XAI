import random
import numpy as np

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import (
    DataLoader,
    Subset,
    WeightedRandomSampler
)

from sklearn.model_selection import train_test_split

from src.config import *


# =====================================
# SET RANDOM SEED
# =====================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)


# =====================================
# DATA AUGMENTATION (TRAIN ONLY)
# =====================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomRotation(10),

    transforms.RandomHorizontalFlip(),

    transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05)
    ),

    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    transforms.RandomAutocontrast(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================
# VALIDATION / TEST TRANSFORMS
# =====================================

test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================
# LOAD DATASET
# =====================================

full_dataset = datasets.ImageFolder(
    DATASET_PATH
)


# Labels of all images
targets = full_dataset.targets


# =====================================
# TRAIN + TEMP SPLIT (70% + 30%)
# =====================================

train_indices, temp_indices = train_test_split(

    range(len(targets)),

    test_size=0.30,

    stratify=targets,

    random_state=RANDOM_SEED
)


# =====================================
# VALIDATION + TEST SPLIT (15% + 15%)
# =====================================

temp_targets = [targets[i] for i in temp_indices]


val_indices, test_indices = train_test_split(

    temp_indices,

    test_size=0.50,

    stratify=temp_targets,

    random_state=RANDOM_SEED
)


# =====================================
# CREATE DATASETS WITH TRANSFORMS
# =====================================

train_dataset_full = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

test_dataset_full = datasets.ImageFolder(
    DATASET_PATH,
    transform=test_transform
)


train_dataset = Subset(
    train_dataset_full,
    train_indices
)

val_dataset = Subset(
    test_dataset_full,
    val_indices
)

test_dataset = Subset(
    test_dataset_full,
    test_indices
)


# =====================================
# DATALOADERS & SAMPLER FOR CLASS BALANCING
# =====================================

from collections import Counter

# Get targets of the train subset
train_targets = [targets[i] for i in train_indices]
train_class_counts = Counter(train_targets)
total_train_samples = len(train_targets)
num_classes = len(train_class_counts)

# Class weight is inversely proportional to its frequency
class_weights = {
    cls: total_train_samples / (num_classes * count)
    for cls, count in train_class_counts.items()
}

# Map each training sample to its corresponding class weight
sample_weights = [class_weights[t] for t in train_targets]

# Define WeightedRandomSampler
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=train_sampler
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False
)