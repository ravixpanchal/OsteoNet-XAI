import torch

from collections import Counter


def calculate_class_weights(dataset):

    if hasattr(dataset, "indices") and hasattr(dataset.dataset, "targets"):
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    elif hasattr(dataset, "targets"):
        labels = dataset.targets
    else:
        labels = [label for _, label in dataset]

    class_counts = Counter(labels)

    print("\nClass Distribution:")

    for cls in sorted(class_counts.keys()):

        print(f"Class {cls}: {class_counts[cls]} images")


    total_samples = sum(class_counts.values())

    num_classes = len(class_counts)


    weights = []

    for i in range(num_classes):

        weight = total_samples / (num_classes * class_counts[i])

        weights.append(weight)


    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )


    print("\nClass Weights:")

    print(weights)

    return weights