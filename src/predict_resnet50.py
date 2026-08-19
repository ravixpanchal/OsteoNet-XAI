import torch
from PIL import Image
from torchvision import transforms

from src.config import (
    IMAGE_SIZE,
    CLASS_NAMES,
    RESNET50_PATH
)

from src.models import (
    get_resnet50_model,
    get_device
)


# =====================================
# DEVICE
# =====================================

device = get_device()


# =====================================
# LOAD MODEL
# =====================================

model = get_resnet50_model()

model.load_state_dict(
    torch.load(
        RESNET50_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =====================================
# IMAGE TRANSFORM
# =====================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================
# INPUT IMAGE
# =====================================

image_path = input(
    "Enter image path: "
)


# =====================================
# LOAD IMAGE
# =====================================

image = Image.open(
    image_path
).convert("RGB")


input_tensor = transform(
    image
).unsqueeze(0)


input_tensor = input_tensor.to(device)


# =====================================
# PREDICTION
# =====================================

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()


# =====================================
# RESULTS
# =====================================

print("\n" + "=" * 50)

print("OSTEOPOROSIS PREDICTION REPORT (ResNet50)")

print("=" * 50)


print(
    f"\nPredicted Class: "
    f"{CLASS_NAMES[predicted_class]}"
)


print("\nConfidence Scores:\n")


for i, class_name in enumerate(CLASS_NAMES):

    confidence = probabilities[i].item() * 100

    print(
        f"{class_name}: "
        f"{confidence:.2f}%"
    )


print("\n" + "=" * 50)

print(
    "\nMedical Disclaimer:\n"
)

print(
    "This AI system is intended "
    "only as a decision-support tool "
    "and should not replace "
    "professional medical diagnosis."
)

print("=" * 50)
