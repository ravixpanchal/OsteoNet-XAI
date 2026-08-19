import os

import torch
import matplotlib.pyplot as plt

from PIL import Image

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

import matplotlib
import matplotlib.cm as cm
if not hasattr(cm, 'get_cmap'):
    cm.get_cmap = matplotlib.colormaps.get_cmap

from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

from src.config import (
    IMAGE_SIZE,
    CLASS_NAMES,
    RESNET18_PATH,
    GRADCAM_PATH
)

from src.models import (
    get_resnet18_model,
    get_device
)


# =====================================
# DEVICE
# =====================================

device = get_device()


# =====================================
# LOAD MODEL
# =====================================

model = get_resnet18_model()

model.load_state_dict(
    torch.load(
        RESNET18_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

for param in model.parameters():
    param.requires_grad = True


# =====================================
# GRADCAM EXTRACTOR
# =====================================

cam_extractor = SmoothGradCAMpp(
    model,
    target_layer="layer4"
)


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
# IMAGE PATH
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

output = model(input_tensor)

predicted_class = output.argmax(
    dim=1
).item()


confidence = torch.softmax(
    output,
    dim=1
)[0][predicted_class].item()


print("\nPrediction:")

print(
    CLASS_NAMES[predicted_class]
)

print(
    f"Confidence: {confidence*100:.2f}%"
)


# =====================================
# GENERATE CAM
# =====================================

activation_map = cam_extractor(
    predicted_class,
    output
)


cam = activation_map[0].squeeze(0)


# =====================================
# OVERLAY HEATMAP
# =====================================

result = overlay_mask(

    image,

    to_pil_image(
        cam,
        mode="F"
    ),

    alpha=0.5
)


# =====================================
# SAVE RESULT
# =====================================

os.makedirs(
    GRADCAM_PATH,
    exist_ok=True
)


save_path = (
    f"{GRADCAM_PATH}/gradcam_result.png"
)


result.save(save_path)


# =====================================
# DISPLAY
# =====================================

plt.figure(figsize=(8, 8))

plt.imshow(result)

plt.title(

    f"{CLASS_NAMES[predicted_class]} "
    f"({confidence*100:.2f}%)"

)

plt.axis("off")

plt.show()


print("\nGrad-CAM saved at:")

print(save_path)