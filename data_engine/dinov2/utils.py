import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms


def resize_image(image: Image.Image, target_size: int = 256, multiple_of: int = 14):
    width, height = image.size
    max_side = max(width, height)
    if max_side < target_size:
        scale = 1
    else:
        scale = target_size / max_side
    width, height = int(width * scale), int(height * scale)
    width = int(np.ceil(width / multiple_of) * multiple_of)
    height = int(np.ceil(height / multiple_of) * multiple_of)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    t_image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return normalize(t_image)
