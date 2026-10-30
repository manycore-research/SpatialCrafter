from typing import Tuple

import torch
import PIL


def crop_image_to_target_ratio(image: PIL.Image, target_aspect_ratio: float = 4.0 / 3.0):
    """Crops an image to satisfy a target aspect ratio."""

    actual_aspect_ratio = image.width / image.height

    if actual_aspect_ratio > target_aspect_ratio:
        # we should crop width
        new_width = image.height * target_aspect_ratio

        left = (image.width - new_width) / 2
        top = 0
        right = (image.width + new_width) / 2
        bottom = image.height

        # Crop the center of the image
        image = image.crop((left, top, right, bottom))

    elif actual_aspect_ratio < target_aspect_ratio:
        # we should crop height
        new_height = image.width / target_aspect_ratio

        left = 0
        top = (image.height - new_height) / 2
        right = image.width
        bottom = (image.height + new_height) / 2

        # Crop the center of the image
        image = image.crop((left, top, right, bottom))

    return image

def resize_pil_image(
    img: PIL.Image,
    target_height: int = None,
    target_width: int = None,
    resampling_mode=PIL.Image.BILINEAR,
    target_aspect_ratio: float = None,
) -> PIL.Image:
    """ " Reads an image file using PIL, then optionally resizes the image,
    with selective resampling, scales values, and returns the image as a
    tensor

    Args:
        img: PIL Image to be resized.
        target_height, target_width: resolution to resize the image to. Both must not be
            None for scaling to take place.
        resampling_mode: resampling method when resizing using PIL. Default
            is PIL.Image.BILINEAR
        target_aspect_ratio: if not None, will crop the image to match this aspect ratio. Default is None

    Returns:
        img: resized image.

    """

    if target_aspect_ratio:
        img = crop_image_to_target_ratio(img, target_aspect_ratio)

    # resize if both width and height are not none.
    if target_height is not None and target_width is not None:
        img_width, img_height = img.size
        # do we really need to resize? If not, skip.
        if (img_width, img_height) != (target_width, target_height):
            # warn if it doesn't make sense.
            if (target_width > img_width or target_height > img_height):
                print(f"WARNING: target size ({target_width}x{target_height}) is larger than input size ({img_width}x{img_height}).")
            img = img.resize((target_width, target_height), resample=resampling_mode)

    return img
