import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Optional, Tuple


class Preprocessing:
    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        mask_size: Tuple[int, int] = (32, 32),
    ):
        """
        Preprocessing pipeline for histopathology images with support for
        center masking and image transformations.

        Args:
            input_size (Tuple[int, int]): Target size for resizing images.
            mask_size (Tuple[int, int]): Size of the center mask to apply (default 32x32).
        """
        self.input_size = input_size
        self.mask_size = mask_size
        self.transforms = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def apply_center_mask(self, img: np.array) -> np.array:
        """
        Apply a center mask of size `mask_size` to the image.

        Args:
            img (np.array): Input image as a numpy array.

        Returns:
            np.array: Cropped image with center mask applied.
        """
        h, w = img.shape[:2]
        mask_h, mask_w = self.mask_size
        center_h, center_w = h // 2, w // 2

        # Calculate start and end coordinates for the mask
        start_h = max(center_h - mask_h // 2, 0)
        start_w = max(center_w - mask_w // 2, 0)
        end_h = min(start_h + mask_h, h)
        end_w = min(start_w + mask_w, w)

        return img[start_h:end_h, start_w:end_w]

    def preprocess_image(self, img: Image.Image, mode: str = "train") -> torch.Tensor:
        """
        Preprocess an image based on the mode (train, validation, or test).

        Args:
            img (PIL.Image.Image): Input image.
            mode (str): Preprocessing mode ('train', 'val', 'test').

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        img = np.array(img)

        # Apply center mask
        img = self.apply_center_mask(img)

        # Convert back to PIL for transformations
        img = Image.fromarray(img)

        # Apply standard transformations
        return self.transforms(img)

    def preprocess_batch(self, img_batch: list) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            img_batch (list): List of PIL.Image.Image objects.

        Returns:
            torch.Tensor: Batch of preprocessed image tensors.
        """
        return torch.stack([self.preprocess_image(img) for img in img_batch])


if __name__ == "__main__":
    # Example usage
    img_path = "example_image.tif"  # Replace with a valid image path
    img = Image.open(img_path).convert("RGB")
    preprocessor = Preprocessing(input_size=(96, 96), mask_size=(32, 32))
    preprocessed_img = preprocessor.preprocess_image(img)
    print("Preprocessed image shape:", preprocessed_img.shape)
