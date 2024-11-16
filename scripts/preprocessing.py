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
        stain_correction: bool = False,
        stain_method: Optional[str] = None,
        mask_size: Tuple[int, int] = (32, 32),
        stain_reference_image_path: Optional[str] = None,
        img_dir: Optional[str] = None,
        sample_size: int = 50,
    ):
        """
        Preprocessing pipeline for histopathology images with support for
        stain normalization, center masking, and image transformations.

        Args:
            input_size (Tuple[int, int]): Target size for resizing images.
            stain_correction (bool): Apply stain correction if True.
            stain_method (str): Type of stain normalization ('macenko', 'reinhard', 'vahadane').
            mask_size (Tuple[int, int]): Size of the center mask to apply (default 32x32).
            stain_reference_image_path (str): Path to a reference image for stain normalization.
            img_dir (str): Directory of images for automatic reference selection.
            sample_size (int): Number of images to sample for reference selection.
        """
        self.input_size = input_size
        self.stain_correction = stain_correction
        self.stain_method = stain_method
        self.mask_size = mask_size
        self.transforms = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Automatically select stain reference image if not provided
        if self.stain_correction and not stain_reference_image_path and img_dir:
            stain_reference_image_path = self._select_reference_image(img_dir, sample_size)

        self.stain_reference_image_path = stain_reference_image_path
        self.stain_normalizer = None
        if self.stain_correction and self.stain_reference_image_path:
            self.stain_normalizer = self._initialize_stain_normalizer(self.stain_reference_image_path)

    def _initialize_stain_normalizer(self, reference_image_path: str):
        """Initialize the stain normalizer using a reference image."""
        try:
            import staintools  # Ensure staintools is installed
        except ImportError:
            raise ImportError("staintools is required for stain normalization. Install it with `pip install staintools`.")

        reference_image = staintools.read_image(reference_image_path)
        normalizer = staintools.StainNormalizer(method=self.stain_method)
        normalizer.fit(reference_image)
        return normalizer

    def _select_reference_image(self, img_dir: str, sample_size: int) -> str:
        """
        Automatically select a reference image based on average pixel values.

        Args:
            img_dir (str): Directory containing images.
            sample_size (int): Number of images to sample for selection.

        Returns:
            str: File path of the selected reference image.
        """
        image_files = os.listdir(img_dir)[:sample_size]
        image_means = []

        for img_file in image_files:
            img_path = os.path.join(img_dir, img_file)
            img = np.array(Image.open(img_path))
            image_means.append((img.mean(), img_file))

        # Find the image closest to the dataset's average mean
        dataset_mean = np.mean([mean for mean, _ in image_means])
        closest_img = min(image_means, key=lambda x: abs(x[0] - dataset_mean))[1]

        return os.path.join(img_dir, closest_img)

    def apply_stain_correction(self, img: np.array) -> np.array:
        """Apply stain normalization to the image."""
        if self.stain_correction and self.stain_normalizer:
            try:
                import staintools
            except ImportError:
                raise ImportError("staintools is required for stain normalization. Install it with `pip install staintools`.")

            img = staintools.LuminosityStandardizer.standardize(img)
            img = self.stain_normalizer.transform(img)
        return img

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
        img = self.apply_stain_correction(img)
        img = self.apply_center_mask(img)
        img = Image.fromarray(img)  # Convert back to PIL for transformations
        return self.transforms(img)
