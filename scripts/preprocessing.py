import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Optional, Tuple
import staintools

class Preprocessing:
    def __init__(self, 
                 input_size: Tuple[int, int] = (224, 224), 
                 stain_correction: bool = False, 
                 stain_method: Optional[str] = None, 
                 mask_size: Optional[Tuple[int, int]] = (32, 32),  # Default to 32x32
                 stain_reference_image_path: Optional[str] = None):
        """
        Preprocessing pipeline for histopathology images with support for 
        stain normalization, center masking, and separate transforms for 
        train, validation, and test datasets.

        Args:
            input_size (Tuple[int, int]): Target size for resizing images.
            stain_correction (bool): Apply stain correction if True.
            stain_method (str): Type of stain normalization ('macenko', 'reinhard', 'vahadane').
            mask_size (Tuple[int, int]): Size of the center mask to apply (default 32x32).
            stain_reference_image_path (str): Path to a reference image for stain normalization.
        """
        self.input_size = input_size
        self.stain_correction = stain_correction
        self.stain_method = stain_method
        self.mask_size = mask_size if mask_size else (32, 32)  # Ensure mask_size is always a tuple
        self.transforms = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.stain_normalizer = None
        if self.stain_correction and stain_reference_image_path:
            self.stain_normalizer = self._initialize_stain_normalizer(stain_reference_image_path)

    def _initialize_stain_normalizer(self, reference_image_path: str):
        """Initialize the stain normalizer using a reference image."""
        reference_image = staintools.read_image(reference_image_path)
        normalizer = staintools.StainNormalizer(method=self.stain_method)
        normalizer.fit(reference_image)
        return normalizer

    def apply_stain_correction(self, img: np.array) -> np.array:
        """Apply stain normalization to the image."""
        if self.stain_correction and self.stain_normalizer:
            img = staintools.LuminosityStandardizer.standardize(img)
            img = self.stain_normalizer.transform(img)
        return img

    def apply_center_mask(self, img: np.array) -> np.array:
        """Apply a center mask of size `mask_size` to the image."""
        h, w = img.shape[:2]
        mask_h, mask_w = self.mask_size
        center_h, center_w = h // 2, w // 2

        # Calculate start and end coordinates for the mask
        start_h = max(center_h - mask_h // 2, 0)
        start_w = max(center_w - mask_w // 2, 0)
        end_h = min(start_h + mask_h, h)
        end_w = min(start_w + mask_w, w)

        # Apply the center mask
        img = img[start_h:end_h, start_w:end_w]
        return img

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
